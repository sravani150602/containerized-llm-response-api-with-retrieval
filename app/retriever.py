"""Hybrid lexical retriever: BM25 + TF-IDF cosine, merged with Reciprocal Rank Fusion.

Why lexical and not dense embeddings? Workplace questions are full of exact
terms ("PTO", "SEV1", "YubiKey", "$75"), which BM25 matches very well, and it
keeps the container small and CPU-friendly. Two scorers are fused because
BM25 favours exact keyword hits while TF-IDF with bigrams favours phrases;
RRF combines rankings without needing to calibrate their score scales.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from app.ingest import Chunk

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "and", "or", "in", "on", "for",
    "with", "at", "by", "from", "as", "it", "this", "that", "do", "does", "did", "i", "my", "me",
    "we", "our", "you", "your", "can", "how", "what", "when", "where", "which", "who", "many", "much",
    "per", "if", "am", "get", "there", "any", "should", "would", "will", "need", "about", "have", "has",
}


def _norm(word: str) -> str:
    # tiny stemmer: good enough to match "days"/"day", "approvals"/"approval"
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9$#][a-z0-9$#\-\.]*[a-z0-9]|[a-z0-9]", text.lower())
    return [_norm(w) for w in words if w not in _STOP]


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class HybridRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
        self.k1, self.b, self.rrf_k = k1, b, rrf_k
        self.chunks: list[Chunk] = []
        self.version = "empty"

    # ------------------------------------------------------------------ build
    def build(self, chunks: list[Chunk]) -> str:
        self.chunks = list(chunks)
        corpus = [f"{c.title}\n{c.text}" for c in self.chunks]
        if not corpus:
            self.version = "empty"
            return self.version

        # BM25 statistics
        self._count = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self._count.fit_transform(corpus).tocsc().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.diff(tf.indptr)  # document frequency per term (CSC column counts)
        self._idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        self._len_norm = self.k1 * (1 - self.b + self.b * doc_len / doc_len.mean())
        self._tf = tf  # CSC: fast column (term) access at query time

        # TF-IDF with bigrams
        self._tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                      ngram_range=(1, 2), sublinear_tf=True)
        self._tfidf_matrix = self._tfidf.fit_transform(corpus)

        digest = hashlib.sha1("".join(c.chunk_id for c in self.chunks).encode()).hexdigest()[:12]
        self.version = digest
        return digest

    # ----------------------------------------------------------------- scoring
    def _bm25(self, query: str) -> np.ndarray:
        q = self._count.transform([query])
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in q.indices:
            start, end = self._tf.indptr[term], self._tf.indptr[term + 1]
            rows, col_tf = self._tf.indices[start:end], self._tf.data[start:end]
            scores[rows] += self._idf[term] * col_tf * (self.k1 + 1) / (col_tf + self._len_norm[rows])
        return scores

    def _cosine(self, query: str) -> np.ndarray:
        q = self._tfidf.transform([query])
        return (self._tfidf_matrix @ q.T).toarray().ravel()

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        bm25, cos = self._bm25(query), self._cosine(query)
        if bm25.max() <= 0 and cos.max() <= 0:
            return []
        fused = np.zeros(len(self.chunks))
        for scores in (bm25, cos):
            ranks = np.argsort(-scores)
            for rank, idx in enumerate(ranks):
                if scores[idx] > 0:
                    fused[idx] += 1.0 / (self.rrf_k + rank + 1)
        order = np.argsort(-fused)[:top_k]
        return [ScoredChunk(self.chunks[i], round(float(fused[i]), 5)) for i in order if fused[i] > min_score]
