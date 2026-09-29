"""The RAG pipeline: cache lookup -> retrieval -> prompt -> LLM -> cache store."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from app import metrics
from app.cache import make_key, normalize_question
from app.config import Settings
from app.ingest import Chunk, chunk_document
from app.prompts import build_messages
from app.retriever import HybridRetriever, ScoredChunk
from app.schemas import ChatRequest, ChatResponse, Source, Timings

log = logging.getLogger(__name__)


def _to_source(sc: ScoredChunk) -> Source:
    c = sc.chunk
    return Source(chunk_id=c.chunk_id, source=c.source, title=c.title, score=sc.score, text=c.text)


class RAGService:
    def __init__(self, settings: Settings, store, cache, engine):
        self.settings = settings
        self.store = store
        self.cache = cache
        self.engine = engine
        self.retriever = HybridRetriever()

    # ------------------------------------------------------------- indexing
    def reindex(self) -> tuple[int, int, str]:
        t0 = time.perf_counter()
        docs = self.store.list_documents()
        chunks: list[Chunk] = []
        for d in docs:
            try:
                chunks.extend(chunk_document(d.key, self.store.read(d.key), self.settings.chunk_size,
                                             self.settings.chunk_overlap))
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the index
                log.error("Failed to ingest %s: %s", d.key, exc)
        version = self.retriever.build(chunks)
        metrics.CHUNKS_INDEXED.set(len(chunks))
        log.info("Indexed %d documents / %d chunks in %.0f ms (version %s)",
                 len(docs), len(chunks), (time.perf_counter() - t0) * 1000, version)
        return len(docs), len(chunks), version

    # ------------------------------------------------------------ retrieval
    def retrieve(self, question: str, top_k: int) -> list[ScoredChunk]:
        with metrics.STAGE_LATENCY.labels("retrieval").time():
            return self.retriever.search(question, top_k, self.settings.min_retrieval_score)

    def _messages(self, req: ChatRequest, hits: list[ScoredChunk] | None):
        passages = None if hits is None else [(h.chunk.title, h.chunk.text) for h in hits]
        return build_messages(req.question, passages)

    def _cache_key(self, req: ChatRequest, top_k: int, max_new: int) -> str:
        payload = {"q": normalize_question(req.question), "rag": req.use_rag, "k": top_k,
                   "max": max_new, "model": self.engine.name, "adapter": self.engine.adapter}
        return make_key("answer", self.retriever.version, payload)

    # ---------------------------------------------------------------- answer
    def answer(self, req: ChatRequest) -> ChatResponse:
        t_start = time.perf_counter()
        top_k = req.top_k or self.settings.top_k
        max_new = req.max_new_tokens or self.settings.max_new_tokens
        use_cache = req.use_cache and self.settings.cache_enabled

        key = self._cache_key(req, top_k, max_new)
        if use_cache and (hit := self.cache.get(key)):
            metrics.CACHE_EVENTS.labels("hit").inc()
            hit["cached"] = True
            hit["timings"] = Timings(total_ms=round((time.perf_counter() - t_start) * 1000, 2))
            return ChatResponse(**hit)
        metrics.CACHE_EVENTS.labels("miss").inc()

        t0 = time.perf_counter()
        hits = self.retrieve(req.question, top_k) if req.use_rag else None
        retrieval_ms = (time.perf_counter() - t0) * 1000

        with metrics.STAGE_LATENCY.labels("generation").time():
            gen = self.engine.generate(self._messages(req, hits), max_new)
        metrics.TOKENS_GENERATED.inc(gen.new_tokens)

        resp = ChatResponse(
            answer=gen.text,
            sources=[_to_source(h) for h in hits or []],
            cached=False,
            model=self.engine.name,
            adapter=self.engine.adapter,
            timings=Timings(retrieval_ms=round(retrieval_ms, 2), generation_ms=round(gen.seconds * 1000, 2),
                            total_ms=round((time.perf_counter() - t_start) * 1000, 2)),
        )
        if use_cache:
            self.cache.set(key, resp.model_dump(exclude={"timings", "cached"}), self.settings.cache_ttl_seconds)
        return resp

    def stream(self, req: ChatRequest) -> Iterator[tuple[str, dict]]:
        """Yields (event, data) tuples: `sources`, many `token`, then `done`."""
        t_start = time.perf_counter()
        top_k = req.top_k or self.settings.top_k
        max_new = req.max_new_tokens or self.settings.max_new_tokens
        hits = self.retrieve(req.question, top_k) if req.use_rag else None
        yield "sources", {"sources": [_to_source(h).model_dump() for h in hits or []]}

        first_token_ms = None
        parts = []
        for piece in self.engine.stream(self._messages(req, hits), max_new):
            if not piece:
                continue
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - t_start) * 1000
                metrics.TTFT.observe(first_token_ms / 1000)
            parts.append(piece)
            yield "token", {"text": piece}
        yield "done", {"answer": "".join(parts).strip(), "time_to_first_token_ms": round(first_token_ms or 0, 2),
                       "total_ms": round((time.perf_counter() - t_start) * 1000, 2)}
