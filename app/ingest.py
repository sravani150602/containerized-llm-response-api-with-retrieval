"""Turn raw documents into retrievable text chunks.

Two kinds of sources are supported:

* Unstructured  (.md, .txt)  -> split by Markdown section, then into
  overlapping windows so each chunk keeps its heading for context.
* Structured    (.csv, .json) -> every CSV row / JSON leaf group becomes a
  small self-describing "record" sentence, e.g.
  "holidays_2026 | holiday: Labor Day | date: 2026-09-07 | ...".
  Writing records as text lets the same retriever and the same LLM prompt
  handle tables and prose.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath


@dataclass
class Chunk:
    chunk_id: str
    source: str  # document key, e.g. "unstructured/employee_handbook.md"
    title: str  # section heading or record label
    text: str
    kind: str  # "unstructured" | "structured"

    def to_dict(self) -> dict:
        return asdict(self)


def _cid(source: str, i: int, text: str) -> str:
    h = hashlib.sha1(f"{source}:{i}:{text}".encode()).hexdigest()[:10]
    return f"{PurePosixPath(source).stem}-{i}-{h}"


# --------------------------------------------------------------------------- #
# Unstructured
# --------------------------------------------------------------------------- #
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs. Doc title is prefixed to headings."""
    doc_title, current, buf, out = "", "", [], []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            if buf and "".join(buf).strip():
                out.append((current, "\n".join(buf).strip()))
            buf = []
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1:
                doc_title = heading
                current = heading
            else:
                current = f"{doc_title} > {heading}" if doc_title else heading
        elif not line.startswith(">"):  # skip disclaimer blockquotes
            buf.append(line)
    if buf and "".join(buf).strip():
        out.append((current, "\n".join(buf).strip()))
    return out


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows, preferring sentence boundaries."""
    if len(text) <= size:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > size and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur += " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def chunk_unstructured(source: str, text: str, size: int = 700, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _split_sections(text):
        for piece in _window(body, size, overlap):
            i = len(chunks)
            chunks.append(Chunk(_cid(source, i, piece), source, heading, piece, "unstructured"))
    return chunks


# --------------------------------------------------------------------------- #
# Structured
# --------------------------------------------------------------------------- #
def _record_text(table: str, row: dict) -> str:
    fields = " | ".join(f"{k.replace('_', ' ')}: {v}" for k, v in row.items() if str(v).strip())
    return f"{table.replace('_', ' ')} | {fields}"


def chunk_csv(source: str, text: str) -> list[Chunk]:
    table = PurePosixPath(source).stem
    chunks = []
    for i, row in enumerate(csv.DictReader(io.StringIO(text))):
        body = _record_text(table, row)
        label = f"{table} row {i + 1}: {next(iter(row.values()), '')}"
        chunks.append(Chunk(_cid(source, i, body), source, label, body, "structured"))
    return chunks


def _flatten(obj, prefix: str = "") -> dict:
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        for i, v in enumerate(obj):
            flat.update(_flatten(v, f"{prefix}{i}."))
    elif isinstance(obj, list):
        flat[prefix.rstrip(".")] = ", ".join(map(str, obj))
    else:
        flat[prefix.rstrip(".")] = obj
    return flat


def chunk_json(source: str, text: str) -> list[Chunk]:
    """One chunk per top-level key (or per list item), with nested keys flattened."""
    table = PurePosixPath(source).stem
    data = json.loads(text)
    groups: list[tuple[str, object]] = []
    if isinstance(data, dict):
        scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
        if scalars:
            groups.append(("general", scalars))
        for k, v in data.items():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                groups.extend((f"{k} {i + 1}", item) for i, item in enumerate(v))
            elif isinstance(v, (dict, list)):
                groups.append((k, {k: v}))
    else:
        groups = [(f"item {i + 1}", item) for i, item in enumerate(data)]

    chunks = []
    for i, (label, value) in enumerate(groups):
        flat = _flatten(value)
        body = _record_text(table, flat)
        chunks.append(Chunk(_cid(source, i, body), source, f"{table}: {label}", body, "structured"))
    return chunks


# --------------------------------------------------------------------------- #
def chunk_document(source: str, raw: bytes, size: int = 700, overlap: int = 120) -> list[Chunk]:
    text = raw.decode("utf-8", errors="replace")
    suffix = PurePosixPath(source).suffix.lower()
    if suffix == ".csv":
        return chunk_csv(source, text)
    if suffix == ".json":
        return chunk_json(source, text)
    return chunk_unstructured(source, text, size, overlap)
