"""FastAPI application: the public REST API.

Endpoints
---------
GET  /                  minimal chat web UI (for demos)
GET  /health            liveness + dependency status
POST /v1/chat           answer a question (JSON)
POST /v1/chat/stream    answer a question token-by-token (Server-Sent Events)
GET  /v1/search         debug retrieval: which chunks match a query
GET  /v1/documents      list knowledge-base documents in storage (S3 / local)
POST /v1/documents      upload a new document, then re-index
POST /v1/index/rebuild  re-read storage and rebuild the retrieval index
DELETE /v1/cache        clear cached answers
GET  /metrics           Prometheus metrics
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import __version__, metrics
from app.cache import build_cache
from app.config import get_settings
from app.llm import build_engine
from app.rag import RAGService, _to_source
from app.schemas import (ChatRequest, ChatResponse, DocumentInfo, HealthResponse, IngestResponse,
                         SearchResponse)
from app.storage import SUPPORTED_SUFFIXES, build_store

STATIC = Path(__file__).parent / "static"


def create_app(service: RAGService | None = None) -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            app.state.service = RAGService(settings, build_store(settings), build_cache(settings.redis_url),
                                           build_engine(settings))
        app.state.service.reindex()
        yield

    app = FastAPI(
        title="Containerized LLM Response API with Retrieval",
        version=__version__,
        description="LoRA fine-tuned open-source LLM + hybrid retrieval (BM25/TF-IDF) over structured and "
                    "unstructured workplace documents, with Redis caching and S3 document storage.",
        lifespan=lifespan,
    )
    app.state.service = service

    def svc(request: Request) -> RAGService:
        return request.app.state.service

    @app.middleware("http")
    async def timing_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
        return response

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC / "index.html")

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request):
        s = svc(request)
        return HealthResponse(
            status="ok", model_loaded=s.engine.loaded, model=s.engine.name, adapter=s.engine.adapter,
            cache=s.cache.name if s.cache.healthy() else "down",
            storage=s.store.name if s.store.healthy() else "down",
            index_version=s.retriever.version, chunks_indexed=len(s.retriever.chunks),
        )

    @app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
    def chat(req: ChatRequest, request: Request):
        t0 = time.perf_counter()
        resp = svc(request).answer(req)
        metrics.REQUEST_LATENCY.labels("chat", str(resp.cached).lower()).observe(time.perf_counter() - t0)
        return resp

    @app.post("/v1/chat/stream", tags=["chat"], response_class=StreamingResponse,
              responses={200: {"content": {"text/event-stream": {}}}})
    def chat_stream(req: ChatRequest, request: Request):
        service = svc(request)

        def events():
            for event, data in service.stream(req):
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v1/search", response_model=SearchResponse, tags=["retrieval"])
    def search(request: Request, q: str = Query(..., min_length=2), top_k: int = Query(4, ge=1, le=20)):
        hits = svc(request).retrieve(q, top_k)
        return SearchResponse(query=q, results=[_to_source(h) for h in hits])

    @app.get("/v1/documents", response_model=list[DocumentInfo], tags=["documents"])
    def list_documents(request: Request):
        return [DocumentInfo(key=d.key, size_bytes=d.size_bytes) for d in svc(request).store.list_documents()]

    @app.post("/v1/documents", response_model=IngestResponse, tags=["documents"])
    async def upload_document(request: Request, file: UploadFile = File(...), folder: str = "uploads"):
        name = PurePosixPath(file.filename or "").name
        if PurePosixPath(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(400, f"Unsupported file type. Allowed: {sorted(SUPPORTED_SUFFIXES)}")
        data = await file.read()
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(413, "File larger than 5 MB")
        service = svc(request)
        service.store.write(f"{folder.strip('/')}/{name}", data)
        docs, chunks, version = service.reindex()
        return IngestResponse(documents=docs, chunks=chunks, index_version=version)

    @app.post("/v1/index/rebuild", response_model=IngestResponse, tags=["documents"])
    def rebuild(request: Request):
        docs, chunks, version = svc(request).reindex()
        return IngestResponse(documents=docs, chunks=chunks, index_version=version)

    @app.delete("/v1/cache", tags=["ops"])
    def clear_cache(request: Request):
        svc(request).cache.clear()
        return {"cleared": True}

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
