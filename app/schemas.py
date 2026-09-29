"""Request / response models (these also drive the Swagger UI at /docs)."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000, examples=["How many PTO days do I get per year?"])
    use_rag: bool = Field(True, description="Retrieve context from the knowledge base before answering.")
    top_k: int | None = Field(None, ge=1, le=10, description="Number of chunks to retrieve (default from settings).")
    max_new_tokens: int | None = Field(None, ge=8, le=512)
    use_cache: bool = Field(True, description="Serve/store the answer in the Redis response cache.")


class Source(BaseModel):
    chunk_id: str
    source: str
    title: str
    score: float
    text: str


class Timings(BaseModel):
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    cached: bool = False
    model: str
    adapter: str | None = None
    timings: Timings


class SearchResponse(BaseModel):
    query: str
    results: list[Source]


class DocumentInfo(BaseModel):
    key: str
    size_bytes: int


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    index_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str
    adapter: str | None
    cache: str
    storage: str
    index_version: str
    chunks_indexed: int
