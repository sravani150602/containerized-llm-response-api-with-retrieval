"""Central configuration.

Every setting can be overridden with an environment variable of the same name
(case-insensitive) or through a `.env` file. See `.env.example`.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- LLM ---------------------------------------------------------------
    # "hf"   -> real Hugging Face model (+ optional LoRA adapter)
    # "echo" -> tiny extractive stand-in used ONLY by unit tests / CI (no torch)
    llm_backend: str = "hf"
    base_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    adapter_path: str = "artifacts/lora-adapter"  # empty string disables LoRA
    merge_adapter: bool = True  # merge LoRA weights into the base model for faster inference
    max_new_tokens: int = 96
    torch_threads: int = 0  # 0 = let torch decide

    # ---- Retrieval ---------------------------------------------------------
    top_k: int = 4
    chunk_size: int = 700  # characters
    chunk_overlap: int = 120
    min_retrieval_score: float = 0.0

    # ---- Storage (S3) ------------------------------------------------------
    # "s3"    -> AWS S3 or any S3-compatible store (MinIO in docker-compose)
    # "local" -> read documents straight from ./knowledge_base (no S3 needed)
    storage_backend: str = "local"
    s3_bucket: str = "llm-rag-knowledge-base"
    s3_prefix: str = "kb/"
    s3_endpoint_url: str | None = None  # e.g. http://minio:9000 ; None = real AWS
    aws_region: str = "us-east-1"
    local_kb_path: str = "knowledge_base"

    # ---- Cache (Redis) -----------------------------------------------------
    redis_url: str | None = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True

    # ---- Service -----------------------------------------------------------
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
