"""Document storage.

The knowledge base lives in an S3 bucket in production (AWS S3) and in MinIO
(an S3-compatible server) when you run `docker compose up`. For quick local
development the same interface can read straight from the ./knowledge_base
folder, so you do not need S3 to try the project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import Settings

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".csv", ".json"}


@dataclass
class StoredDocument:
    key: str
    size_bytes: int


class DocumentStore(Protocol):
    name: str

    def list_documents(self) -> list[StoredDocument]: ...
    def read(self, key: str) -> bytes: ...
    def write(self, key: str, data: bytes) -> None: ...
    def healthy(self) -> bool: ...


class LocalDocumentStore:
    """Reads/writes files under a local folder (default: ./knowledge_base)."""

    name = "local"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_documents(self) -> list[StoredDocument]:
        docs = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                docs.append(StoredDocument(p.relative_to(self.root).as_posix(), p.stat().st_size))
        return docs

    def _safe_path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"Invalid document key: {key}")
        return path

    def read(self, key: str) -> bytes:
        return self._safe_path(key).read_bytes()

    def write(self, key: str, data: bytes) -> None:
        path = self._safe_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def healthy(self) -> bool:
        return self.root.exists()


class S3DocumentStore:
    """Stores documents in an S3 bucket under a prefix (works with AWS S3 and MinIO)."""

    name = "s3"

    def __init__(self, bucket: str, prefix: str = "kb/", endpoint_url: str | None = None, region: str = "us-east-1"):
        import boto3  # imported lazily so local mode does not need boto3 configured

        self.bucket = bucket
        self.prefix = prefix
        self.client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

    def list_documents(self) -> list[StoredDocument]:
        docs: list[StoredDocument] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"][len(self.prefix):]
                if Path(key).suffix.lower() in SUPPORTED_SUFFIXES:
                    docs.append(StoredDocument(key, obj["Size"]))
        return sorted(docs, key=lambda d: d.key)

    def read(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"].read()

    def write(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self.prefix + key, Body=data)

    def healthy(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:  # noqa: BLE001
            return False


def build_store(settings: Settings) -> DocumentStore:
    if settings.storage_backend == "s3":
        log.info("Using S3 document store: s3://%s/%s", settings.s3_bucket, settings.s3_prefix)
        return S3DocumentStore(settings.s3_bucket, settings.s3_prefix, settings.s3_endpoint_url, settings.aws_region)
    log.info("Using local document store: %s", settings.local_kb_path)
    return LocalDocumentStore(settings.local_kb_path)
