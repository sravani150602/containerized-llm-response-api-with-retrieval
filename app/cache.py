"""Response + retrieval cache backed by Redis.

Keys include the index version, so re-ingesting documents automatically
invalidates stale answers. If Redis is unreachable the service keeps working
with an in-process TTL cache and reports `cache: memory` on /health.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict

log = logging.getLogger(__name__)


def make_key(namespace: str, index_version: str, payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:32]
    return f"llmrag:{namespace}:{index_version}:{digest}"


def normalize_question(q: str) -> str:
    return " ".join(q.lower().strip().rstrip("?!. ").split())


class MemoryCache:
    name = "memory"

    def __init__(self, max_items: int = 2048):
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.max_items = max_items

    def get(self, key: str) -> dict | None:
        item = self._data.get(key)
        if not item:
            return None
        expires, value = item
        if expires < time.time():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return json.loads(value)

    def set(self, key: str, value: dict, ttl: int) -> None:
        self._data[key] = (time.time() + ttl, json.dumps(value))
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def healthy(self) -> bool:
        return True


class RedisCache:
    name = "redis"

    def __init__(self, url: str):
        import redis

        self.client = redis.Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)
        self.client.ping()

    def get(self, key: str) -> dict | None:
        try:
            raw = self.client.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:  # noqa: BLE001 - cache must never break a request
            log.warning("redis get failed: %s", exc)
            return None

    def set(self, key: str, value: dict, ttl: int) -> None:
        try:
            self.client.set(key, json.dumps(value), ex=ttl)
        except Exception as exc:  # noqa: BLE001
            log.warning("redis set failed: %s", exc)

    def clear(self) -> None:
        for key in self.client.scan_iter("llmrag:*"):
            self.client.delete(key)

    def healthy(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:  # noqa: BLE001
            return False


def build_cache(redis_url: str | None):
    if redis_url:
        try:
            cache = RedisCache(redis_url)
            log.info("Connected to Redis at %s", redis_url)
            return cache
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis unavailable (%s); falling back to in-memory cache", exc)
    return MemoryCache()
