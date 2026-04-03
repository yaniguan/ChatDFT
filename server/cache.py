"""
Caching Layer
=============
Redis-backed cache with automatic in-memory fallback.

Usage:
    from server.cache import cache

    # Store embedding
    await cache.set("embed:abc123", vector_as_json, ttl=3600)
    result = await cache.get("embed:abc123")

    # Stats
    print(cache.stats())
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class CacheBackend:
    """Abstract cache interface."""

    async def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    async def set(self, key: str, value: str, ttl: int = 0) -> None:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        return {}


class RedisBackend(CacheBackend):
    """Redis-backed cache."""

    def __init__(self, url: str):
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(url, decode_responses=True)
        self._available = True
        log.info("Redis cache connected: %s", url.split("@")[-1])

    async def get(self, key: str) -> Optional[str]:
        try:
            return await self._redis.get(key)
        except Exception:
            return None

    async def set(self, key: str, value: str, ttl: int = 0) -> None:
        try:
            if ttl > 0:
                await self._redis.setex(key, ttl, value)
            else:
                await self._redis.set(key, value)
        except Exception as e:
            log.warning("Redis set failed: %s", e)

    async def delete(self, key: str) -> None:
        try:
            await self._redis.delete(key)
        except Exception:
            pass

    async def exists(self, key: str) -> bool:
        try:
            return bool(await self._redis.exists(key))
        except Exception:
            return False

    def stats(self) -> Dict[str, Any]:
        return {"backend": "redis", "available": self._available}


class InMemoryBackend(CacheBackend):
    """In-memory LRU cache with TTL support."""

    def __init__(self, max_size: int = 10000):
        self._store: Dict[str, tuple] = {}  # key → (value, expire_at)
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        value, expire_at = entry
        if expire_at > 0 and time.time() > expire_at:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return value

    async def set(self, key: str, value: str, ttl: int = 0) -> None:
        expire_at = time.time() + ttl if ttl > 0 else 0
        self._store[key] = (value, expire_at)
        # Evict oldest if over capacity
        if len(self._store) > self._max_size:
            keys = list(self._store.keys())[:self._max_size // 5]
            for k in keys:
                self._store.pop(k, None)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._store

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "backend": "memory",
            "size": len(self._store),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / max(total, 1):.1%}",
        }


class Cache:
    """
    Unified cache with Redis primary + in-memory fallback.

    Set REDIS_URL environment variable to enable Redis.
    Falls back to in-memory cache if Redis is unavailable.
    """

    def __init__(self):
        redis_url = os.environ.get("REDIS_URL", "")
        if redis_url:
            try:
                self._backend = RedisBackend(redis_url)
                self._type = "redis"
            except Exception as e:
                log.warning("Redis unavailable (%s), using in-memory cache", e)
                self._backend = InMemoryBackend()
                self._type = "memory"
        else:
            self._backend = InMemoryBackend()
            self._type = "memory"
            log.info("Cache: in-memory (set REDIS_URL for Redis)")

    async def get(self, key: str) -> Optional[str]:
        return await self._backend.get(key)

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        await self._backend.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        await self._backend.delete(key)

    async def get_json(self, key: str) -> Optional[Any]:
        raw = await self.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set_json(self, key: str, value: Any, ttl: int = 3600) -> None:
        await self.set(key, json.dumps(value, default=str), ttl)

    def stats(self) -> Dict[str, Any]:
        return self._backend.stats()


# Singleton
cache = Cache()
