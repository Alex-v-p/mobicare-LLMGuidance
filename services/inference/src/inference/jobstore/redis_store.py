from __future__ import annotations

import json
import os
from typing import Optional

from redis.asyncio import Redis

from shared.contracts.inference import JobRecord, utc_now_iso


class RedisJobStore:
    def __init__(self, redis_url: str | None = None, queue_name: str | None = None, ttl_seconds: int | None = None) -> None:
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/0")
        self._queue_name = queue_name or os.getenv("REDIS_JOB_QUEUE", "guidance_jobs")
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else int(os.getenv("JOB_TTL_SECONDS", str(7 * 24 * 60 * 60)))
        self._redis: Optional[Redis] = None

    async def _client(self) -> Redis:
        if self._redis is None:
            self._redis = Redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    def _job_key(self, request_id: str) -> str:
        return f"job:{request_id}"

    async def create(self, record: JobRecord) -> None:
        redis = await self._client()
        key = self._job_key(record.request_id)
        created = await redis.set(key, record.model_dump_json(), ex=self._ttl_seconds, nx=True)
        if not created:
            raise FileExistsError(f"Job {record.request_id} already exists")
        await redis.rpush(self._queue_name, record.request_id)

    async def get(self, request_id: str) -> Optional[JobRecord]:
        redis = await self._client()
        payload = await redis.get(self._job_key(request_id))
        if payload is None:
            return None
        return JobRecord.model_validate_json(payload)

    async def update(self, record: JobRecord) -> None:
        redis = await self._client()
        record.updated_at = utc_now_iso()
        await redis.set(self._job_key(record.request_id), record.model_dump_json(), ex=self._ttl_seconds)

    async def claim_next(self, timeout_s: int = 5) -> Optional[JobRecord]:
        redis = await self._client()
        popped = await redis.blpop(self._queue_name, timeout=timeout_s)
        if popped is None:
            return None

        _queue, request_id = popped
        record = await self.get(request_id)
        if record is None:
            return None

        if record.status != "queued":
            return record

        record.status = "running"
        record.started_at = record.started_at or utc_now_iso()
        record.updated_at = utc_now_iso()
        await self.update(record)
        return record
