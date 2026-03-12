from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import redis.asyncio as redis

from shared.config import Settings, get_settings
from shared.contracts.inference import JobRecord, utc_now_iso


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


class RedisGuidanceJobStore:
    def __init__(
        self,
        redis_url: str | None = None,
        queue_name: str | None = None,
        ttl_seconds: int | None = None,
        lease_seconds: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._redis_url = redis_url or self._settings.redis_url
        self._queue_name = queue_name or self._settings.redis_job_queue
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else self._settings.job_ttl_seconds
        self._lease_seconds = lease_seconds if lease_seconds is not None else self._settings.job_lease_seconds
        self._redis: Optional[redis.Redis] = None

    async def _client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    def _job_key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def _lease_expiry_iso(self) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=self._lease_seconds)).isoformat()

    async def create(self, record: JobRecord) -> None:
        client = await self._client()
        key = self._job_key(record.job_id)
        created = await client.set(key, record.model_dump_json(), ex=self._ttl_seconds, nx=True)
        if not created:
            raise FileExistsError(f"Job {record.job_id} already exists")
        await client.rpush(self._queue_name, record.job_id)

    async def get(self, job_id: str) -> Optional[JobRecord]:
        client = await self._client()
        payload = await client.get(self._job_key(job_id))
        if payload is None:
            return None
        return JobRecord.model_validate_json(payload)

    async def update(self, record: JobRecord) -> None:
        client = await self._client()
        record.updated_at = utc_now_iso()
        await client.set(self._job_key(record.job_id), record.model_dump_json(), ex=self._ttl_seconds)

    async def heartbeat(self, job_id: str, worker_id: str) -> bool:
        record = await self.get(job_id)
        if record is None or record.status != "running" or record.worker_id != worker_id:
            return False
        record.lease_expires_at = self._lease_expiry_iso()
        await self.update(record)
        return True

    async def requeue_stale_running_jobs(self) -> int:
        client = await self._client()
        count = 0
        async for key in client.scan_iter(match="job:*"):
            payload = await client.get(key)
            if payload is None:
                continue
            record = JobRecord.model_validate_json(payload)
            if record.status != "running":
                continue
            lease_expires_at = _parse_iso(record.lease_expires_at)
            if lease_expires_at is None or lease_expires_at > datetime.now(timezone.utc):
                continue
            record.status = "queued"
            record.worker_id = None
            record.lease_expires_at = None
            record.error = "Worker lease expired; job re-queued automatically."
            await self.update(record)
            await client.rpush(self._queue_name, record.job_id)
            count += 1
        return count

    async def claim_next(self, worker_id: str, timeout_s: int = 5) -> Optional[JobRecord]:
        client = await self._client()
        await self.requeue_stale_running_jobs()
        popped = await client.blpop(self._queue_name, timeout=timeout_s)
        if popped is None:
            return None

        _queue, job_id = popped
        record = await self.get(job_id)
        if record is None:
            return None
        if record.status != "queued":
            return record

        record.status = "running"
        record.worker_id = worker_id
        record.started_at = record.started_at or utc_now_iso()
        record.lease_expires_at = self._lease_expiry_iso()
        await self.update(record)
        return record
