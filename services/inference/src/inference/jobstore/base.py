from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Callable, Generic, Optional, Protocol, TypeVar

import redis.asyncio as redis
from pydantic import BaseModel

from shared.config import Settings, get_settings

JobRecordT = TypeVar("JobRecordT", bound=BaseModel)

class ReadWriteJobStore(Protocol[JobRecordT]):
    async def create(self, record: JobRecordT) -> None: ...
    async def get(self, job_id: str) -> Optional[JobRecordT]: ...
    async def update(self, record: JobRecordT) -> None: ...
    async def close(self) -> None: ...


StoreT = TypeVar("StoreT", bound=ReadWriteJobStore)
StoreFactory = Callable[[], StoreT]


@asynccontextmanager
async def managed_store(factory: StoreFactory[StoreT]):
    store = factory()
    try:
        yield store
    finally:
        await store.close()


_ALLOWED_TRANSITIONS = {
    "queued": {"running", "failed"},
    "running": {"queued", "completed", "failed"},
    "completed": set(),
    "failed": set(),
}


class RedisJobStoreBase(Generic[JobRecordT]):
    def __init__(
        self,
        *,
        model_cls: type[JobRecordT],
        redis_url: str,
        queue_name: str,
        key_prefix: str,
        key_pattern: str,
        ttl_seconds: int,
        lease_seconds: int,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_cls = model_cls
        self._redis_url = redis_url
        self._queue_name = queue_name
        self._key_prefix = key_prefix
        self._key_pattern = key_pattern
        self._ttl_seconds = ttl_seconds
        self._lease_seconds = lease_seconds
        self._redis: Optional[redis.Redis] = None

    async def _client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    def _job_id(self, record: JobRecordT) -> str:
        return getattr(record, "job_id")

    def _job_key(self, job_id: str) -> str:
        return f"{self._key_prefix}{job_id}"

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _parse_iso(self, value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value)

    def _lease_expiry_iso(self) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=self._lease_seconds)).isoformat()

    def _validate_transition(self, current: str, target: str) -> None:
        if current == target:
            return
        allowed_targets = _ALLOWED_TRANSITIONS.get(current, set())
        if target not in allowed_targets:
            raise ValueError(f"Invalid job state transition: {current} -> {target}")

    def _set_updated_at(self, record: JobRecordT) -> None:
        setattr(record, "updated_at", self._utc_now_iso())

    def _transition_record(self, record: JobRecordT, target_status: str, **changes: object) -> JobRecordT:
        current_status = getattr(record, "status")
        self._validate_transition(current_status, target_status)
        setattr(record, "status", target_status)
        for field_name, value in changes.items():
            setattr(record, field_name, value)
        self._set_updated_at(record)
        return record

    async def create(self, record: JobRecordT) -> None:
        client = await self._client()
        key = self._job_key(self._job_id(record))
        created = await client.set(key, record.model_dump_json(), ex=self._ttl_seconds, nx=True)
        if not created:
            raise FileExistsError(f"Job {self._job_id(record)} already exists")
        await client.rpush(self._queue_name, self._job_id(record))

    async def get(self, job_id: str) -> Optional[JobRecordT]:
        client = await self._client()
        payload = await client.get(self._job_key(job_id))
        if payload is None:
            return None
        return self._model_cls.model_validate_json(payload)

    async def update(self, record: JobRecordT) -> None:
        client = await self._client()
        self._set_updated_at(record)
        await client.set(self._job_key(self._job_id(record)), record.model_dump_json(), ex=self._ttl_seconds)

    async def heartbeat(self, job_id: str, worker_id: str) -> bool:
        record = await self.get(job_id)
        if record is None or getattr(record, "status") != "running" or getattr(record, "worker_id") != worker_id:
            return False
        setattr(record, "lease_expires_at", self._lease_expiry_iso())
        await self.update(record)
        return True

    async def requeue_stale_running_jobs(self) -> int:
        client = await self._client()
        count = 0
        async for key in client.scan_iter(match=self._key_pattern):
            payload = await client.get(key)
            if payload is None:
                continue
            record = self._model_cls.model_validate_json(payload)
            if getattr(record, "status") != "running":
                continue
            lease_expires_at = self._parse_iso(getattr(record, "lease_expires_at"))
            if lease_expires_at is None or lease_expires_at > datetime.now(timezone.utc):
                continue
            self._transition_record(
                record,
                "queued",
                worker_id=None,
                lease_expires_at=None,
                error="Worker lease expired; job re-queued automatically.",
            )
            await self.update(record)
            await client.rpush(self._queue_name, self._job_id(record))
            count += 1
        return count

    async def claim_next(self, worker_id: str, timeout_s: int = 5) -> Optional[JobRecordT]:
        client = await self._client()
        await self.requeue_stale_running_jobs()
        popped = await client.blpop(self._queue_name, timeout=timeout_s)
        if popped is None:
            return None

        _queue, job_id = popped
        record = await self.get(job_id)
        if record is None:
            return None
        if getattr(record, "status") != "queued":
            return record

        self._transition_record(
            record,
            "running",
            worker_id=worker_id,
            started_at=getattr(record, "started_at") or self._utc_now_iso(),
            lease_expires_at=self._lease_expiry_iso(),
        )
        await self.update(record)
        return record

    async def mark_completed(
        self,
        record: JobRecordT,
        *,
        result: object,
        completed_at: str,
        result_object_key: str | None = None,
    ) -> JobRecordT:
        self._transition_record(
            record,
            "completed",
            result=result,
            error=None,
            completed_at=completed_at,
            result_object_key=result_object_key,
            lease_expires_at=None,
        )
        await self.update(record)
        return record

    async def mark_failed(
        self,
        record: JobRecordT,
        *,
        error: str,
        completed_at: str,
        result_object_key: str | None = None,
    ) -> JobRecordT:
        self._transition_record(
            record,
            "failed",
            result=None,
            error=error,
            completed_at=completed_at,
            result_object_key=result_object_key,
            lease_expires_at=None,
        )
        await self.update(record)
        return record
