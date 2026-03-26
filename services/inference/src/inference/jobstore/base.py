from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from time import monotonic
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar, get_args, get_origin

import redis.asyncio as redis
from pydantic import BaseModel

from shared.config import InferenceSettings, get_inference_settings

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


_CREATE_JOB_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 1 then
    return 0
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[2]))
redis.call('RPUSH', KEYS[2], ARGV[3])
return 1
"""

_SYNC_RECORD_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 0 then
    return 0
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[2]))
if ARGV[3] == '1' then
    redis.call('ZADD', KEYS[2], tonumber(ARGV[4]), KEYS[1])
else
    redis.call('ZREM', KEYS[2], KEYS[1])
end
return 1
"""

_CLAIM_JOB_SCRIPT = """
local payload = redis.call('GET', KEYS[1])
if not payload then
    return nil
end
local record = cjson.decode(payload)
if record['status'] ~= 'queued' then
    return nil
end
record['status'] = 'running'
record['worker_id'] = ARGV[1]
if not record['started_at'] then
    record['started_at'] = ARGV[2]
end
record['lease_expires_at'] = ARGV[3]
record['updated_at'] = ARGV[2]
local updated = cjson.encode(record)
redis.call('SET', KEYS[1], updated, 'EX', tonumber(ARGV[4]))
redis.call('ZADD', KEYS[2], tonumber(ARGV[5]), KEYS[1])
return updated
"""

_HEARTBEAT_SCRIPT = """
local payload = redis.call('GET', KEYS[1])
if not payload then
    return nil
end
local record = cjson.decode(payload)
if record['status'] ~= 'running' or record['worker_id'] ~= ARGV[1] then
    return nil
end
record['lease_expires_at'] = ARGV[2]
record['updated_at'] = ARGV[3]
local updated = cjson.encode(record)
redis.call('SET', KEYS[1], updated, 'EX', tonumber(ARGV[4]))
redis.call('ZADD', KEYS[2], tonumber(ARGV[5]), KEYS[1])
return updated
"""

_REQUEUE_STALE_JOB_SCRIPT = """
local payload = redis.call('GET', KEYS[1])
if not payload then
    redis.call('ZREM', KEYS[2], KEYS[1])
    return nil
end
local record = cjson.decode(payload)
if record['status'] ~= 'running' then
    redis.call('ZREM', KEYS[2], KEYS[1])
    return nil
end
local score = redis.call('ZSCORE', KEYS[2], KEYS[1])
if (not score) or tonumber(score) > tonumber(ARGV[1]) then
    return nil
end
record['status'] = 'queued'
record['worker_id'] = cjson.null
record['lease_expires_at'] = cjson.null
record['error'] = ARGV[3]
record['updated_at'] = ARGV[2]
local updated = cjson.encode(record)
redis.call('SET', KEYS[1], updated, 'EX', tonumber(ARGV[4]))
redis.call('ZREM', KEYS[2], KEYS[1])
redis.call('RPUSH', KEYS[3], record['job_id'])
return updated
"""

_FINISH_JOB_SCRIPT = """
local payload = redis.call('GET', KEYS[1])
if not payload then
    return '__missing__'
end
local record = cjson.decode(payload)
if record['status'] ~= 'running' then
    return '__state__:' .. tostring(record['status'])
end
if tostring(record['worker_id']) ~= ARGV[1] then
    return '__worker__:' .. tostring(record['worker_id'])
end
record['status'] = ARGV[2]
if ARGV[3] == '1' then
    record['result'] = cjson.decode(ARGV[4])
    record['error'] = cjson.null
else
    record['result'] = cjson.null
    record['error'] = ARGV[5]
end
if ARGV[6] == '' then
    record['result_object_key'] = cjson.null
else
    record['result_object_key'] = ARGV[6]
end
record['completed_at'] = ARGV[7]
record['lease_expires_at'] = cjson.null
record['updated_at'] = ARGV[8]
local updated = cjson.encode(record)
redis.call('SET', KEYS[1], updated, 'EX', tonumber(ARGV[9]))
redis.call('ZREM', KEYS[2], KEYS[1])
return updated
"""


class JobStateConflictError(RuntimeError):
    pass


class RedisJobStoreBase(Generic[JobRecordT]):
    def __init__(
        self,
        *,
        model_cls: type[JobRecordT],
        redis_url: str,
        queue_name: str,
        key_prefix: str,
        ttl_seconds: int,
        lease_seconds: int,
        settings: InferenceSettings | None = None,
    ) -> None:
        self._settings = settings or get_inference_settings()
        self._model_cls = model_cls
        self._redis_url = redis_url
        self._queue_name = queue_name
        self._key_prefix = key_prefix
        self._lease_index_key = f"{key_prefix.rstrip(':')}:leases"
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

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _utc_now_iso(self) -> str:
        return self._utc_now().isoformat()

    def _parse_iso(self, value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value)

    def _lease_expiry(self) -> datetime:
        return self._utc_now() + timedelta(seconds=self._lease_seconds)

    def _lease_expiry_iso(self) -> str:
        return self._lease_expiry().isoformat()

    def _lease_score(self, lease_expires_at: str) -> float:
        lease_expiry = self._parse_iso(lease_expires_at)
        if lease_expiry is None:
            raise ValueError("lease_expires_at is required to calculate a lease score")
        return lease_expiry.timestamp()

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

    def _normalize_for_annotation(self, value: Any, annotation: Any) -> Any:
        origin = get_origin(annotation)
        if origin is None:
            if annotation is Any:
                return value
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                if isinstance(value, dict):
                    return self._normalize_for_model_dict(value, annotation)
                return value
            return value

        if origin in (list, set, tuple):
            item_annotation = get_args(annotation)[0] if get_args(annotation) else Any
            if value == {}:
                value = []
            if isinstance(value, list):
                return [self._normalize_for_annotation(item, item_annotation) for item in value]
            return value

        if origin is dict:
            key_annotation, value_annotation = get_args(annotation) if get_args(annotation) else (Any, Any)
            if isinstance(value, dict):
                return {
                    self._normalize_for_annotation(k, key_annotation): self._normalize_for_annotation(v, value_annotation)
                    for k, v in value.items()
                }
            return value

        if origin in (tuple,):
            item_annotations = get_args(annotation)
            if isinstance(value, list) and item_annotations:
                return [
                    self._normalize_for_annotation(item, item_annotations[min(index, len(item_annotations) - 1)])
                    for index, item in enumerate(value)
                ]
            return value

        if origin is type(None):
            return value

        if origin is not None:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(args) != len(get_args(annotation)):
                if value is None:
                    return None
                for arg in args:
                    normalized = self._normalize_for_annotation(value, arg)
                    if normalized is not value or arg is Any:
                        return normalized
                return value

        return value

    def _normalize_for_model_dict(self, value: dict[str, Any], model_cls: type[BaseModel]) -> dict[str, Any]:
        normalized = dict(value)
        for field_name, field_info in model_cls.model_fields.items():
            if field_name not in normalized:
                continue
            normalized[field_name] = self._normalize_for_annotation(normalized[field_name], field_info.annotation)
        return normalized

    def _deserialize(self, payload: str | None) -> Optional[JobRecordT]:
        if payload is None:
            return None
        raw = json.loads(payload)
        if isinstance(raw, dict):
            raw = self._normalize_for_model_dict(raw, self._model_cls)
        return self._model_cls.model_validate(raw)

    async def _eval_script(self, script: str, keys: list[str], args: list[Any]) -> Any:
        client = await self._client()
        if hasattr(client, "eval"):
            return await client.eval(script, len(keys), *keys, *args)
        raise NotImplementedError

    async def _sync_record(self, record: JobRecordT) -> None:
        payload = record.model_dump_json()
        lease_expires_at = getattr(record, "lease_expires_at", None)
        has_lease = bool(getattr(record, "status") == "running" and lease_expires_at)
        try:
            updated = await self._eval_script(
                _SYNC_RECORD_SCRIPT,
                [self._job_key(self._job_id(record)), self._lease_index_key],
                [
                    payload,
                    self._ttl_seconds,
                    "1" if has_lease else "0",
                    self._lease_score(lease_expires_at) if has_lease else 0,
                ],
            )
            if updated != 1:
                raise FileNotFoundError(f"Job {self._job_id(record)} not found")
        except NotImplementedError:
            client = await self._client()
            key = self._job_key(self._job_id(record))
            if await client.get(key) is None:
                raise FileNotFoundError(f"Job {self._job_id(record)} not found")
            await client.set(key, payload, ex=self._ttl_seconds)
            if has_lease:
                await client.zadd(self._lease_index_key, {key: self._lease_score(lease_expires_at)})
            else:
                await client.zrem(self._lease_index_key, key)

    async def create(self, record: JobRecordT) -> None:
        key = self._job_key(self._job_id(record))
        try:
            created = await self._eval_script(
                _CREATE_JOB_SCRIPT,
                [key, self._queue_name],
                [record.model_dump_json(), self._ttl_seconds, self._job_id(record)],
            )
        except NotImplementedError:
            client = await self._client()
            created = await client.set(key, record.model_dump_json(), ex=self._ttl_seconds, nx=True)
            if created:
                await client.rpush(self._queue_name, self._job_id(record))
        if not created:
            raise FileExistsError(f"Job {self._job_id(record)} already exists")

    async def get(self, job_id: str) -> Optional[JobRecordT]:
        client = await self._client()
        payload = await client.get(self._job_key(job_id))
        return self._deserialize(payload)

    async def update(self, record: JobRecordT) -> None:
        self._set_updated_at(record)
        await self._sync_record(record)

    async def heartbeat(self, job_id: str, worker_id: str) -> bool:
        key = self._job_key(job_id)
        lease_expires_at = self._lease_expiry_iso()
        updated_at = self._utc_now_iso()
        try:
            payload = await self._eval_script(
                _HEARTBEAT_SCRIPT,
                [key, self._lease_index_key],
                [worker_id, lease_expires_at, updated_at, self._ttl_seconds, self._lease_score(lease_expires_at)],
            )
            return payload is not None
        except NotImplementedError:
            record = await self.get(job_id)
            if record is None or getattr(record, "status") != "running" or getattr(record, "worker_id") != worker_id:
                return False
            setattr(record, "lease_expires_at", lease_expires_at)
            setattr(record, "updated_at", updated_at)
            await self._sync_record(record)
            return True

    async def requeue_stale_running_jobs(self, batch_size: int = 100) -> int:
        client = await self._client()
        count = 0
        now = self._utc_now()
        now_iso = now.isoformat()
        now_score = now.timestamp()
        while True:
            expired_keys = await client.zrangebyscore(self._lease_index_key, min="-inf", max=now_score, start=0, num=batch_size)
            if not expired_keys:
                break
            progressed = False
            for key in expired_keys:
                try:
                    payload = await self._eval_script(
                        _REQUEUE_STALE_JOB_SCRIPT,
                        [key, self._lease_index_key, self._queue_name],
                        [now_score, now_iso, "Worker lease expired; job re-queued automatically.", self._ttl_seconds],
                    )
                except NotImplementedError:
                    raw = await client.get(key)
                    if raw is None:
                        await client.zrem(self._lease_index_key, key)
                        continue
                    record = self._model_cls.model_validate_json(raw)
                    lease_expires_at = self._parse_iso(getattr(record, "lease_expires_at", None))
                    if getattr(record, "status") != "running" or lease_expires_at is None or lease_expires_at > now:
                        if getattr(record, "status") != "running":
                            await client.zrem(self._lease_index_key, key)
                        continue
                    self._transition_record(
                        record,
                        "queued",
                        worker_id=None,
                        lease_expires_at=None,
                        error="Worker lease expired; job re-queued automatically.",
                    )
                    await self._sync_record(record)
                    await client.rpush(self._queue_name, self._job_id(record))
                    payload = record.model_dump_json()
                if payload is not None:
                    count += 1
                    progressed = True
            if not progressed:
                break
        return count

    async def claim_next(self, worker_id: str, timeout_s: int = 5) -> Optional[JobRecordT]:
        client = await self._client()
        await self.requeue_stale_running_jobs()
        deadline = monotonic() + max(timeout_s, 0)
        while True:
            remaining = max(0, int(deadline - monotonic())) if timeout_s > 0 else timeout_s
            popped = await client.blpop(self._queue_name, timeout=remaining)
            if popped is None:
                return None

            _queue, job_id = popped
            key = self._job_key(job_id)
            started_at = self._utc_now_iso()
            lease_expires_at = self._lease_expiry_iso()
            try:
                payload = await self._eval_script(
                    _CLAIM_JOB_SCRIPT,
                    [key, self._lease_index_key],
                    [worker_id, started_at, lease_expires_at, self._ttl_seconds, self._lease_score(lease_expires_at)],
                )
            except NotImplementedError:
                record = await self.get(job_id)
                if record is None or getattr(record, "status") != "queued":
                    payload = None
                else:
                    self._transition_record(
                        record,
                        "running",
                        worker_id=worker_id,
                        started_at=getattr(record, "started_at") or started_at,
                        lease_expires_at=lease_expires_at,
                    )
                    await self._sync_record(record)
                    payload = record.model_dump_json()
            claimed = self._deserialize(payload)
            if claimed is not None:
                return claimed
            if timeout_s <= 0 or monotonic() >= deadline:
                return None

    async def _finish_record(
        self,
        record: JobRecordT,
        *,
        target_status: str,
        result: object | None,
        error: str | None,
        completed_at: str,
        result_object_key: str | None,
    ) -> JobRecordT:
        self._validate_transition(getattr(record, "status"), target_status)
        expected_worker_id = getattr(record, "worker_id", None)
        if not expected_worker_id:
            raise JobStateConflictError(f"Job {self._job_id(record)} has no worker ownership to finish")

        payload: str | None
        try:
            payload = await self._eval_script(
                _FINISH_JOB_SCRIPT,
                [self._job_key(self._job_id(record)), self._lease_index_key],
                [
                    expected_worker_id,
                    target_status,
                    "1" if result is not None else "0",
                    result.model_dump_json() if result is not None else "null",
                    error or "",
                    result_object_key or "",
                    completed_at,
                    self._utc_now_iso(),
                    self._ttl_seconds,
                ],
            )
        except NotImplementedError:
            current = await self.get(self._job_id(record))
            if current is None:
                raise JobStateConflictError(f"Job {self._job_id(record)} no longer exists")
            if getattr(current, "status") != "running":
                raise JobStateConflictError(
                    f"Job {self._job_id(record)} is {getattr(current, 'status')} and cannot transition to {target_status}"
                )
            if getattr(current, "worker_id") != expected_worker_id:
                raise JobStateConflictError(
                    f"Job {self._job_id(record)} is owned by {getattr(current, 'worker_id')} instead of {expected_worker_id}"
                )
            self._transition_record(
                current,
                target_status,
                result=result,
                error=error,
                completed_at=completed_at,
                result_object_key=result_object_key,
                lease_expires_at=None,
            )
            await self._sync_record(current)
            return current

        if payload == "__missing__":
            raise JobStateConflictError(f"Job {self._job_id(record)} no longer exists")
        if isinstance(payload, str) and payload.startswith("__state__:"):
            current_state = payload.split(":", 1)[1]
            raise JobStateConflictError(
                f"Job {self._job_id(record)} is {current_state} and cannot transition to {target_status}"
            )
        if isinstance(payload, str) and payload.startswith("__worker__:"):
            current_worker = payload.split(":", 1)[1]
            raise JobStateConflictError(
                f"Job {self._job_id(record)} is owned by {current_worker} instead of {expected_worker_id}"
            )
        updated = self._deserialize(payload)
        if updated is None:
            raise JobStateConflictError(f"Job {self._job_id(record)} could not be transitioned to {target_status}")
        return updated

    async def mark_completed(
        self,
        record: JobRecordT,
        *,
        result: object,
        completed_at: str,
        result_object_key: str | None = None,
    ) -> JobRecordT:
        return await self._finish_record(
            record,
            target_status="completed",
            result=result,
            error=None,
            completed_at=completed_at,
            result_object_key=result_object_key,
        )

    async def mark_failed(
        self,
        record: JobRecordT,
        *,
        error: str,
        completed_at: str,
        result_object_key: str | None = None,
    ) -> JobRecordT:
        return await self._finish_record(
            record,
            target_status="failed",
            result=None,
            error=error,
            completed_at=completed_at,
            result_object_key=result_object_key,
        )
