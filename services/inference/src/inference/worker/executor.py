from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

from pydantic import BaseModel

from inference.queue import ClaimableJobStore
from inference.jobstore.base import JobStateConflictError
from inference.worker.heartbeat import with_heartbeat
from shared.observability import get_logger


RecordT = TypeVar("RecordT", bound=BaseModel)
RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")

logger = get_logger(__name__, service="inference")


class ExecutableJobStore(ClaimableJobStore[RecordT], Protocol[RecordT]):
    async def mark_completed(
        self,
        record: RecordT,
        *,
        result: object,
        completed_at: str,
        result_object_key: str | None = None,
    ) -> RecordT: ...

    async def mark_failed(
        self,
        record: RecordT,
        *,
        error: str,
        completed_at: str,
        result_object_key: str | None = None,
    ) -> RecordT: ...


class PersistableJobResultStore(Protocol[RecordT]):
    def put_job_result(self, record: RecordT) -> str: ...


WithHeartbeat = Callable[..., Awaitable[None]]
PostProcessHook = Callable[[RecordT], Awaitable[None]]
RunRequest = Callable[[RequestT], Awaitable[ResultT]]
TimestampFactory = Callable[[], str]
BeforeRunHook = Callable[[RecordT], Awaitable[None]]
SuccessHook = Callable[[RecordT, ResultT], Awaitable[None]]
FailureHook = Callable[[RecordT, str], Awaitable[None]]


async def execute_next_job(
    *,
    store: ExecutableJobStore[RecordT],
    result_store: PersistableJobResultStore[RecordT],
    worker_id: str,
    heartbeat_interval_s: int,
    run_request: RunRequest,
    utc_now_iso: TimestampFactory,
    run_with_heartbeat: WithHeartbeat = with_heartbeat,
    post_process: PostProcessHook | None = None,
    before_run: BeforeRunHook | None = None,
    after_success: SuccessHook | None = None,
    after_failure: FailureHook | None = None,
    job_kind: str = "job",
) -> bool:
    try:
        record = await store.claim_next(worker_id=worker_id, timeout_s=1)
        if record is None or getattr(record, "status") != "running":
            return False

        async def process() -> None:
            try:
                if before_run is not None:
                    await before_run(record)
                result = await run_request(getattr(record, "request"))
                completed_at = utc_now_iso()
                record.result_object_key = result_store.put_job_result(
                    record.model_copy(
                        update={
                            "status": "completed",
                            "result": result,
                            "error": None,
                            "completed_at": completed_at,
                            "lease_expires_at": None,
                        }
                    )
                )
                try:
                    await store.mark_completed(
                        record,
                        result=result,
                        completed_at=completed_at,
                        result_object_key=record.result_object_key,
                    )
                except JobStateConflictError:  # pragma: no cover
                    logger.warning(
                        "worker_job_state_conflict",
                        extra={
                            "event": "worker_job_state_conflict",
                            "error_code": "JOB_STATE_CONFLICT",
                            "job_id": getattr(record, "job_id", None),
                            "job_kind": job_kind,
                            "target_status": "completed",
                        },
                    )
                if after_success is not None:
                    try:
                        await after_success(record, result)
                    except Exception:  # pragma: no cover
                        logger.exception(
                            "worker_job_success_hook_failed",
                            extra={
                                "event": "worker_job_success_hook_failed",
                                "error_code": "JOB_SUCCESS_HOOK_FAILED",
                                "job_id": getattr(record, "job_id", None),
                                "job_kind": job_kind,
                            },
                        )
            except Exception as exc:  # pragma: no cover
                error = f"{type(exc).__name__}: {exc}"
                failed_at = utc_now_iso()
                record.result_object_key = result_store.put_job_result(
                    record.model_copy(
                        update={
                            "status": "failed",
                            "result": None,
                            "error": error,
                            "completed_at": failed_at,
                            "lease_expires_at": None,
                        }
                    )
                )
                try:
                    await store.mark_failed(
                        record,
                        error=error,
                        completed_at=failed_at,
                        result_object_key=record.result_object_key,
                    )
                except JobStateConflictError:  # pragma: no cover
                    logger.warning(
                        "worker_job_state_conflict",
                        extra={
                            "event": "worker_job_state_conflict",
                            "error_code": "JOB_STATE_CONFLICT",
                            "job_id": getattr(record, "job_id", None),
                            "job_kind": job_kind,
                            "target_status": "failed",
                        },
                    )
                if after_failure is not None:
                    try:
                        await after_failure(record, error)
                    except Exception:  # pragma: no cover
                        logger.exception(
                            "worker_job_failure_hook_failed",
                            extra={
                                "event": "worker_job_failure_hook_failed",
                                "error_code": "JOB_FAILURE_HOOK_FAILED",
                                "job_id": getattr(record, "job_id", None),
                                "job_kind": job_kind,
                            },
                        )
                logger.exception(
                    "worker_job_execution_failed",
                    extra={
                        "event": "worker_job_execution_failed",
                        "error_code": "JOB_EXECUTION_FAILED",
                        "job_id": getattr(record, "job_id", None),
                        "job_kind": job_kind,
                    },
                )

        await run_with_heartbeat(
            store=store,
            job_id=getattr(record, "job_id"),
            worker_id=worker_id,
            heartbeat_interval_s=heartbeat_interval_s,
            operation=process,
        )

        if post_process is not None:
            try:
                await post_process(record)
                await store.update(record)
            except Exception:  # pragma: no cover
                logger.exception(
                    "worker_job_post_process_failed",
                    extra={
                        "event": "worker_job_post_process_failed",
                        "error_code": "JOB_POST_PROCESS_FAILED",
                        "job_id": getattr(record, "job_id", None),
                        "job_kind": job_kind,
                    },
                )

        return True
    finally:
        await store.close()
