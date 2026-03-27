from __future__ import annotations

from functools import lru_cache

from shared.config.base import resolve_runtime_settings
from shared.config.inference import InferenceSettings


class WorkerSettings(InferenceSettings):
    worker_id: str | None = None
    job_heartbeat_interval_seconds: int = 20


@lru_cache(maxsize=1)
def get_worker_settings() -> WorkerSettings:
    return resolve_runtime_settings(WorkerSettings)
