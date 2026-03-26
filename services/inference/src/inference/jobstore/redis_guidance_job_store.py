from __future__ import annotations

from shared.config import InferenceSettings, get_inference_settings
from shared.contracts.inference import JobRecord

from inference.jobstore.base import RedisJobStoreBase


class RedisGuidanceJobStore(RedisJobStoreBase[JobRecord]):
    def __init__(
        self,
        redis_url: str | None = None,
        queue_name: str | None = None,
        ttl_seconds: int | None = None,
        lease_seconds: int | None = None,
        settings: InferenceSettings | None = None,
    ) -> None:
        resolved_settings = settings or get_inference_settings()
        super().__init__(
            model_cls=JobRecord,
            redis_url=redis_url or resolved_settings.redis_url,
            queue_name=queue_name or resolved_settings.redis_job_queue,
            key_prefix="job:",
            key_pattern="job:*",
            ttl_seconds=ttl_seconds if ttl_seconds is not None else resolved_settings.job_ttl_seconds,
            lease_seconds=lease_seconds if lease_seconds is not None else resolved_settings.job_lease_seconds,
            settings=resolved_settings,
        )
