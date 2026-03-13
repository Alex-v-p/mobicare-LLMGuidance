from __future__ import annotations

from shared.config import Settings, get_settings
from shared.contracts.ingestion import IngestionJobRecord

from inference.jobstore.base import RedisJobStoreBase


class RedisIngestionJobStore(RedisJobStoreBase[IngestionJobRecord]):
    def __init__(
        self,
        redis_url: str | None = None,
        queue_name: str | None = None,
        ttl_seconds: int | None = None,
        lease_seconds: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved_settings = settings or get_settings()
        super().__init__(
            model_cls=IngestionJobRecord,
            redis_url=redis_url or resolved_settings.redis_url,
            queue_name=queue_name or resolved_settings.redis_ingestion_job_queue,
            key_prefix="ingestion_job:",
            key_pattern="ingestion_job:*",
            ttl_seconds=ttl_seconds if ttl_seconds is not None else resolved_settings.job_ttl_seconds,
            lease_seconds=lease_seconds if lease_seconds is not None else resolved_settings.job_lease_seconds,
            settings=resolved_settings,
        )
