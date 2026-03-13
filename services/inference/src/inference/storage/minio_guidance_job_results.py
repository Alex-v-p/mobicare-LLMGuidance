from __future__ import annotations

from shared.config import Settings
from shared.contracts.inference import JobRecord

from inference.storage.base_minio_job_results import MinioJobResultStoreBase


class MinioGuidanceJobResultStore(MinioJobResultStoreBase[JobRecord]):
    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(
            record_model=JobRecord,
            object_prefix="jobs",
            rule_id_prefix="expire-job-results",
            settings=settings,
        )
