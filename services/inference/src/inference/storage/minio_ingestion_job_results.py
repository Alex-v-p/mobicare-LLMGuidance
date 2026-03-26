from __future__ import annotations

from shared.config import InferenceSettings
from shared.contracts.ingestion import IngestionJobRecord

from inference.storage.base_minio_job_results import MinioJobResultStoreBase


class MinioIngestionJobResultStore(MinioJobResultStoreBase[IngestionJobRecord]):
    def __init__(self, settings: InferenceSettings | None = None) -> None:
        super().__init__(
            record_model=IngestionJobRecord,
            object_prefix="ingestion-jobs",
            rule_id_prefix="expire-ingestion-job-results",
            settings=settings,
        )
