from __future__ import annotations

import io
import json
from datetime import datetime, timezone

from minio import Minio
from minio.commonconfig import ENABLED, Filter
from minio.lifecycleconfig import Expiration, LifecycleConfig, Rule

from shared.config import Settings, get_settings
from shared.contracts.ingestion import IngestionJobRecord


class MinioIngestionJobResultStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._bucket = self._settings.minio_results_bucket
        self._retention_days = self._settings.minio_job_retention_days
        self._client = Minio(
            self._settings.minio_client_endpoint,
            access_key=self._settings.minio_root_user,
            secret_key=self._settings.minio_root_password,
            secure=self._settings.minio_secure,
        )

    def ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
        lifecycle = LifecycleConfig(
            [
                Rule(
                    ENABLED,
                    rule_filter=Filter(prefix="ingestion-jobs/"),
                    rule_id=f"expire-ingestion-job-results-after-{self._retention_days}-days",
                    expiration=Expiration(days=self._retention_days),
                )
            ]
        )
        try:
            self._client.set_bucket_lifecycle(self._bucket, lifecycle)
        except Exception:
            pass

    def build_object_key(self, job_id: str, completed_at_iso: str | None = None) -> str:
        completed_at = datetime.fromisoformat(completed_at_iso) if completed_at_iso else datetime.now(timezone.utc)
        return f"ingestion-jobs/{completed_at:%Y/%m/%d}/{job_id}.json"

    def put_job_result(self, record: IngestionJobRecord) -> str:
        self.ensure_bucket()
        object_key = self.build_object_key(record.job_id, record.completed_at)
        payload = json.dumps(record.model_dump(mode="json"), indent=2).encode("utf-8")
        self._client.put_object(
            self._bucket,
            object_key,
            data=io.BytesIO(payload),
            length=len(payload),
            content_type="application/json",
        )
        return object_key

    def get_job_result(self, object_key: str) -> IngestionJobRecord:
        response = self._client.get_object(self._bucket, object_key)
        try:
            return IngestionJobRecord.model_validate_json(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
