from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone

from minio import Minio
from minio.commonconfig import ENABLED, Filter
from minio.lifecycleconfig import Expiration, LifecycleConfig, Rule

from shared.contracts.inference import JobRecord


class MinioResultStore:
    def __init__(self) -> None:
        endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000").replace("http://", "").replace("https://", "")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        self._bucket = os.getenv("MINIO_RESULTS_BUCKET", "guidance-job-results")
        self._retention_days = int(os.getenv("MINIO_JOB_RETENTION_DAYS", "7"))
        self._client = Minio(
            endpoint,
            access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
            secure=secure,
        )

    def ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
        lifecycle = LifecycleConfig(
            [
                Rule(
                    ENABLED,
                    rule_filter=Filter(prefix="jobs/"),
                    rule_id=f"expire-job-results-after-{self._retention_days}-days",
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
        return f"jobs/{completed_at:%Y/%m/%d}/{job_id}.json"

    def put_job_result(self, record: JobRecord) -> str:
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

    def get_job_result(self, object_key: str) -> JobRecord:
        response = self._client.get_object(self._bucket, object_key)
        try:
            return JobRecord.model_validate_json(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
