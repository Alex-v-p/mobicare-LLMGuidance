from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Generic, Protocol, TypeVar

from minio import Minio
from minio.commonconfig import ENABLED, Filter
from minio.lifecycleconfig import Expiration, LifecycleConfig, Rule
from pydantic import BaseModel

from shared.bootstrap import create_minio_client_from_settings, ensure_minio_bucket
from shared.config import Settings, get_settings
from shared.observability import get_logger

JobRecordT = TypeVar("JobRecordT", bound=BaseModel)

logger = get_logger(__name__, service="inference")


class JobResultStore(Protocol[JobRecordT]):
    def get_job_result(self, object_key: str) -> JobRecordT: ...


class MinioJobResultStoreBase(Generic[JobRecordT]):
    def __init__(
        self,
        *,
        record_model: type[JobRecordT],
        object_prefix: str,
        rule_id_prefix: str,
        settings: Settings | None = None,
        client: Minio | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._record_model = record_model
        self._object_prefix = object_prefix.rstrip("/")
        self._rule_id_prefix = rule_id_prefix
        self._bucket = self._settings.minio_results_bucket
        self._retention_days = self._settings.minio_job_retention_days
        self._client = client or create_minio_client_from_settings(self._settings)

    def _lifecycle_rule(self) -> LifecycleConfig:
        return LifecycleConfig(
            [
                Rule(
                    ENABLED,
                    rule_filter=Filter(prefix=f"{self._object_prefix}/"),
                    rule_id=f"{self._rule_id_prefix}-after-{self._retention_days}-days",
                    expiration=Expiration(days=self._retention_days),
                )
            ]
        )

    def ensure_bucket(self) -> None:
        ensure_minio_bucket(self._client, self._bucket)
        try:
            self._client.set_bucket_lifecycle(self._bucket, self._lifecycle_rule())
        except Exception as exc:
            logger.warning(
                "minio_set_bucket_lifecycle_failed",
                extra={
                    "event": "minio_set_bucket_lifecycle_failed",
                    "dependency": "minio",
                },
                exc_info=exc,
            )

    def build_object_key(self, job_id: str, completed_at_iso: str | None = None) -> str:
        completed_at = datetime.fromisoformat(completed_at_iso) if completed_at_iso else datetime.now(timezone.utc)
        return f"{self._object_prefix}/{completed_at:%Y/%m/%d}/{job_id}.json"

    def put_job_result(self, record: JobRecordT) -> str:
        self.ensure_bucket()
        completed_at = getattr(record, "completed_at", None)
        object_key = self.build_object_key(getattr(record, "job_id"), completed_at)
        payload = json.dumps(record.model_dump(mode="json"), indent=2).encode("utf-8")
        self._client.put_object(
            self._bucket,
            object_key,
            data=io.BytesIO(payload),
            length=len(payload),
            content_type="application/json",
        )
        return object_key

    def get_job_result(self, object_key: str) -> JobRecordT:
        response = self._client.get_object(self._bucket, object_key)
        try:
            return self._record_model.model_validate_json(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
