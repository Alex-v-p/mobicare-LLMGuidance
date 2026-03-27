from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

from minio import Minio
from minio.error import S3Error

from shared.bootstrap.clinical_defaults import build_managed_clinical_object_names, load_clinical_config_default_bytes
from shared.config import SharedServiceSettings
from shared.observability import get_logger

logger = get_logger(__name__, service="shared")

_BUCKET_ALREADY_EXISTS_CODES = {"BucketAlreadyExists", "BucketAlreadyOwnedByYou"}


def create_minio_client_from_settings(settings: SharedServiceSettings) -> Minio:
    try:
        return Minio(
            settings.minio_client_endpoint,
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            secure=settings.minio_secure,
        )
    except TypeError:
        return Minio()



@dataclass(slots=True)
class BootstrapReport:
    buckets_created: list[str] = field(default_factory=list)
    clinical_configs_seeded: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.buckets_created or self.clinical_configs_seeded)



def ensure_minio_bucket(client: Minio, bucket: str) -> bool:
    if not hasattr(client, "bucket_exists") or not hasattr(client, "make_bucket"):
        return False
    if client.bucket_exists(bucket):
        return False
    try:
        client.make_bucket(bucket)
        return True
    except S3Error as exc:
        if getattr(exc, "code", "") in _BUCKET_ALREADY_EXISTS_CODES:
            return False
        raise



def _bucket_has_objects(client: Minio, bucket: str) -> bool:
    if not hasattr(client, "list_objects"):
        return False
    for obj in client.list_objects(bucket, recursive=True):
        if not getattr(obj, "is_dir", False):
            return True
    return False



def bootstrap_minio_resources(*, settings: SharedServiceSettings, client: Minio) -> BootstrapReport:
    report = BootstrapReport()

    for bucket in (settings.minio_documents_bucket, settings.minio_results_bucket):
        if ensure_minio_bucket(client, bucket):
            report.buckets_created.append(bucket)

    clinical_bucket = settings.clinical_config_bucket
    clinical_bucket_created = ensure_minio_bucket(client, clinical_bucket)
    if clinical_bucket_created:
        report.buckets_created.append(clinical_bucket)

    if clinical_bucket_created or not _bucket_has_objects(client, clinical_bucket):
        for config_name, object_name in build_managed_clinical_object_names(settings).items():
            payload = load_clinical_config_default_bytes(config_name)
            client.put_object(
                clinical_bucket,
                object_name,
                data=io.BytesIO(payload),
                length=len(payload),
                content_type="application/json",
            )
            report.clinical_configs_seeded.append(config_name)

    return report



def bootstrap_minio_resources_on_startup(*, settings: SharedServiceSettings, client: Minio, service: str) -> BootstrapReport:
    try:
        report = bootstrap_minio_resources(settings=settings, client=client)
    except Exception as exc:
        logger.warning(
            "bootstrap_minio_resources_failed",
            extra={
                "event": "bootstrap_minio_resources_failed",
                "dependency": "minio",
                "service": service,
            },
            exc_info=exc,
        )
        return BootstrapReport()

    if report.changed:
        logger.info(
            "bootstrap_minio_resources_applied",
            extra={
                "event": "bootstrap_minio_resources_applied",
                "dependency": "minio",
                "service": service,
                "response_body": {
                    "buckets_created": report.buckets_created,
                    "clinical_configs_seeded": report.clinical_configs_seeded,
                },
            },
        )
    return report
