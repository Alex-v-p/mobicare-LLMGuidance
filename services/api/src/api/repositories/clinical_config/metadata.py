from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from api.repositories.clinical_config.models import ManagedClinicalConfig
from shared.config import Settings
from shared.contracts.clinical_config import ClinicalConfigMetadata, ClinicalConfigName, ClinicalConfigVersionMetadata


def build_managed_configs(settings: Settings) -> dict[str, ManagedClinicalConfig]:
    bucket = settings.clinical_config_bucket
    return {
        "drug_dosing_catalog": ManagedClinicalConfig(
            config_name="drug_dosing_catalog",
            bucket=bucket,
            object_name=join_object_name(
                settings.clinical_config_prefix,
                settings.clinical_drug_dosing_catalog_object_name,
            ),
        ),
        "marker_ranges": ManagedClinicalConfig(
            config_name="marker_ranges",
            bucket=bucket,
            object_name=join_object_name(
                settings.clinical_config_prefix,
                settings.clinical_marker_ranges_object_name,
            ),
        ),
    }


def build_metadata(
    managed: ManagedClinicalConfig,
    *,
    stat: object | None,
    checksum_sha256: str | None,
) -> ClinicalConfigMetadata:
    return ClinicalConfigMetadata(
        config_name=managed.config_name,
        bucket=managed.bucket,
        object_name=managed.object_name,
        exists_in_minio=stat is not None,
        content_type="application/json",
        size_bytes=getattr(stat, "size", None) if stat is not None else None,
        etag=getattr(stat, "etag", None) if stat is not None else None,
        checksum_sha256=checksum_sha256,
        last_modified=getattr(stat, "last_modified", None) if stat is not None else None,
    )


def build_version_metadata_from_snapshot(
    snapshot: dict[str, Any],
    *,
    bucket: str,
    object_name: str,
    config_name: ClinicalConfigName,
    size_bytes: int | None = None,
) -> ClinicalConfigVersionMetadata:
    created_at_raw = snapshot.get("created_at")
    if isinstance(created_at_raw, str):
        created_at = datetime.fromisoformat(created_at_raw)
    elif isinstance(created_at_raw, datetime):
        created_at = created_at_raw
    else:
        created_at = datetime.now(UTC)

    version_id = snapshot.get("version_id")
    if not isinstance(version_id, str) or not version_id:
        version_id = object_name.rsplit("/", 1)[-1].removesuffix(".json")

    reason = snapshot.get("reason")
    if reason not in {"create", "update", "delete", "rollback"}:
        reason = "update"

    return ClinicalConfigVersionMetadata(
        config_name=config_name,
        version_id=version_id,
        bucket=bucket,
        object_name=object_name,
        reason=reason,
        source_etag=snapshot.get("source_etag") if isinstance(snapshot.get("source_etag"), str) else None,
        source_checksum_sha256=(
            snapshot.get("source_checksum_sha256") if isinstance(snapshot.get("source_checksum_sha256"), str) else None
        ),
        created_at=created_at,
        size_bytes=size_bytes,
    )


def join_object_name(prefix: str, object_name: str) -> str:
    normalized_prefix = prefix.strip().strip("/")
    normalized_name = object_name.strip().lstrip("/")
    if not normalized_prefix:
        return normalized_name
    return f"{normalized_prefix}/{normalized_name}"


__all__ = [
    "build_managed_configs",
    "build_metadata",
    "build_version_metadata_from_snapshot",
    "join_object_name",
]
