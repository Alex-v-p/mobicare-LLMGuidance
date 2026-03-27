from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from api.infrastructure.repositories.clinical_config.errors import ClinicalConfigOptimisticLockError, InvalidClinicalConfigError
from api.infrastructure.repositories.clinical_config.metadata import build_version_metadata_from_snapshot
from api.infrastructure.repositories.clinical_config.models import CurrentConfigState, ManagedClinicalConfig
from api.infrastructure.repositories.clinical_config.storage import ClinicalConfigStorage
from shared.config import ApiSettings
from shared.contracts.clinical_config import ClinicalConfigMetadata, ClinicalConfigName, ClinicalConfigVersionMetadata


class ClinicalConfigVersionStore:
    def __init__(self, *, storage: ClinicalConfigStorage, settings: ApiSettings) -> None:
        self._storage = storage
        self._settings = settings

    def archive_state(
        self,
        managed: ManagedClinicalConfig,
        state: CurrentConfigState,
        *,
        reason: str,
    ) -> ClinicalConfigVersionMetadata:
        version_id = build_version_id()
        object_name = self.version_object_name(managed, version_id)
        snapshot = {
            "version_id": version_id,
            "config_name": managed.config_name,
            "reason": reason,
            "created_at": datetime.now(UTC).isoformat(),
            "source_etag": state.metadata.etag,
            "source_checksum_sha256": state.metadata.checksum_sha256,
            "payload": state.payload,
        }
        size_bytes = self._storage.put_version_snapshot(
            bucket=managed.bucket,
            object_name=object_name,
            snapshot=snapshot,
        )
        return build_version_metadata_from_snapshot(
            snapshot,
            bucket=managed.bucket,
            object_name=object_name,
            config_name=managed.config_name,
            size_bytes=size_bytes,
        )

    def list_versions(self, managed: ManagedClinicalConfig) -> list[ClinicalConfigVersionMetadata]:
        versions: list[ClinicalConfigVersionMetadata] = []
        for object_name in self._storage.list_object_names(bucket=managed.bucket, prefix=self.versions_prefix(managed)):
            snapshot = self.get_version_snapshot(managed.bucket, object_name)
            versions.append(
                build_version_metadata_from_snapshot(
                    snapshot,
                    bucket=managed.bucket,
                    object_name=object_name,
                    config_name=managed.config_name,
                )
            )
        versions.sort(key=lambda item: item.created_at, reverse=True)
        return versions

    def read_version_snapshot(self, managed: ManagedClinicalConfig, version_id: str) -> dict[str, Any]:
        return self.get_version_snapshot(
            managed.bucket,
            self.version_object_name(managed, version_id),
            version_id=version_id,
        )

    def restored_version_metadata(
        self,
        snapshot: dict[str, Any],
        *,
        managed: ManagedClinicalConfig,
        version_id: str,
    ) -> ClinicalConfigVersionMetadata:
        return build_version_metadata_from_snapshot(
            snapshot,
            bucket=managed.bucket,
            object_name=self.version_object_name(managed, version_id),
            config_name=managed.config_name,
        )

    def assert_optimistic_lock(
        self,
        *,
        config_name: ClinicalConfigName,
        current: ClinicalConfigMetadata,
        expected_etag: str | None,
        expected_checksum_sha256: str | None,
    ) -> None:
        if expected_etag is not None and current.etag != expected_etag:
            raise ClinicalConfigOptimisticLockError(
                f"Clinical config '{config_name}' has changed since it was read (etag mismatch)"
            )
        if expected_checksum_sha256 is not None and current.checksum_sha256 != expected_checksum_sha256:
            raise ClinicalConfigOptimisticLockError(
                f"Clinical config '{config_name}' has changed since it was read (checksum mismatch)"
            )

    def version_object_name(self, managed: ManagedClinicalConfig, version_id: str) -> str:
        return f"{self.versions_prefix(managed)}/{version_id}.json"

    def versions_prefix(self, managed: ManagedClinicalConfig) -> str:
        base = managed.object_name.rsplit("/", 1)[0] if "/" in managed.object_name else ""
        versions_segment = self._settings.clinical_config_versions_prefix.strip().strip("/") or "_versions"
        parts = [part for part in (base, versions_segment, managed.config_name) if part]
        return "/".join(parts)

    def get_version_snapshot(
        self,
        bucket: str,
        object_name: str,
        *,
        version_id: str | None = None,
    ) -> dict[str, Any]:
        raw_bytes = self._storage.get_bytes(bucket, object_name)
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidClinicalConfigError(
                f"Clinical config version '{version_id or object_name}' contains invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise InvalidClinicalConfigError(
                f"Clinical config version '{version_id or object_name}' must be stored as a JSON object"
            )
        return payload


def build_version_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


__all__ = ["ClinicalConfigVersionStore", "build_version_id"]
