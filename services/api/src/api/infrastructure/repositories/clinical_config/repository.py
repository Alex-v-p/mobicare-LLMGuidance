from __future__ import annotations

import hashlib
import json
from typing import Any

from minio import Minio

from api.infrastructure.repositories.clinical_config.errors import (
    ClinicalConfigAlreadyExistsError,
    ClinicalConfigNotFoundError,
    ClinicalConfigOptimisticLockError,
    UnknownClinicalConfigError,
    InvalidClinicalConfigError,
)
from api.infrastructure.repositories.clinical_config.metadata import build_managed_configs, build_metadata
from api.infrastructure.repositories.clinical_config.models import CurrentConfigState, ManagedClinicalConfig
from api.infrastructure.repositories.clinical_config.storage import ClinicalConfigStorage
from api.infrastructure.repositories.clinical_config.versioning import ClinicalConfigVersionStore
from shared.config import Settings, get_settings
from shared.contracts.clinical_config import ClinicalConfigMetadata, ClinicalConfigName, ClinicalConfigVersionMetadata


class ClinicalConfigRepository:
    def __init__(self, *, client: Minio, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or get_settings()
        self._managed = build_managed_configs(self._settings)
        self._storage = ClinicalConfigStorage(client=self._client, settings=self._settings)
        self._versions = ClinicalConfigVersionStore(storage=self._storage, settings=self._settings)

    def list_configs(self) -> list[ClinicalConfigMetadata]:
        self._storage.ensure_bootstrapped()
        configs: list[ClinicalConfigMetadata] = []
        for config_name in sorted(self._managed):
            managed = self._resolve(config_name)
            state = self._try_read_current_state(managed)
            metadata = state.metadata if state is not None else build_metadata(managed, stat=None, checksum_sha256=None)
            configs.append(metadata)
        return configs

    def get_payload(self, config_name: ClinicalConfigName | str) -> tuple[ClinicalConfigMetadata, dict[str, Any]]:
        self._storage.ensure_bootstrapped()
        state = self._read_current_state(self._resolve(config_name))
        return state.metadata, state.payload

    def create_payload(
        self,
        config_name: ClinicalConfigName | str,
        payload: dict[str, Any],
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> tuple[ClinicalConfigMetadata, ClinicalConfigVersionMetadata | None]:
        managed = self._resolve(config_name)
        if self._storage.try_stat(managed) is not None:
            raise ClinicalConfigAlreadyExistsError(f"Clinical config '{managed.config_name}' already exists in MinIO")
        if expected_etag or expected_checksum_sha256:
            raise ClinicalConfigOptimisticLockError(
                f"Clinical config '{managed.config_name}' does not yet exist, so an optimistic lock cannot be applied"
            )
        self._put_current_payload(managed, payload)
        state = self._read_current_state(managed)
        return state.metadata, None

    def upsert_payload(
        self,
        config_name: ClinicalConfigName | str,
        payload: dict[str, Any],
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> tuple[ClinicalConfigMetadata, str, ClinicalConfigVersionMetadata | None]:
        managed = self._resolve(config_name)
        existing = self._try_read_current_state(managed)
        archived_version: ClinicalConfigVersionMetadata | None = None
        status = "created"
        if existing is not None:
            self._versions.assert_optimistic_lock(
                config_name=managed.config_name,
                current=existing.metadata,
                expected_etag=expected_etag,
                expected_checksum_sha256=expected_checksum_sha256,
            )
            archived_version = self._versions.archive_state(managed, existing, reason="update")
            status = "updated"
        elif expected_etag or expected_checksum_sha256:
            raise ClinicalConfigOptimisticLockError(
                f"Clinical config '{managed.config_name}' does not yet exist, so the provided optimistic lock cannot match"
            )
        self._put_current_payload(managed, payload)
        state = self._read_current_state(managed)
        return state.metadata, status, archived_version

    def delete_payload(
        self,
        config_name: ClinicalConfigName | str,
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> tuple[ClinicalConfigMetadata, ClinicalConfigVersionMetadata]:
        self._storage.ensure_bootstrapped()
        managed = self._resolve(config_name)
        current = self._read_current_state(managed)
        self._versions.assert_optimistic_lock(
            config_name=managed.config_name,
            current=current.metadata,
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )
        archived_version = self._versions.archive_state(managed, current, reason="delete")
        self._storage.remove_object(managed)
        return build_metadata(managed, stat=None, checksum_sha256=None), archived_version

    def list_versions(self, config_name: ClinicalConfigName | str) -> list[ClinicalConfigVersionMetadata]:
        self._storage.ensure_bootstrapped()
        managed = self._resolve(config_name)
        self._storage.ensure_bucket_exists(create=False)
        return self._versions.list_versions(managed)

    def rollback_payload(
        self,
        config_name: ClinicalConfigName | str,
        version_id: str,
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> tuple[ClinicalConfigMetadata, ClinicalConfigVersionMetadata, ClinicalConfigVersionMetadata | None]:
        self._storage.ensure_bootstrapped()
        managed = self._resolve(config_name)
        target_snapshot = self._versions.read_version_snapshot(managed, version_id)
        existing = self._try_read_current_state(managed)
        archived_version: ClinicalConfigVersionMetadata | None = None
        if existing is not None:
            self._versions.assert_optimistic_lock(
                config_name=managed.config_name,
                current=existing.metadata,
                expected_etag=expected_etag,
                expected_checksum_sha256=expected_checksum_sha256,
            )
            archived_version = self._versions.archive_state(managed, existing, reason="rollback")
        elif expected_etag or expected_checksum_sha256:
            raise ClinicalConfigOptimisticLockError(
                f"Clinical config '{managed.config_name}' does not currently exist, so the provided optimistic lock cannot match"
            )
        payload = target_snapshot.get("payload")
        if not isinstance(payload, dict):
            raise InvalidClinicalConfigError(
                f"Clinical config version '{version_id}' does not contain a valid payload"
            )
        self._put_current_payload(managed, payload)
        state = self._read_current_state(managed)
        restored = self._versions.restored_version_metadata(target_snapshot, managed=managed, version_id=version_id)
        return state.metadata, restored, archived_version

    def _resolve(self, config_name: ClinicalConfigName | str) -> ManagedClinicalConfig:
        managed = self._managed.get(config_name)
        if managed is None:
            raise UnknownClinicalConfigError(f"Clinical config '{config_name}' is not managed")
        return managed

    def _try_read_current_state(self, managed: ManagedClinicalConfig) -> CurrentConfigState | None:
        try:
            return self._read_current_state(managed)
        except ClinicalConfigNotFoundError:
            return None

    def _read_current_state(self, managed: ManagedClinicalConfig) -> CurrentConfigState:
        stat = self._storage.stat_object(managed)
        raw_bytes = self._storage.get_bytes(managed.bucket, managed.object_name)
        payload = _decode_payload(raw_bytes, config_name=managed.config_name)
        checksum = _sha256_hexdigest(raw_bytes)
        metadata = build_metadata(managed, stat=stat, checksum_sha256=checksum)
        return CurrentConfigState(metadata=metadata, payload=payload, raw_bytes=raw_bytes)

    def _put_current_payload(self, managed: ManagedClinicalConfig, payload: dict[str, Any]) -> None:
        self._storage.put_json_bytes(managed, _canonical_json_bytes(payload))


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _decode_payload(raw_bytes: bytes, *, config_name: ClinicalConfigName) -> dict[str, Any]:
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidClinicalConfigError(f"Clinical config '{config_name}' contains invalid JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidClinicalConfigError(f"Clinical config '{config_name}' must be stored as a JSON object")
    return payload


def _sha256_hexdigest(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


__all__ = ["ClinicalConfigRepository"]
