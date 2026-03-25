from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from minio import Minio
from minio.error import S3Error
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError, NewConnectionError

from shared.config import Settings, get_settings
from shared.contracts.clinical_config import ClinicalConfigMetadata, ClinicalConfigName, ClinicalConfigVersionMetadata
from shared.contracts.error_codes import ErrorCode


@dataclass(frozen=True)
class ManagedClinicalConfig:
    config_name: ClinicalConfigName
    bucket: str
    object_name: str


@dataclass(frozen=True)
class CurrentConfigState:
    metadata: ClinicalConfigMetadata
    payload: dict[str, Any]
    raw_bytes: bytes


class ClinicalConfigRepositoryError(RuntimeError):
    def __init__(self, message: str, *, code: str = ErrorCode.CLINICAL_CONFIG_STORAGE_UNAVAILABLE) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ClinicalConfigNotFoundError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_NOT_FOUND)


class ClinicalConfigVersionNotFoundError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_VERSION_NOT_FOUND)


class ClinicalConfigAlreadyExistsError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_CONFLICT)


class ClinicalConfigOptimisticLockError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_OPTIMISTIC_LOCK_FAILED)


class InvalidClinicalConfigError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_INVALID)


class UnknownClinicalConfigError(ClinicalConfigNotFoundError):
    pass


class ClinicalConfigRepository:
    def __init__(self, *, client: Minio, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or get_settings()
        self._managed = _build_managed_configs(self._settings)

    def list_configs(self) -> list[ClinicalConfigMetadata]:
        configs: list[ClinicalConfigMetadata] = []
        for config_name in sorted(self._managed):
            managed = self._managed[config_name]
            state = self._try_read_current_state(managed)
            metadata = state.metadata if state is not None else _build_metadata(managed, stat=None, checksum_sha256=None)
            configs.append(metadata)
        return configs

    def get_payload(self, config_name: ClinicalConfigName) -> tuple[ClinicalConfigMetadata, dict[str, Any]]:
        state = self._read_current_state(self._resolve(config_name))
        return state.metadata, state.payload

    def create_payload(
        self,
        config_name: ClinicalConfigName,
        payload: dict[str, Any],
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> tuple[ClinicalConfigMetadata, ClinicalConfigVersionMetadata | None]:
        managed = self._resolve(config_name)
        if self._try_stat(managed) is not None:
            raise ClinicalConfigAlreadyExistsError(f"Clinical config '{config_name}' already exists in MinIO")
        if expected_etag or expected_checksum_sha256:
            raise ClinicalConfigOptimisticLockError(
                f"Clinical config '{config_name}' does not yet exist, so an optimistic lock cannot be applied"
            )
        self._put_current_payload(managed, payload)
        state = self._read_current_state(managed)
        return state.metadata, None

    def upsert_payload(
        self,
        config_name: ClinicalConfigName,
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
            self._assert_optimistic_lock(
                config_name=config_name,
                current=existing.metadata,
                expected_etag=expected_etag,
                expected_checksum_sha256=expected_checksum_sha256,
            )
            archived_version = self._archive_state(managed, existing, reason="update")
            status = "updated"
        elif expected_etag or expected_checksum_sha256:
            raise ClinicalConfigOptimisticLockError(
                f"Clinical config '{config_name}' does not yet exist, so the provided optimistic lock cannot match"
            )
        self._put_current_payload(managed, payload)
        state = self._read_current_state(managed)
        return state.metadata, status, archived_version

    def delete_payload(
        self,
        config_name: ClinicalConfigName,
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> tuple[ClinicalConfigMetadata, ClinicalConfigVersionMetadata]:
        managed = self._resolve(config_name)
        current = self._read_current_state(managed)
        self._assert_optimistic_lock(
            config_name=config_name,
            current=current.metadata,
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )
        archived_version = self._archive_state(managed, current, reason="delete")
        try:
            self._client.remove_object(managed.bucket, managed.object_name)
        except Exception as exc:
            raise _map_storage_error(exc, managed.object_name) from exc
        return _build_metadata(managed, stat=None, checksum_sha256=None), archived_version

    def list_versions(self, config_name: ClinicalConfigName) -> list[ClinicalConfigVersionMetadata]:
        managed = self._resolve(config_name)
        self._ensure_bucket_exists(create=False)
        versions: list[ClinicalConfigVersionMetadata] = []
        try:
            for obj in self._client.list_objects(managed.bucket, prefix=self._versions_prefix(managed), recursive=True):
                if getattr(obj, "is_dir", False):
                    continue
                snapshot = self._get_version_snapshot(managed.bucket, obj.object_name)
                versions.append(_build_version_metadata_from_snapshot(snapshot, bucket=managed.bucket, object_name=obj.object_name, config_name=config_name))
        except Exception as exc:
            raise _map_storage_error(exc, managed.object_name) from exc
        versions.sort(key=lambda item: item.created_at, reverse=True)
        return versions

    def rollback_payload(
        self,
        config_name: ClinicalConfigName,
        version_id: str,
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> tuple[ClinicalConfigMetadata, ClinicalConfigVersionMetadata, ClinicalConfigVersionMetadata | None]:
        managed = self._resolve(config_name)
        target_snapshot = self._read_version_snapshot(managed, version_id)
        existing = self._try_read_current_state(managed)
        archived_version: ClinicalConfigVersionMetadata | None = None
        if existing is not None:
            self._assert_optimistic_lock(
                config_name=config_name,
                current=existing.metadata,
                expected_etag=expected_etag,
                expected_checksum_sha256=expected_checksum_sha256,
            )
            archived_version = self._archive_state(managed, existing, reason="rollback")
        elif expected_etag or expected_checksum_sha256:
            raise ClinicalConfigOptimisticLockError(
                f"Clinical config '{config_name}' does not currently exist, so the provided optimistic lock cannot match"
            )
        payload = target_snapshot.get("payload")
        if not isinstance(payload, dict):
            raise InvalidClinicalConfigError(f"Clinical config version '{version_id}' does not contain a valid payload")
        self._put_current_payload(managed, payload)
        state = self._read_current_state(managed)
        restored = _build_version_metadata_from_snapshot(
            target_snapshot,
            bucket=managed.bucket,
            object_name=self._version_object_name(managed, version_id),
            config_name=config_name,
        )
        return state.metadata, restored, archived_version

    def _resolve(self, config_name: ClinicalConfigName) -> ManagedClinicalConfig:
        managed = self._managed.get(config_name)
        if managed is None:
            raise UnknownClinicalConfigError(f"Clinical config '{config_name}' is not managed")
        return managed

    def _ensure_bucket_exists(self, *, create: bool) -> None:
        bucket = self._settings.clinical_config_bucket
        try:
            exists = self._client.bucket_exists(bucket)
        except Exception as exc:
            raise _map_storage_error(exc, bucket) from exc
        if not exists and create:
            try:
                self._client.make_bucket(bucket)
            except Exception as exc:
                raise _map_storage_error(exc, bucket) from exc

    def _try_stat(self, managed: ManagedClinicalConfig) -> object | None:
        try:
            self._ensure_bucket_exists(create=False)
        except ClinicalConfigNotFoundError:
            return None
        try:
            return self._client.stat_object(managed.bucket, managed.object_name)
        except Exception as exc:
            mapped = _map_storage_error(exc, managed.object_name)
            if isinstance(mapped, ClinicalConfigNotFoundError):
                return None
            raise mapped from exc

    def _stat_object(self, managed: ManagedClinicalConfig) -> object:
        self._ensure_bucket_exists(create=False)
        try:
            return self._client.stat_object(managed.bucket, managed.object_name)
        except Exception as exc:
            raise _map_storage_error(exc, managed.object_name) from exc

    def _try_read_current_state(self, managed: ManagedClinicalConfig) -> CurrentConfigState | None:
        try:
            return self._read_current_state(managed)
        except ClinicalConfigNotFoundError:
            return None

    def _read_current_state(self, managed: ManagedClinicalConfig) -> CurrentConfigState:
        stat = self._stat_object(managed)
        raw_bytes = self._get_bytes(managed.bucket, managed.object_name)
        payload = _decode_payload(raw_bytes, config_name=managed.config_name)
        checksum = _sha256_hexdigest(raw_bytes)
        metadata = _build_metadata(managed, stat=stat, checksum_sha256=checksum)
        return CurrentConfigState(metadata=metadata, payload=payload, raw_bytes=raw_bytes)

    def _put_current_payload(self, managed: ManagedClinicalConfig, payload: dict[str, Any]) -> None:
        self._ensure_bucket_exists(create=True)
        data = _canonical_json_bytes(payload)
        stream = io.BytesIO(data)
        try:
            self._client.put_object(
                managed.bucket,
                managed.object_name,
                data=stream,
                length=len(data),
                content_type="application/json",
            )
        except Exception as exc:
            raise _map_storage_error(exc, managed.object_name) from exc

    def _get_bytes(self, bucket: str, object_name: str) -> bytes:
        response = None
        try:
            response = self._client.get_object(bucket, object_name)
            return response.read()
        except Exception as exc:
            raise _map_storage_error(exc, object_name) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def _archive_state(self, managed: ManagedClinicalConfig, state: CurrentConfigState, *, reason: str) -> ClinicalConfigVersionMetadata:
        version_id = _build_version_id()
        object_name = self._version_object_name(managed, version_id)
        snapshot = {
            "version_id": version_id,
            "config_name": managed.config_name,
            "reason": reason,
            "created_at": datetime.now(UTC).isoformat(),
            "source_etag": state.metadata.etag,
            "source_checksum_sha256": state.metadata.checksum_sha256,
            "payload": state.payload,
        }
        data = _canonical_json_bytes(snapshot)
        stream = io.BytesIO(data)
        try:
            self._client.put_object(
                managed.bucket,
                object_name,
                data=stream,
                length=len(data),
                content_type="application/json",
            )
        except Exception as exc:
            raise _map_storage_error(exc, object_name) from exc
        return _build_version_metadata_from_snapshot(snapshot, bucket=managed.bucket, object_name=object_name, config_name=managed.config_name, size_bytes=len(data))

    def _version_object_name(self, managed: ManagedClinicalConfig, version_id: str) -> str:
        return f"{self._versions_prefix(managed)}/{version_id}.json"

    def _versions_prefix(self, managed: ManagedClinicalConfig) -> str:
        base = managed.object_name.rsplit("/", 1)[0] if "/" in managed.object_name else ""
        versions_segment = self._settings.clinical_config_versions_prefix.strip().strip("/") or "_versions"
        parts = [part for part in (base, versions_segment, managed.config_name) if part]
        return "/".join(parts)

    def _read_version_snapshot(self, managed: ManagedClinicalConfig, version_id: str) -> dict[str, Any]:
        object_name = self._version_object_name(managed, version_id)
        return self._get_version_snapshot(managed.bucket, object_name, version_id=version_id)

    def _get_version_snapshot(self, bucket: str, object_name: str, *, version_id: str | None = None) -> dict[str, Any]:
        raw_bytes = self._get_bytes(bucket, object_name)
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidClinicalConfigError(f"Clinical config version '{version_id or object_name}' contains invalid JSON") from exc
        if not isinstance(payload, dict):
            raise InvalidClinicalConfigError(f"Clinical config version '{version_id or object_name}' must be stored as a JSON object")
        return payload

    def _assert_optimistic_lock(
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


def _build_managed_configs(settings: Settings) -> dict[str, ManagedClinicalConfig]:
    bucket = settings.clinical_config_bucket
    return {
        "drug_dosing_catalog": ManagedClinicalConfig(
            config_name="drug_dosing_catalog",
            bucket=bucket,
            object_name=_join_object_name(settings.clinical_config_prefix, settings.clinical_drug_dosing_catalog_object_name),
        ),
        "marker_ranges": ManagedClinicalConfig(
            config_name="marker_ranges",
            bucket=bucket,
            object_name=_join_object_name(settings.clinical_config_prefix, settings.clinical_marker_ranges_object_name),
        ),
    }


def _build_metadata(managed: ManagedClinicalConfig, *, stat: object | None, checksum_sha256: str | None) -> ClinicalConfigMetadata:
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


def _build_version_metadata_from_snapshot(
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
        source_checksum_sha256=snapshot.get("source_checksum_sha256") if isinstance(snapshot.get("source_checksum_sha256"), str) else None,
        created_at=created_at,
        size_bytes=size_bytes,
    )


def _join_object_name(prefix: str, object_name: str) -> str:
    normalized_prefix = prefix.strip().strip("/")
    normalized_name = object_name.strip().lstrip("/")
    if not normalized_prefix:
        return normalized_name
    return f"{normalized_prefix}/{normalized_name}"


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


def _build_version_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _map_storage_error(exc: Exception, object_name: str) -> ClinicalConfigRepositoryError:
    if isinstance(exc, S3Error):
        code = getattr(exc, "code", "")
        if code in {"NoSuchKey", "NoSuchObject", "NoSuchVersion", "ResourceNotFound", "NoSuchBucket"}:
            if "/_versions/" in object_name:
                return ClinicalConfigVersionNotFoundError(f"Clinical config version '{object_name}' was not found")
            return ClinicalConfigNotFoundError(f"Clinical config '{object_name}' was not found")
        if code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
            return ClinicalConfigRepositoryError(
                "Clinical config storage authentication failed",
                code=ErrorCode.DEPENDENCY_AUTH_FAILED,
            )
        return ClinicalConfigRepositoryError("Clinical config storage request failed")
    if isinstance(exc, (ConnectTimeoutError, TimeoutError)):
        return ClinicalConfigRepositoryError("Clinical config storage timed out", code=ErrorCode.DEPENDENCY_TIMEOUT)
    if isinstance(exc, (MaxRetryError, NewConnectionError, ConnectionError)):
        return ClinicalConfigRepositoryError("Could not reach clinical config storage", code=ErrorCode.DEPENDENCY_UNAVAILABLE)
    return ClinicalConfigRepositoryError("Clinical config storage request failed")
