from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

from minio import Minio
from minio.error import S3Error
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError, NewConnectionError

from shared.config import Settings, get_settings
from shared.contracts.clinical_config import ClinicalConfigMetadata, ClinicalConfigName
from shared.contracts.error_codes import ErrorCode


@dataclass(frozen=True)
class ManagedClinicalConfig:
    config_name: ClinicalConfigName
    bucket: str
    object_name: str


class ClinicalConfigRepositoryError(RuntimeError):
    def __init__(self, message: str, *, code: str = ErrorCode.CLINICAL_CONFIG_STORAGE_UNAVAILABLE) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ClinicalConfigNotFoundError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_NOT_FOUND)


class ClinicalConfigAlreadyExistsError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_CONFLICT)


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
            stat = self._try_stat(managed)
            configs.append(_build_metadata(managed, stat=stat))
        return configs

    def get_payload(self, config_name: ClinicalConfigName) -> tuple[ClinicalConfigMetadata, dict[str, Any]]:
        managed = self._resolve(config_name)
        stat = self._stat_object(managed)
        payload = self._get_json(managed)
        if not isinstance(payload, dict):
            raise InvalidClinicalConfigError(f"Clinical config '{config_name}' must be stored as a JSON object")
        return _build_metadata(managed, stat=stat), payload

    def create_payload(self, config_name: ClinicalConfigName, payload: dict[str, Any]) -> ClinicalConfigMetadata:
        managed = self._resolve(config_name)
        if self._try_stat(managed) is not None:
            raise ClinicalConfigAlreadyExistsError(f"Clinical config '{config_name}' already exists in MinIO")
        self._put_json(managed, payload)
        return _build_metadata(managed, stat=self._stat_object(managed))

    def upsert_payload(self, config_name: ClinicalConfigName, payload: dict[str, Any]) -> tuple[ClinicalConfigMetadata, str]:
        managed = self._resolve(config_name)
        existed = self._try_stat(managed) is not None
        self._put_json(managed, payload)
        status = "updated" if existed else "created"
        return _build_metadata(managed, stat=self._stat_object(managed)), status

    def delete_payload(self, config_name: ClinicalConfigName) -> ClinicalConfigMetadata:
        managed = self._resolve(config_name)
        self._stat_object(managed)
        try:
            self._client.remove_object(managed.bucket, managed.object_name)
        except Exception as exc:
            raise _map_storage_error(exc, managed.object_name) from exc
        return _build_metadata(managed, stat=None)

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

    def _get_json(self, managed: ManagedClinicalConfig) -> dict[str, Any] | list[Any] | str | int | float | bool | None:
        self._ensure_bucket_exists(create=False)
        response = None
        try:
            response = self._client.get_object(managed.bucket, managed.object_name)
            return json.loads(response.read().decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidClinicalConfigError(f"Clinical config '{managed.config_name}' contains invalid JSON") from exc
        except Exception as exc:
            raise _map_storage_error(exc, managed.object_name) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def _put_json(self, managed: ManagedClinicalConfig, payload: dict[str, Any]) -> None:
        self._ensure_bucket_exists(create=True)
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
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



def _build_metadata(managed: ManagedClinicalConfig, *, stat: object | None) -> ClinicalConfigMetadata:
    return ClinicalConfigMetadata(
        config_name=managed.config_name,
        bucket=managed.bucket,
        object_name=managed.object_name,
        exists_in_minio=stat is not None,
        content_type="application/json",
        size_bytes=getattr(stat, "size", None) if stat is not None else None,
        etag=getattr(stat, "etag", None) if stat is not None else None,
        last_modified=getattr(stat, "last_modified", None) if stat is not None else None,
    )



def _join_object_name(prefix: str, object_name: str) -> str:
    normalized_prefix = prefix.strip().strip("/")
    normalized_name = object_name.strip().lstrip("/")
    if not normalized_prefix:
        return normalized_name
    return f"{normalized_prefix}/{normalized_name}"



def _map_storage_error(exc: Exception, object_name: str) -> ClinicalConfigRepositoryError:
    if isinstance(exc, S3Error):
        code = getattr(exc, "code", "")
        if code in {"NoSuchKey", "NoSuchObject", "NoSuchVersion", "ResourceNotFound", "NoSuchBucket"}:
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
