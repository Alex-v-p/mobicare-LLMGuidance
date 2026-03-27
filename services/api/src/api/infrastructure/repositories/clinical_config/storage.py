from __future__ import annotations

import io
import json
from typing import Any

from minio import Minio

from api.infrastructure.repositories.clinical_config.errors import ClinicalConfigNotFoundError, map_storage_error
from api.infrastructure.repositories.clinical_config.models import ManagedClinicalConfig
from shared.bootstrap import bootstrap_minio_resources, ensure_minio_bucket
from shared.config import ApiSettings


class ClinicalConfigStorage:
    def __init__(self, *, client: Minio, settings: ApiSettings) -> None:
        self._client = client
        self._settings = settings

    def ensure_bootstrapped(self) -> None:
        try:
            bootstrap_minio_resources(settings=self._settings, client=self._client)
        except Exception as exc:
            raise map_storage_error(exc, self._settings.clinical_config_bucket) from exc

    def ensure_bucket_exists(self, *, create: bool) -> None:
        bucket = self._settings.clinical_config_bucket
        try:
            if create:
                ensure_minio_bucket(self._client, bucket)
                return
            self._client.bucket_exists(bucket)
        except Exception as exc:
            raise map_storage_error(exc, bucket) from exc

    def try_stat(self, managed: ManagedClinicalConfig) -> object | None:
        try:
            self.ensure_bucket_exists(create=False)
        except ClinicalConfigNotFoundError:
            return None
        try:
            return self._client.stat_object(managed.bucket, managed.object_name)
        except Exception as exc:
            mapped = map_storage_error(exc, managed.object_name)
            if isinstance(mapped, ClinicalConfigNotFoundError):
                return None
            raise mapped from exc

    def stat_object(self, managed: ManagedClinicalConfig) -> object:
        self.ensure_bucket_exists(create=False)
        try:
            return self._client.stat_object(managed.bucket, managed.object_name)
        except Exception as exc:
            raise map_storage_error(exc, managed.object_name) from exc

    def get_bytes(self, bucket: str, object_name: str) -> bytes:
        response = None
        try:
            response = self._client.get_object(bucket, object_name)
            return response.read()
        except Exception as exc:
            raise map_storage_error(exc, object_name) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def put_json_bytes(self, managed: ManagedClinicalConfig, data: bytes) -> None:
        self.ensure_bucket_exists(create=True)
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
            raise map_storage_error(exc, managed.object_name) from exc

    def put_version_snapshot(self, *, bucket: str, object_name: str, snapshot: dict[str, Any]) -> int:
        data = json.dumps(snapshot, indent=2, sort_keys=True).encode("utf-8")
        stream = io.BytesIO(data)
        try:
            self._client.put_object(
                bucket,
                object_name,
                data=stream,
                length=len(data),
                content_type="application/json",
            )
        except Exception as exc:
            raise map_storage_error(exc, object_name) from exc
        return len(data)

    def remove_object(self, managed: ManagedClinicalConfig) -> None:
        try:
            self._client.remove_object(managed.bucket, managed.object_name)
        except Exception as exc:
            raise map_storage_error(exc, managed.object_name) from exc

    def list_object_names(self, *, bucket: str, prefix: str) -> list[str]:
        self.ensure_bucket_exists(create=False)
        object_names: list[str] = []
        try:
            for obj in self._client.list_objects(bucket, prefix=prefix, recursive=True):
                if getattr(obj, "is_dir", False):
                    continue
                object_names.append(obj.object_name)
        except Exception as exc:
            raise map_storage_error(exc, prefix) from exc
        return object_names


__all__ = ["ClinicalConfigStorage"]
