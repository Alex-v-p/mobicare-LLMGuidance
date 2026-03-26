from __future__ import annotations

import json
from dataclasses import dataclass
from time import monotonic
from typing import Any

from minio import Minio

from shared.bootstrap import bootstrap_minio_resources, create_minio_client_from_settings, load_clinical_config_default_payload
from shared.config import InferenceSettings, get_inference_settings
from shared.observability import get_logger


logger = get_logger(__name__, service="inference")


@dataclass(slots=True)
class _CacheEntry:
    value: dict[str, Any]
    expires_at: float


_CACHE: dict[str, _CacheEntry] = {}


class ClinicalConfigRepository:
    def __init__(self, settings: InferenceSettings | None = None, client: Minio | None = None) -> None:
        self._settings = settings or get_inference_settings()
        self._client = client

    def load_marker_ranges_payload(self) -> dict[str, Any]:
        return self._load_json(
            packaged_filename="marker_ranges.json",
            object_name=self._settings.clinical_marker_ranges_object_name,
        )

    def load_drug_dosing_catalog_payload(self) -> dict[str, Any]:
        return self._load_json(
            packaged_filename="drug_dosing_catalog.json",
            object_name=self._settings.clinical_drug_dosing_catalog_object_name,
        )

    def _load_json(self, *, packaged_filename: str, object_name: str) -> dict[str, Any]:
        cache_key = self._cache_key(packaged_filename=packaged_filename, object_name=object_name)
        cached = _get_cached_payload(cache_key)
        if cached is not None:
            return cached

        payload: dict[str, Any]
        source = (self._settings.clinical_config_source or "packaged").strip().lower()
        if source == "minio":
            try:
                payload = self._load_from_minio(object_name)
            except Exception as exc:
                logger.warning(
                    "clinical_config_minio_load_failed",
                    extra={
                        "event": "clinical_config_minio_load_failed",
                        "dependency": "minio",
                        "error_code": "CLINICAL_CONFIG_LOAD_FAILED",
                    },
                    exc_info=exc,
                )
                payload = self._load_from_package(packaged_filename)
        else:
            payload = self._load_from_package(packaged_filename)

        _store_cached_payload(cache_key, payload, ttl_seconds=max(self._settings.clinical_config_cache_seconds, 0))
        return payload

    def _load_from_package(self, packaged_filename: str) -> dict[str, Any]:
        config_name = packaged_filename.removesuffix(".json")
        return load_clinical_config_default_payload(config_name)

    def _load_from_minio(self, object_name: str) -> dict[str, Any]:
        client = self._client or create_minio_client_from_settings(self._settings)
        bootstrap_minio_resources(settings=self._settings, client=client)
        resolved_name = _join_object_name(self._settings.clinical_config_prefix, object_name)
        response = client.get_object(self._settings.clinical_config_bucket, resolved_name)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()

    def _cache_key(self, *, packaged_filename: str, object_name: str) -> str:
        return "|".join(
            [
                self._settings.clinical_config_source,
                self._settings.clinical_config_bucket,
                self._settings.clinical_config_prefix,
                packaged_filename,
                object_name,
            ]
        )



def load_marker_ranges_payload(settings: InferenceSettings | None = None, client: Minio | None = None) -> dict[str, Any]:
    return ClinicalConfigRepository(settings=settings, client=client).load_marker_ranges_payload()



def load_drug_dosing_catalog_payload(settings: InferenceSettings | None = None, client: Minio | None = None) -> dict[str, Any]:
    return ClinicalConfigRepository(settings=settings, client=client).load_drug_dosing_catalog_payload()



def clear_clinical_config_cache() -> None:
    _CACHE.clear()



def _get_cached_payload(key: str) -> dict[str, Any] | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    if entry.expires_at < monotonic():
        _CACHE.pop(key, None)
        return None
    return entry.value



def _store_cached_payload(key: str, payload: dict[str, Any], *, ttl_seconds: int) -> None:
    if ttl_seconds <= 0:
        _CACHE.pop(key, None)
        return
    _CACHE[key] = _CacheEntry(value=payload, expires_at=monotonic() + ttl_seconds)



def _join_object_name(prefix: str, object_name: str) -> str:
    normalized_prefix = prefix.strip().strip("/")
    normalized_name = object_name.strip().lstrip("/")
    if not normalized_prefix:
        return normalized_name
    return f"{normalized_prefix}/{normalized_name}"
