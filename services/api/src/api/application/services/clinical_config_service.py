from __future__ import annotations

from typing import Any

from api.repositories.clinical_config_repository import (
    ClinicalConfigAlreadyExistsError,
    ClinicalConfigRepository,
    ClinicalConfigRepositoryError,
    ClinicalConfigVersionNotFoundError,
    InvalidClinicalConfigError,
    UnknownClinicalConfigError,
)
from shared.contracts.clinical_config import (
    ClinicalConfigDeleteResponse,
    ClinicalConfigListResponse,
    ClinicalConfigName,
    ClinicalConfigReadResponse,
    ClinicalConfigRollbackResponse,
    ClinicalConfigVersionListResponse,
    ClinicalConfigWriteResponse,
)


class ClinicalConfigService:
    def __init__(self, repository: ClinicalConfigRepository) -> None:
        self._repository = repository

    def list_configs(self) -> ClinicalConfigListResponse:
        return ClinicalConfigListResponse(configs=self._repository.list_configs())

    def get_config(self, config_name: ClinicalConfigName) -> ClinicalConfigReadResponse:
        metadata, payload = self._repository.get_payload(config_name)
        return ClinicalConfigReadResponse(config=metadata, payload=payload)

    def create_config(
        self,
        config_name: ClinicalConfigName,
        payload: dict[str, Any],
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> ClinicalConfigWriteResponse:
        validated = _validate_payload(config_name, payload)
        metadata, archived_version = self._repository.create_payload(
            config_name,
            validated,
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )
        return ClinicalConfigWriteResponse(config=metadata, status="created", archived_version=archived_version)

    def upsert_config(
        self,
        config_name: ClinicalConfigName,
        payload: dict[str, Any],
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> ClinicalConfigWriteResponse:
        validated = _validate_payload(config_name, payload)
        metadata, status, archived_version = self._repository.upsert_payload(
            config_name,
            validated,
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )
        return ClinicalConfigWriteResponse(config=metadata, status=status, archived_version=archived_version)

    def delete_config(
        self,
        config_name: ClinicalConfigName,
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> ClinicalConfigDeleteResponse:
        metadata, archived_version = self._repository.delete_payload(
            config_name,
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )
        return ClinicalConfigDeleteResponse(
            config_name=metadata.config_name,
            bucket=metadata.bucket,
            object_name=metadata.object_name,
            archived_version=archived_version,
        )

    def list_versions(self, config_name: ClinicalConfigName) -> ClinicalConfigVersionListResponse:
        return ClinicalConfigVersionListResponse(
            config_name=config_name,
            versions=self._repository.list_versions(config_name),
        )

    def rollback_config(
        self,
        config_name: ClinicalConfigName,
        version_id: str,
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> ClinicalConfigRollbackResponse:
        metadata, restored_from_version, archived_version = self._repository.rollback_payload(
            config_name,
            version_id,
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )
        return ClinicalConfigRollbackResponse(
            config=metadata,
            restored_from_version=restored_from_version,
            archived_version=archived_version,
        )



def _validate_payload(config_name: ClinicalConfigName, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise InvalidClinicalConfigError("Clinical config payload must be a JSON object")
    if not payload:
        raise InvalidClinicalConfigError("Clinical config payload must not be empty")

    if config_name == "marker_ranges":
        return _validate_marker_ranges_payload(payload)
    if config_name == "drug_dosing_catalog":
        return _validate_drug_dosing_catalog_payload(payload)
    raise UnknownClinicalConfigError(f"Clinical config '{config_name}' is not managed")



def _validate_marker_ranges_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for marker_name, definition in payload.items():
        if not isinstance(marker_name, str) or not marker_name.strip():
            raise InvalidClinicalConfigError("Marker range keys must be non-empty strings")
        if not isinstance(definition, dict):
            raise InvalidClinicalConfigError(f"Marker '{marker_name}' must map to an object")
        if "label" in definition and not isinstance(definition["label"], str):
            raise InvalidClinicalConfigError(f"Marker '{marker_name}' label must be a string")
        if "bands" in definition and not isinstance(definition["bands"], list):
            raise InvalidClinicalConfigError(f"Marker '{marker_name}' bands must be a list when provided")
    return payload



def _validate_drug_dosing_catalog_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required_keys = ("default_agents", "family_query_order", "family_priority", "families")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise InvalidClinicalConfigError(
            "Drug dosing catalog is missing required keys: " + ", ".join(missing)
        )

    default_agents = payload["default_agents"]
    family_query_order = payload["family_query_order"]
    family_priority = payload["family_priority"]
    families = payload["families"]

    if not isinstance(default_agents, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in default_agents.items()):
        raise InvalidClinicalConfigError("Drug dosing catalog default_agents must be an object of strings")
    if not isinstance(family_query_order, list) or not all(isinstance(item, str) for item in family_query_order):
        raise InvalidClinicalConfigError("Drug dosing catalog family_query_order must be a list of strings")
    if not isinstance(family_priority, dict) or not all(isinstance(k, str) and isinstance(v, int) for k, v in family_priority.items()):
        raise InvalidClinicalConfigError("Drug dosing catalog family_priority must be an object of integers")
    if not isinstance(families, dict) or not families:
        raise InvalidClinicalConfigError("Drug dosing catalog families must be a non-empty object")

    for family_name, definition in families.items():
        if not isinstance(family_name, str) or not family_name.strip():
            raise InvalidClinicalConfigError("Drug dosing family names must be non-empty strings")
        if not isinstance(definition, dict):
            raise InvalidClinicalConfigError(f"Drug dosing family '{family_name}' must map to an object")
        keywords = definition.get("keywords")
        query_template = definition.get("query_template")
        if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
            raise InvalidClinicalConfigError(f"Drug dosing family '{family_name}' keywords must be a list of strings")
        if not isinstance(query_template, str) or not query_template.strip():
            raise InvalidClinicalConfigError(f"Drug dosing family '{family_name}' query_template must be a non-empty string")

    return payload


__all__ = [
    "ClinicalConfigAlreadyExistsError",
    "ClinicalConfigRepositoryError",
    "ClinicalConfigService",
    "ClinicalConfigVersionNotFoundError",
    "InvalidClinicalConfigError",
    "UnknownClinicalConfigError",
]
