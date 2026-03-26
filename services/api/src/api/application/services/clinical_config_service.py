from __future__ import annotations

from typing import Any

from api.application.validators.clinical_config import (
    normalize_clinical_config_name,
    validate_clinical_config_payload,
)
from api.infrastructure.repositories.clinical_config import (
    ClinicalConfigAlreadyExistsError,
    ClinicalConfigRepository,
    ClinicalConfigRepositoryError,
    ClinicalConfigVersionNotFoundError,
    InvalidClinicalConfigError,
    UnknownClinicalConfigError,
)
from shared.contracts.clinical_config import (
    ClinicalConfigDeleteResponse,
    ClinicalConfigName,
    ClinicalConfigListResponse,
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

    def get_config(self, config_name: ClinicalConfigName | str) -> ClinicalConfigReadResponse:
        normalized_name = normalize_clinical_config_name(config_name)
        metadata, payload = self._repository.get_payload(normalized_name)
        return ClinicalConfigReadResponse(config=metadata, payload=payload)

    def create_config(
        self,
        config_name: ClinicalConfigName | str,
        payload: dict[str, Any],
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> ClinicalConfigWriteResponse:
        normalized_name, validated = validate_clinical_config_payload(config_name, payload)
        metadata, archived_version = self._repository.create_payload(
            normalized_name,
            validated,
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )
        return ClinicalConfigWriteResponse(config=metadata, status="created", archived_version=archived_version)

    def upsert_config(
        self,
        config_name: ClinicalConfigName | str,
        payload: dict[str, Any],
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> ClinicalConfigWriteResponse:
        normalized_name, validated = validate_clinical_config_payload(config_name, payload)
        metadata, status, archived_version = self._repository.upsert_payload(
            normalized_name,
            validated,
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )
        return ClinicalConfigWriteResponse(config=metadata, status=status, archived_version=archived_version)

    def delete_config(
        self,
        config_name: ClinicalConfigName | str,
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> ClinicalConfigDeleteResponse:
        normalized_name = normalize_clinical_config_name(config_name)
        metadata, archived_version = self._repository.delete_payload(
            normalized_name,
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )
        return ClinicalConfigDeleteResponse(
            config_name=metadata.config_name,
            bucket=metadata.bucket,
            object_name=metadata.object_name,
            archived_version=archived_version,
        )

    def list_versions(self, config_name: ClinicalConfigName | str) -> ClinicalConfigVersionListResponse:
        normalized_name = normalize_clinical_config_name(config_name)
        return ClinicalConfigVersionListResponse(
            config_name=normalized_name,
            versions=self._repository.list_versions(normalized_name),
        )

    def rollback_config(
        self,
        config_name: ClinicalConfigName | str,
        version_id: str,
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> ClinicalConfigRollbackResponse:
        normalized_name = normalize_clinical_config_name(config_name)
        metadata, restored_from_version, archived_version = self._repository.rollback_payload(
            normalized_name,
            version_id,
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )
        return ClinicalConfigRollbackResponse(
            config=metadata,
            restored_from_version=restored_from_version,
            archived_version=archived_version,
        )


__all__ = [
    "ClinicalConfigAlreadyExistsError",
    "ClinicalConfigRepositoryError",
    "ClinicalConfigService",
    "ClinicalConfigVersionNotFoundError",
    "InvalidClinicalConfigError",
    "UnknownClinicalConfigError",
]
