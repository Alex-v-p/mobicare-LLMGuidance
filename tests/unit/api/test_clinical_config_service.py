from __future__ import annotations

from datetime import datetime, timezone

import pytest

from api.application.services.clinical_config_service import ClinicalConfigService
from api.repositories.clinical_config_repository import InvalidClinicalConfigError
from shared.contracts.clinical_config import ClinicalConfigMetadata, ClinicalConfigVersionMetadata


class StubClinicalConfigRepository:
    def list_configs(self):
        return []

    def get_payload(self, config_name):
        return (
            ClinicalConfigMetadata(
                config_name=config_name,
                bucket="guidance-config",
                object_name=f"clinical/{config_name}.json",
                exists_in_minio=True,
            ),
            {},
        )

    def create_payload(self, config_name, payload, *, expected_etag=None, expected_checksum_sha256=None):
        return (
            ClinicalConfigMetadata(
                config_name=config_name,
                bucket="guidance-config",
                object_name=f"clinical/{config_name}.json",
                exists_in_minio=True,
            ),
            None,
        )

    def upsert_payload(self, config_name, payload, *, expected_etag=None, expected_checksum_sha256=None):
        return (
            ClinicalConfigMetadata(
                config_name=config_name,
                bucket="guidance-config",
                object_name=f"clinical/{config_name}.json",
                exists_in_minio=True,
            ),
            "updated",
            ClinicalConfigVersionMetadata(
                config_name=config_name,
                version_id="v1",
                bucket="guidance-config",
                object_name=f"clinical/_versions/{config_name}/v1.json",
                reason="update",
                created_at=datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc),
            ),
        )

    def delete_payload(self, config_name, *, expected_etag=None, expected_checksum_sha256=None):
        return (
            ClinicalConfigMetadata(
                config_name=config_name,
                bucket="guidance-config",
                object_name=f"clinical/{config_name}.json",
                exists_in_minio=False,
            ),
            ClinicalConfigVersionMetadata(
                config_name=config_name,
                version_id="v1",
                bucket="guidance-config",
                object_name=f"clinical/_versions/{config_name}/v1.json",
                reason="delete",
                created_at=datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc),
            ),
        )

    def list_versions(self, config_name):
        return []

    def rollback_payload(self, config_name, version_id, *, expected_etag=None, expected_checksum_sha256=None):
        return (
            ClinicalConfigMetadata(
                config_name=config_name,
                bucket="guidance-config",
                object_name=f"clinical/{config_name}.json",
                exists_in_minio=True,
            ),
            ClinicalConfigVersionMetadata(
                config_name=config_name,
                version_id=version_id,
                bucket="guidance-config",
                object_name=f"clinical/_versions/{config_name}/{version_id}.json",
                reason="update",
                created_at=datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc),
            ),
            None,
        )


@pytest.fixture
def service() -> ClinicalConfigService:
    return ClinicalConfigService(repository=StubClinicalConfigRepository())


def test_drug_dosing_catalog_requires_expected_top_level_keys(service: ClinicalConfigService):
    with pytest.raises(InvalidClinicalConfigError):
        service.upsert_config("drug_dosing_catalog", {"families": {}})


def test_marker_ranges_reject_non_object_definition(service: ClinicalConfigService):
    with pytest.raises(InvalidClinicalConfigError):
        service.upsert_config("marker_ranges", {"potassium": []})


def test_upsert_returns_archived_version_metadata(service: ClinicalConfigService):
    response = service.upsert_config("marker_ranges", {"potassium": {"label": "Potassium"}})

    assert response.status == "updated"
    assert response.archived_version is not None
    assert response.archived_version.version_id == "v1"


def test_rollback_returns_restored_version(service: ClinicalConfigService):
    response = service.rollback_config("marker_ranges", "v2")

    assert response.status == "rolled_back"
    assert response.restored_from_version.version_id == "v2"
