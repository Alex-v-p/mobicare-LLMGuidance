from __future__ import annotations

import pytest

from api.application.services.clinical_config_service import ClinicalConfigService
from api.repositories.clinical_config_repository import InvalidClinicalConfigError
from shared.contracts.clinical_config import ClinicalConfigMetadata


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

    def create_payload(self, config_name, payload):
        return ClinicalConfigMetadata(
            config_name=config_name,
            bucket="guidance-config",
            object_name=f"clinical/{config_name}.json",
            exists_in_minio=True,
        )

    def upsert_payload(self, config_name, payload):
        return self.create_payload(config_name, payload), "updated"

    def delete_payload(self, config_name):
        return ClinicalConfigMetadata(
            config_name=config_name,
            bucket="guidance-config",
            object_name=f"clinical/{config_name}.json",
            exists_in_minio=False,
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
