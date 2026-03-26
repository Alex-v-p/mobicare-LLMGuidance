from __future__ import annotations

from datetime import datetime, timezone

from api.dependencies import get_clinical_config_service
from shared.contracts.clinical_config import (
    ClinicalConfigDeleteResponse,
    ClinicalConfigListResponse,
    ClinicalConfigMetadata,
    ClinicalConfigReadResponse,
    ClinicalConfigRollbackResponse,
    ClinicalConfigVersionListResponse,
    ClinicalConfigVersionMetadata,
    ClinicalConfigWriteResponse,
)


class StubClinicalConfigService:
    def list_configs(self):
        return ClinicalConfigListResponse(
            configs=[
                ClinicalConfigMetadata(
                    config_name="marker_ranges",
                    bucket="guidance-config",
                    object_name="clinical/marker_ranges.json",
                    exists_in_minio=True,
                    size_bytes=123,
                    etag="etag-1",
                    checksum_sha256="sha-1",
                    last_modified=datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc),
                )
            ]
        )

    def get_config(self, config_name):
        return ClinicalConfigReadResponse(
            config=ClinicalConfigMetadata(
                config_name=config_name,
                bucket="guidance-config",
                object_name=f"clinical/{config_name}.json",
                exists_in_minio=True,
                etag="etag-2",
                checksum_sha256="sha-2",
            ),
            payload={"potassium": {"label": "Potassium", "bands": [{"low": 3.5, "high": 5.0}]}},
        )

    def create_config(self, config_name, payload, *, expected_etag=None, expected_checksum_sha256=None):
        return ClinicalConfigWriteResponse(
            config=ClinicalConfigMetadata(
                config_name=config_name,
                bucket="guidance-config",
                object_name=f"clinical/{config_name}.json",
                exists_in_minio=True,
                etag="etag-3",
                checksum_sha256="sha-3",
            ),
            status="created",
        )

    def upsert_config(self, config_name, payload, *, expected_etag=None, expected_checksum_sha256=None):
        return ClinicalConfigWriteResponse(
            config=ClinicalConfigMetadata(
                config_name=config_name,
                bucket="guidance-config",
                object_name=f"clinical/{config_name}.json",
                exists_in_minio=True,
                etag="etag-4",
                checksum_sha256="sha-4",
            ),
            status="updated",
            archived_version=ClinicalConfigVersionMetadata(
                config_name=config_name,
                version_id="20260325T100000000000Z",
                bucket="guidance-config",
                object_name=f"clinical/_versions/{config_name}/20260325T100000000000Z.json",
                reason="update",
                source_etag="etag-old",
                source_checksum_sha256="sha-old",
                created_at=datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc),
            ),
        )

    def delete_config(self, config_name, *, expected_etag=None, expected_checksum_sha256=None):
        return ClinicalConfigDeleteResponse(
            config_name=config_name,
            bucket="guidance-config",
            object_name=f"clinical/{config_name}.json",
            archived_version=ClinicalConfigVersionMetadata(
                config_name=config_name,
                version_id="20260325T100000000001Z",
                bucket="guidance-config",
                object_name=f"clinical/_versions/{config_name}/20260325T100000000001Z.json",
                reason="delete",
                source_etag="etag-del",
                source_checksum_sha256="sha-del",
                created_at=datetime(2026, 3, 25, 10, 1, tzinfo=timezone.utc),
            ),
        )

    def list_versions(self, config_name):
        return ClinicalConfigVersionListResponse(
            config_name=config_name,
            versions=[
                ClinicalConfigVersionMetadata(
                    config_name=config_name,
                    version_id="20260325T095959000000Z",
                    bucket="guidance-config",
                    object_name=f"clinical/_versions/{config_name}/20260325T095959000000Z.json",
                    reason="update",
                    source_etag="etag-prev",
                    source_checksum_sha256="sha-prev",
                    created_at=datetime(2026, 3, 25, 9, 59, 59, tzinfo=timezone.utc),
                )
            ],
        )

    def rollback_config(self, config_name, version_id, *, expected_etag=None, expected_checksum_sha256=None):
        return ClinicalConfigRollbackResponse(
            config=ClinicalConfigMetadata(
                config_name=config_name,
                bucket="guidance-config",
                object_name=f"clinical/{config_name}.json",
                exists_in_minio=True,
                etag="etag-rollback",
                checksum_sha256="sha-rollback",
            ),
            restored_from_version=ClinicalConfigVersionMetadata(
                config_name=config_name,
                version_id=version_id,
                bucket="guidance-config",
                object_name=f"clinical/_versions/{config_name}/{version_id}.json",
                reason="update",
                source_etag="etag-source",
                source_checksum_sha256="sha-source",
                created_at=datetime(2026, 3, 25, 9, 59, 59, tzinfo=timezone.utc),
            ),
        )



def test_list_clinical_configs_without_auth(api_app, api_client):
    api_app.dependency_overrides[get_clinical_config_service] = lambda: StubClinicalConfigService()

    response = api_client.get("/clinical-configs")

    assert response.status_code == 200
    assert response.json()["configs"][0]["config_name"] == "marker_ranges"



def test_get_clinical_config_returns_concurrency_headers(api_app, api_client):
    api_app.dependency_overrides[get_clinical_config_service] = lambda: StubClinicalConfigService()

    response = api_client.get("/clinical-configs/marker_ranges")

    assert response.status_code == 200
    assert response.headers["etag"] == "etag-2"
    assert response.headers["x-content-sha256"] == "sha-2"



def test_upsert_clinical_config_returns_updated_status(api_app, api_client):
    api_app.dependency_overrides[get_clinical_config_service] = lambda: StubClinicalConfigService()

    response = api_client.put(
        "/clinical-configs/drug_dosing_catalog",
        json={
            "payload": {
                "default_agents": {"beta_blocker": "carvedilol"},
                "family_query_order": ["beta_blocker"],
                "family_priority": {"beta_blocker": 1},
                "families": {"beta_blocker": {"keywords": ["beta-blocker"], "query_template": "{agent} beta-blocker dose"}},
            }
        },
        headers={"If-Match": "etag-old", "X-Content-SHA256": "sha-old"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "updated"
    assert response.json()["archived_version"]["reason"] == "update"
    assert response.headers["etag"] == "etag-4"
    assert response.headers["x-content-sha256"] == "sha-4"



def test_list_versions_returns_version_history(api_app, api_client):
    api_app.dependency_overrides[get_clinical_config_service] = lambda: StubClinicalConfigService()

    response = api_client.get("/clinical-configs/marker_ranges/versions")

    assert response.status_code == 200
    assert response.json()["versions"][0]["version_id"] == "20260325T095959000000Z"



def test_rollback_returns_restored_version(api_app, api_client):
    api_app.dependency_overrides[get_clinical_config_service] = lambda: StubClinicalConfigService()

    response = api_client.post(
        "/clinical-configs/marker_ranges/rollback",
        json={"version_id": "20260325T095959000000Z"},
        headers={"If-Match": "etag-2", "X-Content-SHA256": "sha-2"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "rolled_back"
    assert response.json()["restored_from_version"]["version_id"] == "20260325T095959000000Z"
    assert response.headers["etag"] == "etag-rollback"


class ConflictClinicalConfigService:
    def upsert_config(self, config_name, payload, *, expected_etag=None, expected_checksum_sha256=None):
        from api.infrastructure.repositories.clinical_config import ClinicalConfigOptimisticLockError

        raise ClinicalConfigOptimisticLockError("etag mismatch")


def test_upsert_clinical_config_maps_optimistic_lock_to_conflict(api_app, api_client):
    api_app.dependency_overrides[get_clinical_config_service] = lambda: ConflictClinicalConfigService()

    response = api_client.put(
        "/clinical-configs/marker_ranges",
        json={"payload": {"potassium": {"label": "Potassium", "bands": [{"low": 3.5, "high": 5.0}]}}},
        headers={"If-Match": "etag-old"},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CLINICAL_CONFIG_OPTIMISTIC_LOCK_FAILED"
    assert response.json()["error"]["details"]["config_name"] == "marker_ranges"
