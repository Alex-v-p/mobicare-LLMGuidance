from __future__ import annotations

from datetime import datetime, timezone

from api.auth import create_access_token
from api.dependencies import get_clinical_config_service
from shared.contracts.clinical_config import (
    ClinicalConfigDeleteResponse,
    ClinicalConfigListResponse,
    ClinicalConfigMetadata,
    ClinicalConfigReadResponse,
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
            ),
            payload={"potassium": {"label": "Potassium", "bands": [{"low": 3.5, "high": 5.0}]}},
        )

    def create_config(self, config_name, payload):
        return ClinicalConfigWriteResponse(
            config=ClinicalConfigMetadata(
                config_name=config_name,
                bucket="guidance-config",
                object_name=f"clinical/{config_name}.json",
                exists_in_minio=True,
            ),
            status="created",
        )

    def upsert_config(self, config_name, payload):
        return ClinicalConfigWriteResponse(
            config=ClinicalConfigMetadata(
                config_name=config_name,
                bucket="guidance-config",
                object_name=f"clinical/{config_name}.json",
                exists_in_minio=True,
            ),
            status="updated",
        )

    def delete_config(self, config_name):
        return ClinicalConfigDeleteResponse(
            config_name=config_name,
            bucket="guidance-config",
            object_name=f"clinical/{config_name}.json",
        )



def _auth_headers() -> dict[str, str]:
    token, _ = create_access_token(email="admin@example.com")
    return {"Authorization": f"Bearer {token}"}



def test_list_clinical_configs_requires_auth(api_app, api_client):
    api_app.dependency_overrides[get_clinical_config_service] = lambda: StubClinicalConfigService()

    response = api_client.get("/clinical-configs")

    assert response.status_code == 401



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
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "updated"
    assert response.json()["config"]["object_name"] == "clinical/drug_dosing_catalog.json"
