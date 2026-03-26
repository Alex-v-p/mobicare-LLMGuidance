from __future__ import annotations

from fastapi.testclient import TestClient

from api.dependencies import get_ingestion_service
from api.main import create_app
from shared.contracts.ingestion import IngestionCollectionDeleteResponse, IngestionJobAcceptedResponse, IngestionJobRecord


class StubIngestionService:
    async def submit_job(self, payload):
        return IngestionJobAcceptedResponse(job_id="ing-123", status_url="http://upstream/jobs/ing-123")

    async def get_job_status(self, job_id: str):
        return IngestionJobRecord(job_id=job_id, status="completed")

    async def delete_collection(self):
        return IngestionCollectionDeleteResponse(collection="guidance_chunks", existed=True)


def test_create_ingestion_job_rewrites_status_url():
    app = create_app(bootstrap_minio_on_startup=False)
    app.dependency_overrides[get_ingestion_service] = lambda: StubIngestionService()

    with TestClient(app) as client:
        response = client.post("/ingestion/jobs", json={"options": {}})

    assert response.status_code == 202
    assert response.json()["status_url"] == "http://testserver/ingestion/jobs/ing-123"


def test_delete_collection_route_returns_response():
    app = create_app(bootstrap_minio_on_startup=False)
    app.dependency_overrides[get_ingestion_service] = lambda: StubIngestionService()

    with TestClient(app) as client:
        response = client.delete("/ingestion/collection")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "collection": "guidance_chunks", "existed": True}
