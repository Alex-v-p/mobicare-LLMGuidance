from __future__ import annotations

from fastapi.testclient import TestClient

from api.dependencies import get_guidance_service
from api.main import create_app
from shared.contracts.inference import ApiGuidanceJobStatus, JobAcceptedResponse


class StubGuidanceService:
    async def submit_job(self, request):
        return JobAcceptedResponse(
            job_id="job-123",
            request_id=request.request_id,
            status_url="http://testserver/guidance/jobs/job-123",
        )

    async def get_job_status(self, job_id: str):
        return ApiGuidanceJobStatus(job_id=job_id, request_id="req-123", status="completed", answer="ok", model="model")


def test_create_guidance_job_returns_accepted_response(guidance_request):
    app = create_app(bootstrap_minio_on_startup=False)
    app.dependency_overrides[get_guidance_service] = lambda: StubGuidanceService()

    with TestClient(app) as client:
        response = client.post("/guidance/jobs", json=guidance_request.model_dump(mode="json"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-123"
    assert payload["request_id"] == guidance_request.request_id


def test_get_guidance_job_status_returns_payload():
    app = create_app(bootstrap_minio_on_startup=False)
    app.dependency_overrides[get_guidance_service] = lambda: StubGuidanceService()

    with TestClient(app) as client:
        response = client.get("/guidance/jobs/job-123")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
