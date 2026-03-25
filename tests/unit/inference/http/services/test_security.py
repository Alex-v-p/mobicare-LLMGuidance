from __future__ import annotations

from fastapi.testclient import TestClient

from inference.http.dependencies import get_guidance_job_service
from inference.http.main import create_app
from shared.config.settings import get_settings
from shared.contracts.inference import JobRecord


class StubGuidanceJobService:
    async def create(self, request):
        return JobRecord(request_id=request.request_id, status="queued", request=request)


def test_inference_routes_require_internal_service_token_in_prod(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET_KEY", "secret")
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "token")
    get_settings.cache_clear()
    try:
        app = create_app()
        app.dependency_overrides[get_guidance_job_service] = lambda: StubGuidanceJobService()
        with TestClient(app) as client:
            response = client.post("/guidance/jobs", json={"request_id": "req-1", "question": "q", "patient_variables": {}, "options": {}})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "AUTH_TOKEN_INVALID"
    finally:
        get_settings.cache_clear()
