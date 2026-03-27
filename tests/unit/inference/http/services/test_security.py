from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from inference.infrastructure.http.dependencies import get_guidance_job_service
from inference.infrastructure.http.exceptions import register_exception_handlers
from inference.infrastructure.http.routes.guidance import router as guidance_router
from inference.infrastructure.http.security import require_internal_service_request
from shared.config import get_inference_settings
from shared.contracts.inference import JobRecord


class StubGuidanceJobService:
    async def create(self, request):
        return JobRecord(request_id=request.request_id, status="queued", request=request)


def test_inference_routes_require_internal_service_token_in_prod(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "token")
    get_inference_settings.cache_clear()

    try:
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(
            guidance_router,
            dependencies=[Depends(require_internal_service_request)],
        )
        app.dependency_overrides[get_guidance_job_service] = lambda: StubGuidanceJobService()

        client = TestClient(app)
        try:
            response = client.post(
                "/guidance/jobs",
                json={
                    "request_id": "req-1",
                    "question": "q",
                    "patient_variables": {},
                    "options": {},
                },
            )
        finally:
            client.close()

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "AUTH_TOKEN_INVALID"
    finally:
        get_inference_settings.cache_clear()