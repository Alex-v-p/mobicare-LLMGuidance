from __future__ import annotations

from fastapi.testclient import TestClient

from inference.http.dependencies import (
    get_guidance_job_service,
    get_inference_settings,
)
from inference.http.main import create_app
from shared.config.settings import get_settings
from shared.contracts.inference import JobRecord


class StubGuidanceJobService:
    async def create(self, request):
        return JobRecord(request_id=request.request_id, status="queued", request=request)


def test_inference_routes_require_internal_service_token_in_prod(monkeypatch):
    from unittest.mock import MagicMock
    import inference.http.main as http_main

    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET_KEY", "secret")
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "token")

    get_settings.cache_clear()
    get_inference_settings.cache_clear()

    try:
        mock_document_store = MagicMock()
        mock_document_store.client = MagicMock()

        mock_guidance_job_result_store = MagicMock()
        mock_ingestion_job_result_store = MagicMock()

        monkeypatch.setattr(http_main, "get_document_store", lambda: mock_document_store)
        monkeypatch.setattr(http_main, "get_guidance_job_result_store", lambda: mock_guidance_job_result_store)
        monkeypatch.setattr(http_main, "get_ingestion_job_result_store", lambda: mock_ingestion_job_result_store)

        app = create_app()
        app.dependency_overrides[get_guidance_job_service] = lambda: StubGuidanceJobService()

        with TestClient(app) as client:
            response = client.post(
                "/guidance/jobs",
                json={
                    "request_id": "req-1",
                    "question": "q",
                    "patient_variables": {},
                    "options": {},
                },
            )

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "AUTH_TOKEN_INVALID"
    finally:
        get_settings.cache_clear()
        get_inference_settings.cache_clear()