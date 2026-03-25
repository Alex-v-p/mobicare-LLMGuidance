from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import create_app
from shared.config.settings import get_settings


def test_guidance_routes_require_bearer_token_in_prod(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET_KEY", "secret")
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "token")
    get_settings.cache_clear()
    try:
        app = create_app()
        with TestClient(app) as client:
            response = client.post("/guidance/jobs", json={"question": "q", "patient": {"values": {}}})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "AUTH_TOKEN_INVALID"
    finally:
        get_settings.cache_clear()
