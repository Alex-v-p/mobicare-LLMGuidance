from __future__ import annotations

from fastapi.testclient import TestClient

from api.dependencies import get_health_service
from api.main import create_app
from shared.contracts.health import DependencyStatus, HealthReport


class StubHealthService:
    async def report(self) -> HealthReport:
        return HealthReport(status="ok", deps={"redis": DependencyStatus(ok=True, url="redis://redis:6379/0")})


def test_health_route_returns_report():
    app = create_app()
    app.dependency_overrides[get_health_service] = lambda: StubHealthService()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"]


def test_metrics_route_returns_text_payload():
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "app_requests_total" in response.text
