from __future__ import annotations

from fastapi.testclient import TestClient

from api.dependencies import get_health_service
from api.main import create_app
from shared.contracts.health import DependencyStatus, HealthReport


class StubHealthService:
    async def report(self) -> HealthReport:
        return HealthReport(status="ok", deps={"redis": DependencyStatus(ok=True, url="redis://redis:6379/0")})


def test_live_route_returns_liveness_report():
    app = create_app(bootstrap_minio_on_startup=False)

    with TestClient(app) as client:
        response = client.get("/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "deps": {}}


def test_ready_route_returns_report():
    app = create_app(bootstrap_minio_on_startup=False)
    app.dependency_overrides[get_health_service] = lambda: StubHealthService()

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"]


def test_health_route_returns_report_for_backwards_compatibility():
    app = create_app(bootstrap_minio_on_startup=False)
    app.dependency_overrides[get_health_service] = lambda: StubHealthService()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_route_returns_text_payload():
    app = create_app(bootstrap_minio_on_startup=False)

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "app_requests_total" in response.text
