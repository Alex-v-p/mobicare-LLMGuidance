from __future__ import annotations

import pytest

from api.application.services.health_service import HealthService
from shared.config import ApiSettings
from shared.contracts.health import DependencyStatus


@pytest.mark.asyncio
async def test_http_dependency_urls_use_trimmed_settings():
    settings = ApiSettings(inference_url="http://inference:8001/", qdrant_url="http://qdrant:6333/", minio_endpoint="http://minio:9000/", ollama_url="http://ollama:11434/")
    service = HealthService(settings=settings)

    urls = service._http_dependency_urls()

    assert urls == {
        "inference": "http://inference:8001/health",
        "qdrant": "http://qdrant:6333/readyz",
        "minio": "http://minio:9000/minio/health/ready",
        "ollama": "http://ollama:11434/api/tags",
    }


@pytest.mark.asyncio
async def test_report_is_degraded_when_any_dependency_fails(monkeypatch: pytest.MonkeyPatch):
    settings = ApiSettings(redis_url="redis://redis:6379/0")
    service = HealthService(settings=settings)

    async def fake_check_all(urls, timeout_s):
        return {
            "inference": DependencyStatus(ok=True, url=urls["inference"]),
            "qdrant": DependencyStatus(ok=False, url=urls["qdrant"], error="DEPENDENCY_UNAVAILABLE"),
            "minio": DependencyStatus(ok=True, url=urls["minio"]),
            "ollama": DependencyStatus(ok=True, url=urls["ollama"]),
        }

    async def fake_check_redis():
        return DependencyStatus(ok=True, url=settings.redis_url)

    monkeypatch.setattr("api.application.services.health_service.check_all", fake_check_all)
    monkeypatch.setattr(service, "_check_redis", fake_check_redis)

    report = await service.report()

    assert report.status == "degraded"
    assert report.deps["qdrant"].ok is False
    assert report.deps["redis"].ok is True
