from __future__ import annotations

from time import perf_counter

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from shared.clients.health import check_all
from shared.config import get_inference_settings
from shared.contracts.error_codes import ErrorCode
from shared.contracts.health import DependencyStatus, HealthReport
from shared.observability import get_metrics_registry

router = APIRouter(tags=["health"])
metrics = get_metrics_registry()


async def _check_redis() -> DependencyStatus:
    settings = get_inference_settings()
    client = redis.from_url(settings.redis_url, decode_responses=True)
    start = perf_counter()
    try:
        await client.ping()
        latency_ms = round((perf_counter() - start) * 1000, 2)
        metrics.set_gauge("dependency_health_status", 1.0, labels={"dependency": "redis", "url": settings.redis_url})
        metrics.observe("dependency_health_latency_ms", latency_ms, labels={"dependency": "redis", "url": settings.redis_url})
        return DependencyStatus(ok=True, url=settings.redis_url, status_code=200, latency_ms=latency_ms)
    except redis.AuthenticationError:
        latency_ms = round((perf_counter() - start) * 1000, 2)
        metrics.set_gauge("dependency_health_status", 0.0, labels={"dependency": "redis", "url": settings.redis_url})
        return DependencyStatus(ok=False, url=settings.redis_url, error=ErrorCode.DEPENDENCY_AUTH_FAILED, latency_ms=latency_ms)
    except redis.RedisError:
        latency_ms = round((perf_counter() - start) * 1000, 2)
        metrics.set_gauge("dependency_health_status", 0.0, labels={"dependency": "redis", "url": settings.redis_url})
        return DependencyStatus(ok=False, url=settings.redis_url, error=ErrorCode.DEPENDENCY_UNAVAILABLE, latency_ms=latency_ms)
    finally:
        await client.aclose()


async def _readiness_report() -> HealthReport:
    settings = get_inference_settings()
    deps = await check_all({
        "qdrant": f"{settings.qdrant_url.rstrip('/')}/readyz",
        "minio": f"{settings.minio_endpoint.rstrip('/')}/minio/health/ready",
        "ollama": f"{settings.ollama_url.rstrip('/')}/api/tags",
    }, timeout_s=settings.healthcheck_timeout_s)
    deps["redis"] = await _check_redis()
    overall_ok = all(dep.ok for dep in deps.values())
    return HealthReport(status="ok" if overall_ok else "degraded", deps=deps)


def _liveness_report() -> HealthReport:
    return HealthReport(status="ok", deps={})


@router.get("/live", response_model=HealthReport)
async def live() -> HealthReport:
    return _liveness_report()


@router.get("/ready", response_model=HealthReport)
async def ready() -> HealthReport:
    return await _readiness_report()


@router.get("/health", response_model=HealthReport)
async def health() -> HealthReport:
    return await _readiness_report()


@router.get("/metrics", response_class=PlainTextResponse)
def get_metrics() -> PlainTextResponse:
    return PlainTextResponse(get_metrics_registry().render_prometheus(), media_type="text/plain; version=0.0.4")
