from __future__ import annotations

from time import perf_counter

import redis.asyncio as redis

from shared.clients.health import check_all
from shared.config import Settings, get_settings
from shared.contracts.error_codes import ErrorCode
from shared.contracts.health import DependencyStatus, HealthReport
from shared.observability.metrics import get_metrics_registry


class HealthService:
    def __init__(self, timeout_s: float | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._timeout_s = timeout_s if timeout_s is not None else self._settings.healthcheck_timeout_s
        self._metrics = get_metrics_registry()

    def _http_dependency_urls(self) -> dict[str, str]:
        return {
            "inference": f"{self._settings.inference_url.rstrip('/')}/health",
            "qdrant": f"{self._settings.qdrant_url.rstrip('/')}/readyz",
            "minio": f"{self._settings.minio_endpoint.rstrip('/')}/minio/health/ready",
            "ollama": f"{self._settings.ollama_url.rstrip('/')}/api/tags",
        }

    async def _check_redis(self) -> DependencyStatus:
        client = redis.from_url(self._settings.redis_url, decode_responses=True)
        start = perf_counter()
        try:
            await client.ping()
            latency_ms = round((perf_counter() - start) * 1000, 2)
            self._metrics.set_gauge("dependency_health_status", 1.0, labels={"dependency": "redis", "url": self._settings.redis_url})
            self._metrics.observe("dependency_health_latency_ms", latency_ms, labels={"dependency": "redis", "url": self._settings.redis_url})
            return DependencyStatus(ok=True, url=self._settings.redis_url, status_code=200, error=None, latency_ms=latency_ms)
        except redis.AuthenticationError:
            latency_ms = round((perf_counter() - start) * 1000, 2)
            self._metrics.set_gauge("dependency_health_status", 0.0, labels={"dependency": "redis", "url": self._settings.redis_url})
            return DependencyStatus(ok=False, url=self._settings.redis_url, status_code=None, error=ErrorCode.DEPENDENCY_AUTH_FAILED, latency_ms=latency_ms)
        except redis.RedisError:
            latency_ms = round((perf_counter() - start) * 1000, 2)
            self._metrics.set_gauge("dependency_health_status", 0.0, labels={"dependency": "redis", "url": self._settings.redis_url})
            return DependencyStatus(ok=False, url=self._settings.redis_url, status_code=None, error=ErrorCode.DEPENDENCY_UNAVAILABLE, latency_ms=latency_ms)
        finally:
            await client.aclose()

    async def report(self) -> HealthReport:
        deps = await check_all(self._http_dependency_urls(), timeout_s=self._timeout_s)
        deps["redis"] = await self._check_redis()
        overall_ok = all(dep.ok for dep in deps.values())
        return HealthReport(status="ok" if overall_ok else "degraded", deps=deps)
