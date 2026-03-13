from __future__ import annotations

import redis.asyncio as redis

from shared.clients.health import check_all
from shared.config import Settings, get_settings
from shared.contracts.health import DependencyStatus, HealthReport


class HealthService:
    def __init__(self, timeout_s: float | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._timeout_s = timeout_s if timeout_s is not None else self._settings.healthcheck_timeout_s

    def _http_dependency_urls(self) -> dict[str, str]:
        return {
            "inference": f"{self._settings.inference_url.rstrip('/')}/health",
            "qdrant": f"{self._settings.qdrant_url.rstrip('/')}/readyz",
            "minio": f"{self._settings.minio_endpoint.rstrip('/')}/minio/health/ready",
            "ollama": f"{self._settings.ollama_url.rstrip('/')}/api/tags",
        }

    async def _check_redis(self) -> DependencyStatus:
        client = redis.from_url(self._settings.redis_url, decode_responses=True)

        try:
            await client.ping()
            return DependencyStatus(
                ok=True,
                url=self._settings.redis_url,
                status_code=200,
                error=None,
            )
        except redis.RedisError as exc:
            return DependencyStatus(
                ok=False,
                url=self._settings.redis_url,
                status_code=None,
                error=type(exc).__name__,
            )
        finally:
            await client.aclose()

    async def report(self) -> HealthReport:
        deps = await check_all(self._http_dependency_urls(), timeout_s=self._timeout_s)
        deps["redis"] = await self._check_redis()

        overall_ok = all(dep.ok for dep in deps.values())
        return HealthReport(status="ok" if overall_ok else "degraded", deps=deps)
