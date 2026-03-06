from __future__ import annotations

import os

import redis.asyncio as redis

from shared.clients.health import check_all
from shared.contracts.health import DependencyStatus, HealthReport


class HealthService:
    def __init__(self, timeout_s: float | None = None) -> None:
        self._timeout_s = timeout_s if timeout_s is not None else float(os.getenv("HEALTHCHECK_TIMEOUT_S", "2.0"))

    def _env(self, name: str, default: str) -> str:
        return os.getenv(name, default)

    def _http_dependency_urls(self) -> dict[str, str]:
        inference_url = self._env("INFERENCE_URL", "http://inference:8001")
        qdrant_url = self._env("QDRANT_URL", "http://qdrant:6333")
        minio_endpoint = self._env("MINIO_ENDPOINT", "http://minio:9000")
        ollama_url = self._env("OLLAMA_URL", "http://ollama:11434")

        return {
            "inference": f"{inference_url.rstrip('/')}/health",
            "qdrant": f"{qdrant_url.rstrip('/')}/readyz",
            "minio": f"{minio_endpoint.rstrip('/')}/minio/health/ready",
            "ollama": f"{ollama_url.rstrip('/')}/api/tags",
        }

    async def _check_redis(self) -> DependencyStatus:
        redis_url = self._env("REDIS_URL", "redis://redis:6379/0")
        client = redis.from_url(redis_url, decode_responses=True)

        try:
            await client.ping()
            return DependencyStatus(
                ok=True,
                url=redis_url,
                status_code=200,
                error=None,
            )
        except Exception as e:
            return DependencyStatus(
                ok=False,
                url=redis_url,
                status_code=None,
                error=type(e).__name__,
            )
        finally:
            await client.aclose()

    async def report(self) -> HealthReport:
        deps = await check_all(self._http_dependency_urls(), timeout_s=self._timeout_s)
        deps["redis"] = await self._check_redis()

        overall_ok = all(dep.ok for dep in deps.values())
        return HealthReport(status="ok" if overall_ok else "degraded", deps=deps)