from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Dict

import httpx

from shared.clients.http import create_async_client
from shared.contracts.error_codes import ErrorCode
from shared.contracts.health import DependencyStatus
from shared.observability.metrics import get_metrics_registry


def _classify_http_error(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return ErrorCode.DEPENDENCY_TIMEOUT
    return ErrorCode.DEPENDENCY_UNAVAILABLE


async def check_url(client: httpx.AsyncClient, url: str, *, dependency: str) -> DependencyStatus:
    metrics = get_metrics_registry()
    start = perf_counter()
    try:
        resp = await client.get(url)
        latency_ms = round((perf_counter() - start) * 1000, 2)
        ok = 200 <= resp.status_code < 300
        metrics.set_gauge(
            "dependency_health_status",
            1.0 if ok else 0.0,
            labels={"dependency": dependency, "url": url},
        )
        metrics.observe(
            "dependency_health_latency_ms",
            latency_ms,
            labels={"dependency": dependency, "url": url},
        )
        error = None if ok else ErrorCode.DEPENDENCY_BAD_RESPONSE
        return DependencyStatus(ok=ok, url=url, status_code=resp.status_code, error=error, latency_ms=latency_ms)
    except httpx.HTTPError as exc:
        latency_ms = round((perf_counter() - start) * 1000, 2)
        code = _classify_http_error(exc)
        metrics.set_gauge(
            "dependency_health_status",
            0.0,
            labels={"dependency": dependency, "url": url},
        )
        metrics.observe(
            "dependency_health_latency_ms",
            latency_ms,
            labels={"dependency": dependency, "url": url},
        )
        return DependencyStatus(ok=False, url=url, error=code, latency_ms=latency_ms)


async def check_all(urls: Dict[str, str], timeout_s: float) -> Dict[str, DependencyStatus]:
    async with create_async_client(timeout_s=timeout_s) as client:
        tasks = {
            name: asyncio.create_task(check_url(client, url, dependency=name))
            for name, url in urls.items()
        }
        return {name: await task for name, task in tasks.items()}
