from __future__ import annotations

import asyncio
from typing import Dict

import httpx

from shared.clients.http import create_async_client
from shared.contracts.health import DependencyStatus


async def check_url(client: httpx.AsyncClient, url: str) -> DependencyStatus:
    try:
        resp = await client.get(url)
        ok = 200 <= resp.status_code < 300
        return DependencyStatus(ok=ok, url=url, status_code=resp.status_code)
    except Exception as e:  # keep broad: DNS, connect, timeouts, etc.
        return DependencyStatus(ok=False, url=url, error=type(e).__name__)


async def check_all(urls: Dict[str, str], timeout_s: float) -> Dict[str, DependencyStatus]:
    """Check multiple dependency URLs concurrently."""

    async with create_async_client(timeout_s=timeout_s) as client:
        tasks = {name: asyncio.create_task(check_url(client, url)) for name, url in urls.items()}
        return {name: await task for name, task in tasks.items()}
