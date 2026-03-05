from __future__ import annotations

import httpx


def create_async_client(timeout_s: float) -> httpx.AsyncClient:
    """Shared httpx client factory.

    Centralizing this makes it easy to apply consistent timeouts, headers,
    retry middleware (later), etc.
    """

    return httpx.AsyncClient(timeout=timeout_s)
