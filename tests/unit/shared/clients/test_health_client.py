from __future__ import annotations

import httpx
import pytest

from shared.clients.health import _classify_http_error, check_all, check_url
from tests.support.fakes import DummyAsyncClientContext


@pytest.mark.asyncio
async def test_check_url_marks_2xx_as_healthy():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    status = await check_url(client, "http://svc/health", dependency="svc")

    assert status.ok is True
    assert status.status_code == 200
    assert status.error is None


@pytest.mark.asyncio
async def test_check_url_marks_non_2xx_as_bad_response():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"ok": False})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    status = await check_url(client, "http://svc/health", dependency="svc")

    assert status.ok is False
    assert status.error == "DEPENDENCY_BAD_RESPONSE"


@pytest.mark.asyncio
async def test_check_all_runs_all_urls(monkeypatch: pytest.MonkeyPatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("shared.clients.health.create_async_client", lambda timeout_s: DummyAsyncClientContext(client))

    statuses = await check_all({"a": "http://a/health", "b": "http://b/health"}, timeout_s=1.0)

    assert set(statuses) == {"a", "b"}
    assert all(status.ok for status in statuses.values())


def test_classify_http_error_maps_timeout_to_timeout_code():
    assert _classify_http_error(httpx.TimeoutException("boom")) == "DEPENDENCY_TIMEOUT"
    assert _classify_http_error(RuntimeError("boom")) == "DEPENDENCY_UNAVAILABLE"
