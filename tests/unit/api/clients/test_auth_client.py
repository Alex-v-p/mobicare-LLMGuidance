from __future__ import annotations

import httpx
import pytest

from api.clients.auth_client import AuthClient, AuthClientError, _classify_transport_error, _extract_boolean
from tests.support.fakes import DummyAsyncClientContext


def test_extract_boolean_supports_multiple_payload_shapes():
    assert _extract_boolean(True) is True
    assert _extract_boolean({"valid": False}) is False
    assert _extract_boolean({"success": True}) is True
    assert _extract_boolean(" true ") is True
    assert _extract_boolean("false") is False
    assert _extract_boolean({"valid": "yes"}) is None


@pytest.mark.asyncio
async def test_validate_credentials_returns_false_for_401(monkeypatch: pytest.MonkeyPatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("api.clients.auth_client.create_async_client", lambda timeout_s: DummyAsyncClientContext(client))

    result = await AuthClient(validation_url="http://auth/validate").validate_credentials(email="user@example.com", password="bad")

    assert result is False


@pytest.mark.asyncio
async def test_validate_credentials_returns_boolean_payload(monkeypatch: pytest.MonkeyPatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"authenticated": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("api.clients.auth_client.create_async_client", lambda timeout_s: DummyAsyncClientContext(client))

    result = await AuthClient(validation_url="http://auth/validate").validate_credentials(email="user@example.com", password="secret")

    assert result is True


@pytest.mark.asyncio
async def test_validate_credentials_maps_http_errors(monkeypatch: pytest.MonkeyPatch):
    class TimeoutClient:
        async def post(self, *args, **kwargs):
            raise httpx.TimeoutException("boom")

        async def aclose(self):
            return None

    monkeypatch.setattr("api.clients.auth_client.create_async_client", lambda timeout_s: DummyAsyncClientContext(TimeoutClient()))

    with pytest.raises(AuthClientError) as exc:
        await AuthClient(validation_url="http://auth/validate").validate_credentials(email="user@example.com", password="secret")

    assert exc.value.status_code == 503
    assert exc.value.details["reason"] == "TimeoutException"


def test_classify_transport_error_maps_connect_error():
    request = httpx.Request("POST", "http://auth/validate")
    code, message = _classify_transport_error(httpx.ConnectError("no route", request=request))
    assert code == "AUTH_VALIDATION_UNAVAILABLE"
    assert "could not be reached" in message.lower()
