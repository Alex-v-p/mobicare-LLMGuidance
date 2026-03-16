from __future__ import annotations

import json

import httpx
import pytest

from api.clients.inference_client import (
    InferenceClient,
    InferenceClientError,
    _classify_transport_error,
    _extract_error_payload,
)
from shared.contracts.inference import InferenceRequest
from tests.support.fakes import DummyAsyncClientContext


class TimeoutClient:
    async def request(self, *args, **kwargs):
        raise httpx.TimeoutException("boom")


@pytest.mark.asyncio
async def test_generate_guidance_success(monkeypatch: pytest.MonkeyPatch):
    payload = {
        "request_id": "req-1",
        "status": "ok",
        "model": "qwen2.5:0.5b",
        "answer": "Use standard therapy.",
        "retrieved_context": [],
        "used_variables": {},
        "warnings": [],
        "metadata": {},
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/guidance/generate"
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("api.clients.inference_client.create_async_client", lambda timeout_s: DummyAsyncClientContext(client))

    result = await InferenceClient(base_url="http://inference.test").generate_guidance(InferenceRequest(request_id="req-1", question="What now?"))

    assert result.answer == "Use standard therapy."
    assert result.model == "qwen2.5:0.5b"


@pytest.mark.asyncio
async def test_request_error_includes_upstream_request_id(monkeypatch: pytest.MonkeyPatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": "UPSTREAM_DOWN", "message": "Nope", "details": {"x": 1}}}, headers={"X-Request-ID": "upstream-123"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("api.clients.inference_client.create_async_client", lambda timeout_s: DummyAsyncClientContext(client))

    with pytest.raises(InferenceClientError) as exc:
        await InferenceClient(base_url="http://inference.test").get_guidance_job_status("job-1")

    assert exc.value.code == "UPSTREAM_DOWN"
    assert exc.value.details["upstream_request_id"] == "upstream-123"


@pytest.mark.asyncio
async def test_request_transport_timeout_maps_to_service_unavailable():
    client = InferenceClient(base_url="http://inference.test")

    with pytest.raises(InferenceClientError) as exc:
        await client._request(TimeoutClient(), method="GET", path="/guidance/jobs/job-1")

    assert exc.value.status_code == 503
    assert exc.value.details["reason"] == "TimeoutException"


def test_extract_error_payload_supports_http_exception_detail():
    response = httpx.Response(400, json={"detail": "bad request body"})

    code, message, details = _extract_error_payload(response)

    assert code == "INFERENCE_REQUEST_FAILED"
    assert message == "bad request body"
    assert details == {}


def test_extract_error_payload_handles_non_json_body():
    response = httpx.Response(500, text="plain failure")

    code, message, details = _extract_error_payload(response)

    assert code == "INFERENCE_BAD_RESPONSE"
    assert message == "plain failure"
    assert details == {"reason": "non_json_error_response"}


def test_classify_transport_error_for_connect_error():
    request = httpx.Request("GET", "http://inference.test")
    code, message = _classify_transport_error(httpx.ConnectError("failed", request=request))

    assert code == "INFERENCE_SERVICE_UNAVAILABLE"
    assert "could not be reached" in message.lower()
