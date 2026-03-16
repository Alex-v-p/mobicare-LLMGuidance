from __future__ import annotations

import httpx

from inference.callbacks.notifier import CallbackNotifier
from shared.contracts.inference import InferenceRequest, JobRecord


class FakeClient:
    def __init__(self, responses=None, errors=None):
        self.responses = responses or []
        self.errors = errors or []
        self.posts = []
        self.index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers):
        self.posts.append((url, json, headers))
        if self.index < len(self.errors) and self.errors[self.index] is not None:
            error = self.errors[self.index]
            self.index += 1
            raise error
        response = self.responses[self.index]
        self.index += 1
        return response


class FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=httpx.Request("POST", "http://cb"), response=httpx.Response(self.status_code))


async def test_notifier_returns_immediately_when_no_callback_url():
    notifier = CallbackNotifier()
    status, error, attempts = await notifier.notify(None, None, JobRecord(request_id="req-1", request=InferenceRequest(request_id="req-1", question="Q"), status="queued"))
    assert (status, error, attempts) == (None, None, 0)


async def test_notifier_retries_and_returns_last_error(monkeypatch):
    record = JobRecord(request_id="req-1", request=InferenceRequest(request_id="req-1", question="Q"), status="completed")
    clients = [
        FakeClient(errors=[httpx.ConnectError("boom")]),
        FakeClient(responses=[FakeResponse(500)]),
        FakeClient(errors=[httpx.ReadTimeout("slow")]),
    ]

    def fake_create_async_client(timeout_s):
        return clients.pop(0)

    sleeps = []
    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("inference.callbacks.notifier.create_async_client", fake_create_async_client)
    monkeypatch.setattr("inference.callbacks.notifier.asyncio.sleep", fake_sleep)

    notifier = CallbackNotifier(max_attempts=3, backoff_seconds=(0.1, 0.2, 0.3))
    status, error, attempts = await notifier.notify("http://callback", {"X-Test": "1"}, record)

    assert status is None
    assert "ReadTimeout" in error
    assert attempts == 3
    assert sleeps == [0.1, 0.2]


async def test_notifier_returns_status_on_success(monkeypatch):
    client = FakeClient(responses=[FakeResponse(202)])
    monkeypatch.setattr("inference.callbacks.notifier.create_async_client", lambda timeout_s: client)
    notifier = CallbackNotifier(max_attempts=1)
    record = JobRecord(request_id="req-1", request=InferenceRequest(request_id="req-1", question="Q"), status="completed")

    status, error, attempts = await notifier.notify("http://callback", {"Authorization": "Bearer x"}, record)

    assert status == "202"
    assert error is None
    assert attempts == 1
    assert client.posts[0][0] == "http://callback"
    assert client.posts[0][2]["Authorization"] == "Bearer x"
