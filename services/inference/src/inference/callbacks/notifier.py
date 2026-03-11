from __future__ import annotations

import asyncio
from typing import Mapping

import httpx

from shared.clients.http import create_async_client
from shared.contracts.inference import JobRecord


class CallbackNotifier:
    def __init__(
        self,
        timeout_s: float = 10.0,
        max_attempts: int = 3,
        backoff_seconds: tuple[float, ...] = (1.0, 2.0, 5.0),
    ) -> None:
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds

    def _payload(self, record: JobRecord) -> dict:
        return record.model_dump(mode="json")

    async def notify(
        self,
        callback_url: str | None,
        callback_headers: Mapping[str, str] | None,
        record: JobRecord,
    ) -> tuple[str | None, str | None, int]:
        if not callback_url:
            return None, None, 0

        headers = dict(callback_headers or {})
        last_error: str | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                async with create_async_client(timeout_s=self._timeout_s) as client:
                    response = await client.post(
                        str(callback_url),
                        json=self._payload(record),
                        headers=headers,
                    )
                    response.raise_for_status()
                    return str(response.status_code), None, attempt
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self._max_attempts:
                    delay = self._backoff_seconds[
                        min(attempt - 1, len(self._backoff_seconds) - 1)
                    ]
                    await asyncio.sleep(delay)
        return None, last_error, self._max_attempts
