from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

from shared.clients.http import create_async_client
from shared.contracts.inference import JobRecord


class CallbackNotifier:
    def __init__(self) -> None:
        self._timeout_s = float(os.getenv("CALLBACK_TIMEOUT_S", "10"))
        self._max_attempts = int(os.getenv("CALLBACK_MAX_ATTEMPTS", "3"))
        self._backoff_seconds = [1.0, 5.0, 15.0]

    def _payload(self, record: JobRecord) -> Dict[str, Any]:
        result = record.result
        return {
            "job_id": record.job_id,
            "request_id": record.request_id,
            "status": record.status,
            "result_object_key": record.result_object_key,
            "error": record.error,
            "completed_at": record.completed_at,
            "result": None
            if result is None
            else {
                "answer": result.answer,
                "model": result.model,
                "rag": [item.model_dump(mode="json") for item in result.retrieved_context],
                "used_variables": result.used_variables,
                "warnings": result.warnings,
                "metadata": result.metadata,
            },
        }

    async def notify(self, record: JobRecord) -> tuple[str | None, str | None, int]:
        callback_url = record.request.options.callback_url
        if callback_url is None:
            return None, None, record.callback_attempts

        headers = dict(record.request.options.callback_headers)
        headers.setdefault("Content-Type", "application/json")
        headers.setdefault("X-Job-ID", record.job_id)
        headers.setdefault("X-Request-ID", record.request_id)

        last_error: str | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                async with create_async_client(timeout_s=self._timeout_s) as client:
                    response = await client.post(str(callback_url), json=self._payload(record), headers=headers)
                    response.raise_for_status()
                    return str(response.status_code), None, attempt
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self._max_attempts:
                    await asyncio.sleep(self._backoff_seconds[min(attempt - 1, len(self._backoff_seconds) - 1)])
        return None, last_error, self._max_attempts
