from __future__ import annotations

import os

from shared.clients.http import create_async_client
from shared.contracts.inference import InferenceRequest, InferenceResponse


class InferenceClient:
    def __init__(self, base_url: str | None = None, timeout_s: float | None = None) -> None:
        self._base_url = (base_url or os.getenv("INFERENCE_URL", "http://inference:8001")).rstrip("/")
        self._timeout_s = timeout_s if timeout_s is not None else float(os.getenv("INFERENCE_TIMEOUT_S", "60"))

    async def generate(self, payload: InferenceRequest) -> InferenceResponse:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.post(f"{self._base_url}/generate", json=payload.model_dump())
            response.raise_for_status()
            return InferenceResponse.model_validate(response.json())
