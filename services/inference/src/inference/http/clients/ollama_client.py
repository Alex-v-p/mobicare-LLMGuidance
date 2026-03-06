from __future__ import annotations

import os

from shared.clients.http import create_async_client
from shared.contracts.inference import OllamaGenerateResponse


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout_s: float | None = None) -> None:
        self._base_url = (base_url or os.getenv("OLLAMA_URL", "http://ollama:11434")).rstrip("/")
        self._model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
        self._timeout_s = timeout_s if timeout_s is not None else float(os.getenv("OLLAMA_TIMEOUT_S", "120"))

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> OllamaGenerateResponse:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.post(f"{self._base_url}/api/generate", json=payload)
            response.raise_for_status()
            return OllamaGenerateResponse.model_validate(response.json())
