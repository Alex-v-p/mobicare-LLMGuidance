from __future__ import annotations

from shared.clients.http import create_async_client
from shared.config import InferenceSettings, get_inference_settings
from shared.contracts.inference import OllamaGenerateResponse


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        settings: InferenceSettings | None = None,
    ) -> None:
        self._settings = settings or get_inference_settings()
        self._base_url = (base_url or self._settings.ollama_url).rstrip("/")
        self._model = model or self._settings.ollama_model
        self._timeout_s = timeout_s if timeout_s is not None else self._settings.ollama_timeout_s

    def with_model(self, model: str | None) -> "OllamaClient":
        if not model or model == self._model:
            return self
        return OllamaClient(
            base_url=self._base_url,
            model=model,
            timeout_s=self._timeout_s,
            settings=self._settings,
        )

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> OllamaGenerateResponse:
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
            response = await client.post(
                f"{self._base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            return OllamaGenerateResponse.model_validate(response.json())
