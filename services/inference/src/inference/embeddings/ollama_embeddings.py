from __future__ import annotations

from shared.clients.http import create_async_client
from shared.config import Settings, get_settings


class OllamaEmbeddingsClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._base_url = (base_url or self._settings.ollama_url).rstrip("/")
        self._model = model or self._settings.ollama_embedding_model
        self._timeout_s = timeout_s if timeout_s is not None else self._settings.ollama_timeout_s

    def with_model(self, model: str | None) -> "OllamaEmbeddingsClient":
        if not model or model == self._model:
            return self
        return OllamaEmbeddingsClient(
            base_url=self._base_url,
            model=model,
            timeout_s=self._timeout_s,
            settings=self._settings,
        )

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": text},
            )
            if response.status_code == 404:
                legacy = await client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                legacy.raise_for_status()
                payload = legacy.json()
                return payload["embedding"]
            if response.is_error:
                print("OLLAMA EMBED ERROR STATUS:", response.status_code)
                print("OLLAMA EMBED ERROR BODY:", response.text)
                response.raise_for_status()

            payload = response.json()
            return payload["embeddings"][0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]
