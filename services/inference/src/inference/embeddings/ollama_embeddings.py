from __future__ import annotations

import os

import httpx

from shared.clients.http import create_async_client


class OllamaEmbeddingsClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout_s: float | None = None) -> None:
        self._base_url = (base_url or os.getenv("OLLAMA_URL", "http://ollama:11434")).rstrip("/")
        self._model = model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self._timeout_s = timeout_s if timeout_s is not None else float(os.getenv("OLLAMA_TIMEOUT_S", "120"))

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
            response.raise_for_status()
            payload = response.json()
            return payload["embeddings"][0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]
