from __future__ import annotations

import asyncio

import httpx

from shared.clients.http import create_async_client
from shared.config import Settings, get_settings
from shared.observability import get_logger


logger = get_logger(__name__, service="inference")


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

    @property
    def model(self) -> str:
        return self._model

    def with_model(self, model: str | None) -> "OllamaEmbeddingsClient":
        if not model or model == self._model:
            return self
        return OllamaEmbeddingsClient(
            base_url=self._base_url,
            model=model,
            timeout_s=self._timeout_s,
            settings=self._settings,
        )

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
                logger.error(
                    "ollama_embed_failed",
                    extra={
                        "event": "ollama_embed_failed",
                        "dependency": "ollama",
                        "status_code": response.status_code,
                        "error_code": "OLLAMA_EMBED_FAILED",
                        "dependency_endpoint": f"{self._base_url}/api/embed",
                        "model": self._model,
                        "response_body": response.text[:500],
                    },
                )
                response.raise_for_status()

            payload = response.json()
            return payload["embeddings"][0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            async with create_async_client(timeout_s=self._timeout_s) as client:
                response = await client.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": texts},
                )
                if response.status_code == 404:
                    return await self._embed_many_concurrently(texts)
                if response.is_error:
                    logger.warning(
                        "ollama_embed_many_failed_falling_back_to_single",
                        extra={
                            "event": "ollama_embed_many_failed_falling_back_to_single",
                            "dependency": "ollama",
                            "status_code": response.status_code,
                            "error_code": "OLLAMA_EMBED_MANY_FAILED",
                            "dependency_endpoint": f"{self._base_url}/api/embed",
                            "model": self._model,
                            "response_body": response.text[:500],
                        },
                    )
                    return await self._embed_many_concurrently(texts)

                payload = response.json()
                embeddings = payload.get("embeddings")
                if isinstance(embeddings, list) and len(embeddings) == len(texts):
                    return embeddings
        except httpx.TimeoutException:
            logger.warning(
                "ollama_embed_many_timeout_falling_back_to_single",
                extra={
                    "event": "ollama_embed_many_timeout_falling_back_to_single",
                    "dependency": "ollama",
                    "error_code": "OLLAMA_EMBED_MANY_TIMEOUT",
                    "dependency_endpoint": f"{self._base_url}/api/embed",
                    "model": self._model,
                    "batch_size": len(texts),
                },
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "ollama_embed_many_http_error_falling_back_to_single",
                extra={
                    "event": "ollama_embed_many_http_error_falling_back_to_single",
                    "dependency": "ollama",
                    "error_code": "OLLAMA_EMBED_MANY_HTTP_ERROR",
                    "dependency_endpoint": f"{self._base_url}/api/embed",
                    "model": self._model,
                    "batch_size": len(texts),
                    "message": str(exc),
                },
            )

        return await self._embed_many_concurrently(texts)

    async def _embed_many_concurrently(self, texts: list[str]) -> list[list[float]]:
        concurrency = max(1, self._settings.ollama_embedding_fallback_concurrency)
        semaphore = asyncio.Semaphore(concurrency)

        async def _embed_one(text: str) -> list[float]:
            async with semaphore:
                return await self.embed(text)

        return list(await asyncio.gather(*(_embed_one(text) for text in texts)))
