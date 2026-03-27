from __future__ import annotations

from typing import Literal

import redis.asyncio as redis
from pydantic import BaseModel

from inference.infrastructure.http.exceptions import ConflictError
from inference.storage.qdrant_store import MissingCollectionEmbeddingModelError, MissingCollectionError, QdrantVectorStore
from shared.config import InferenceSettings, get_inference_settings
from shared.observability import get_logger


logger = get_logger(__name__, service="inference")

RetrievalStateValue = Literal["empty", "ingesting", "ready", "failed"]


class RetrievalStateSnapshot(BaseModel):
    state: RetrievalStateValue
    message: str | None = None


class RetrievalStateController:
    def __init__(
        self,
        *,
        vector_store: QdrantVectorStore,
        redis_url: str | None = None,
        settings: InferenceSettings | None = None,
        state_key: str = "retrieval:state",
        message_key: str = "retrieval:state:message",
    ) -> None:
        self._settings = settings or get_inference_settings()
        self._vector_store = vector_store
        self._redis_url = redis_url or self._settings.redis_url
        self._state_key = state_key
        self._message_key = message_key
        self._redis: redis.Redis | None = None

    async def _client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def set_state(self, state: RetrievalStateValue, message: str | None = None) -> RetrievalStateSnapshot:
        client = await self._client()
        await client.set(self._state_key, state)
        if message:
            await client.set(self._message_key, message)
        else:
            await client.delete(self._message_key)
        return RetrievalStateSnapshot(state=state, message=message)

    async def get_state(self) -> RetrievalStateSnapshot | None:
        client = await self._client()
        state = await client.get(self._state_key)
        if state is None:
            return None
        message = await client.get(self._message_key)
        return RetrievalStateSnapshot(state=state, message=message)

    async def mark_ingesting(self, *, job_id: str) -> RetrievalStateSnapshot:
        return await self.set_state("ingesting", f"Ingestion job {job_id} is rebuilding the active retrieval collection.")

    async def mark_ready(self, *, collection: str, embedding_model: str | None = None) -> RetrievalStateSnapshot:
        model_suffix = f" with embedding model '{embedding_model}'" if embedding_model else ""
        return await self.set_state("ready", f"Collection '{collection}' is ready for guidance{model_suffix}.")

    async def mark_empty(self, message: str | None = None) -> RetrievalStateSnapshot:
        return await self.set_state(
            "empty",
            message or "No guidance collection is ready yet. Run document ingestion first.",
        )

    async def mark_failed(self, message: str) -> RetrievalStateSnapshot:
        return await self.set_state("failed", message)

    async def refresh_from_vector_store(self) -> RetrievalStateSnapshot:
        try:
            if not self._vector_store.collection_exists() or not self._vector_store.collection_has_points():
                return await self.mark_empty()
            embedding_model = self._vector_store.get_collection_embedding_model()
            return await self.mark_ready(
                collection=self._vector_store.collection_name,
                embedding_model=embedding_model,
            )
        except MissingCollectionError:
            return await self.mark_empty()
        except MissingCollectionEmbeddingModelError as exc:
            return await self.mark_failed(str(exc))
        except Exception as exc:  # pragma: no cover
            logger.exception(
                "retrieval_state_refresh_failed",
                extra={
                    "event": "retrieval_state_refresh_failed",
                    "error_code": "RETRIEVAL_STATE_REFRESH_FAILED",
                    "collection": self._vector_store.collection_name,
                },
            )
            return await self.mark_failed(f"Unable to verify retrieval readiness: {type(exc).__name__}: {exc}")

    async def is_guidance_ready(self) -> bool:
        current = await self.get_state()
        if current is not None and current.state == "ingesting":
            return False
        current = await self.refresh_from_vector_store()
        return current.state == "ready"

    async def ensure_guidance_ready(self) -> RetrievalStateSnapshot:
        current = await self.get_state()
        if current is not None and current.state == "ingesting":
            raise ConflictError(
                current.message or "Guidance is temporarily unavailable while ingestion is running.",
                details={"retrieval_state": current.state},
            )

        current = await self.refresh_from_vector_store()
        if current.state != "ready":
            raise ConflictError(
                current.message or "Guidance is unavailable until retrieval is ready.",
                details={"retrieval_state": current.state},
            )
        return current
