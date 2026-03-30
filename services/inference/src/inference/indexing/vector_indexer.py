from __future__ import annotations

import asyncio
from functools import partial

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.indexing.models import TextChunk
from inference.storage.qdrant_store import QdrantVectorStore
from shared.config import InferenceSettings, get_inference_settings


class VectorIndexingService:
    def __init__(
        self,
        *,
        embedding_client: OllamaEmbeddingsClient,
        vector_store: QdrantVectorStore,
        settings: InferenceSettings | None = None,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._settings = settings or get_inference_settings()

    async def index_chunks(self, chunks: list[TextChunk], embedding_model: str | None = None) -> int:
        safe_chunks = [chunk for chunk in chunks if chunk.text and chunk.text.strip()]
        if not safe_chunks:
            return 0

        client = self._embedding_client.with_model(embedding_model)
        resolved_embedding_model = embedding_model or client.model
        batch_size = max(1, self._settings.ollama_embedding_batch_size)

        indexed = 0
        collection_ready = False

        for i in range(0, len(safe_chunks), batch_size):
            batch = safe_chunks[i:i + batch_size]
            batch_embeddings = await client.embed_many([chunk.text for chunk in batch])
            if not batch_embeddings:
                continue

            if not collection_ready:
                await asyncio.to_thread(self._vector_store.ensure_collection, len(batch_embeddings[0]))
                collection_ready = True

            indexed += await asyncio.to_thread(
                partial(
                    self._vector_store.upsert_chunks,
                    batch,
                    batch_embeddings,
                    embedding_model=resolved_embedding_model,
                )
            )

        return indexed
