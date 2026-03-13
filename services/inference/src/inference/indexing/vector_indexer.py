from __future__ import annotations

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.indexing.models import TextChunk
from inference.storage.qdrant_store import QdrantVectorStore


class VectorIndexingService:
    def __init__(
        self,
        *,
        embedding_client: OllamaEmbeddingsClient,
        vector_store: QdrantVectorStore,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    async def index_chunks(self, chunks: list[TextChunk], embedding_model: str | None = None) -> int:
        safe_chunks = [chunk for chunk in chunks if chunk.text and chunk.text.strip()]
        if not safe_chunks:
            return 0

        embeddings = await self._embedding_client.with_model(embedding_model).embed_many(
            [chunk.text for chunk in safe_chunks]
        )
        self._vector_store.ensure_collection(vector_size=len(embeddings[0]))
        return self._vector_store.upsert_chunks(safe_chunks, embeddings)
