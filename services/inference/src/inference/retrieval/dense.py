from __future__ import annotations

import os

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.storage.qdrant_store import MissingCollectionError, QdrantVectorStore
from shared.contracts.inference import RetrievedContext


class RetrievalCollectionNotReadyError(RuntimeError):
    pass


class DenseRetriever:
    def __init__(
        self,
        embedding_client: OllamaEmbeddingsClient | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self._embedding_client = embedding_client or OllamaEmbeddingsClient()
        self._vector_store = vector_store or QdrantVectorStore()
        self._default_top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedContext]:
        use_limit = limit or self._default_top_k
        if not self._vector_store.collection_exists():
            raise RetrievalCollectionNotReadyError(
                f"Qdrant collection '{self._vector_store.collection_name}' does not exist yet. Run document ingestion first."
            )
        if not self._vector_store.collection_has_points():
            raise RetrievalCollectionNotReadyError(
                f"Qdrant collection '{self._vector_store.collection_name}' is empty. Run document ingestion first."
            )

        query_vector = await self._embedding_client.with_model(embedding_model).embed(query)
        try:
            hits = self._vector_store.search(query_vector=query_vector, limit=use_limit)
        except MissingCollectionError as exc:
            raise RetrievalCollectionNotReadyError(str(exc)) from exc
        results: list[RetrievedContext] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                RetrievedContext(
                    source_id=str(payload.get("source_id") or payload.get("chunk_id") or hit.id),
                    title=str(payload.get("title") or payload.get("object_name") or "Untitled"),
                    snippet=str(payload.get("text") or ""),
                    chunk_id=str(payload.get("chunk_id")) if payload.get("chunk_id") is not None else None,
                    page_number=int(payload.get("page_number")) if payload.get("page_number") is not None else None,
                )
            )
        return results
