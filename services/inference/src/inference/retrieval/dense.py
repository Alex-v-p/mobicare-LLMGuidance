from __future__ import annotations

import os

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.indexing.ingestion_service import IngestionService
from inference.storage.minio_documents import MinioDocumentStore
from inference.storage.qdrant_store import QdrantVectorStore
from shared.contracts.inference import RetrievedContext


class DenseRetriever:
    def __init__(
        self,
        embedding_client: OllamaEmbeddingsClient | None = None,
        vector_store: QdrantVectorStore | None = None,
        ingestion_service: IngestionService | None = None,
    ) -> None:
        self._embedding_client = embedding_client or OllamaEmbeddingsClient()
        self._vector_store = vector_store or QdrantVectorStore()
        self._ingestion_service = ingestion_service or IngestionService(
            document_store=MinioDocumentStore(),
            embedding_client=self._embedding_client,
            vector_store=self._vector_store,
        )
        self._default_top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))

    async def retrieve(self, query: str, limit: int | None = None) -> list[RetrievedContext]:
        use_limit = limit or self._default_top_k
        if not self._vector_store.collection_has_points():
            await self._ingestion_service.ingest()
            if not self._vector_store.collection_has_points():
                return []

        query_vector = await self._embedding_client.embed(query)
        hits = self._vector_store.search(query_vector=query_vector, limit=use_limit)
        results: list[RetrievedContext] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                RetrievedContext(
                    source_id=str(payload.get("source_id") or payload.get("chunk_id") or hit.id),
                    title=str(payload.get("title") or payload.get("object_name") or "Untitled"),
                    snippet=str(payload.get("text") or ""),
                )
            )
        return results
