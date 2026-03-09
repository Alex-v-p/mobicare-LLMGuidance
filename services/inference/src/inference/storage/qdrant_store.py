from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from inference.indexing.models import TextChunk


class QdrantVectorStore:
    def __init__(self, url: str | None = None, collection_name: str | None = None) -> None:
        self._url = url or os.getenv("QDRANT_URL", "http://qdrant:6333")
        self._collection = collection_name or os.getenv("QDRANT_COLLECTION", "guidance_chunks")
        self._client = QdrantClient(url=self._url)

    @property
    def collection_name(self) -> str:
        return self._collection

    def recreate_collection(self, vector_size: int) -> None:
        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def ensure_collection(self, vector_size: int) -> None:
        collections = {c.name for c in self._client.get_collections().collections}
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def collection_has_points(self) -> bool:
        try:
            self.ensure_collection(vector_size=self._infer_vector_size())
            count_result = self._client.count(collection_name=self._collection, exact=False)
            return int(count_result.count) > 0
        except Exception:
            return False

    def _infer_vector_size(self) -> int:
        configured = os.getenv("OLLAMA_EMBEDDING_DIMENSIONS")
        if configured:
            return int(configured)
        # nomic-embed-text default
        return 768

    def upsert_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> int:
        points: list[PointStruct] = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = abs(hash(chunk.chunk_id)) % (10**18) + index
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "title": chunk.title,
                "text": chunk.text,
                **chunk.metadata,
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        if not points:
            return 0
        self.ensure_collection(vector_size=len(embeddings[0]))
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(self, query_vector: list[float], limit: int = 3):
        self.ensure_collection(vector_size=len(query_vector))
        result = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit,
        )
        if hasattr(result, "points"):
            return result.points
        return result
