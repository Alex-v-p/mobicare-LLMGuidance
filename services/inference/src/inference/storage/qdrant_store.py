from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from inference.indexing.models import TextChunk


class QdrantVectorStore:
    def __init__(self) -> None:
        self._url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self._collection = os.getenv("QDRANT_COLLECTION", "guidance_chunks")
        self._client = QdrantClient(url=self._url)

    @property
    def collection_name(self) -> str:
        return self._collection

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        if recreate:
            try:
                self._client.delete_collection(self._collection)
            except Exception:
                pass
        try:
            self._client.get_collection(self._collection)
            return
        except Exception:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def collection_has_points(self) -> bool:
        try:
            count_result = self._client.count(collection_name=self._collection, exact=False)
            return int(count_result.count) > 0
        except Exception:
            return False

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
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(self, query_vector: list[float], limit: int = 3):
        result = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit,
        )
        if hasattr(result, "points"):
            return result.points
        return result
