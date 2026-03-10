from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from inference.indexing.models import TextChunk


class MissingCollectionError(RuntimeError):
    pass


class QdrantVectorStore:
    def __init__(self, url: str | None = None, collection_name: str | None = None) -> None:
        self._url = url or os.getenv("QDRANT_URL", "http://qdrant:6333")
        self._collection = collection_name or os.getenv("QDRANT_COLLECTION", "guidance_chunks")
        self._client = QdrantClient(url=self._url)

    @property
    def collection_name(self) -> str:
        return self._collection

    def collection_exists(self) -> bool:
        collections = {c.name for c in self._client.get_collections().collections}
        return self._collection in collections

    def ensure_collection(self, vector_size: int) -> None:
        if not self.collection_exists():
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def collection_has_points(self) -> bool:
        if not self.collection_exists():
            return False
        count_result = self._client.count(collection_name=self._collection, exact=False)
        return int(count_result.count) > 0

    def upsert_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> int:
        points: list[PointStruct] = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = abs(hash(chunk.chunk_id)) % (10**18) + index
            chunk_metadata = {
                key: value
                for key, value in chunk.metadata.items()
                if key not in {"normalized_source_text", "page_ranges", "raw_page_texts"}
            }
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "title": chunk.title,
                "text": chunk.text,
                "page_number": chunk.metadata.get("page_number"),
                **chunk_metadata,
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        if not points:
            return 0
        self.ensure_collection(vector_size=len(embeddings[0]))
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)


    def get_all_payloads(self, batch_size: int = 256) -> list[dict[str, Any]]:
        if not self.collection_exists():
            raise MissingCollectionError(
                f"Qdrant collection '{self._collection}' does not exist yet. Run document ingestion first."
            )

        payloads: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for point in points:
                payload = dict(point.payload or {})
                if payload:
                    payloads.append(payload)
            if offset is None:
                break
        return payloads

    def search(self, query_vector: list[float], limit: int = 3):
        if not self.collection_exists():
            raise MissingCollectionError(
                f"Qdrant collection '{self._collection}' does not exist yet. Run document ingestion first."
            )
        result = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit,
        )
        if hasattr(result, "points"):
            return result.points
        return result
