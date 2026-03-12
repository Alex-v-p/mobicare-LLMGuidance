from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from shared.config import Settings, get_settings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from inference.indexing.models import TextChunk


class MissingCollectionError(RuntimeError):
    pass


def stable_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._url = url or self._settings.qdrant_url
        self._collection = collection_name or self._settings.qdrant_collection
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
            return

        info = self._client.get_collection(self._collection)
        existing_size = info.config.params.vectors.size

        if existing_size != vector_size:
            raise ValueError(
                f"Collection '{self._collection}' expects dimension {existing_size}, "
                f"but current embedding model produced dimension {vector_size}. "
                f"Delete/recreate the collection or use a different collection name."
            )

    def collection_has_points(self) -> bool:
        if not self.collection_exists():
            return False
        count_result = self._client.count(collection_name=self._collection, exact=False)
        return int(count_result.count) > 0

    def upsert_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> int:
        points: list[PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = stable_point_id(chunk.chunk_id)
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
