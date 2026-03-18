from __future__ import annotations

from typing import Any

class QdrantScrollClient:
    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "guidance_chunks") -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=url)
        self._collection_name = collection_name

    def fetch_all_payloads(self, *, batch_size: int = 256) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection_name,
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
