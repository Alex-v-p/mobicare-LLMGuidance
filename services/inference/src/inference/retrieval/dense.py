from __future__ import annotations

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.retrieval.common import payload_to_context, search_qdrant
from inference.storage.qdrant_store import QdrantVectorStore
from shared.config import Settings, get_settings
from shared.contracts.inference import RetrievedContext


class DenseRetriever:
    def __init__(
        self,
        embedding_client: OllamaEmbeddingsClient | None = None,
        vector_store: QdrantVectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding_client = embedding_client or OllamaEmbeddingsClient(settings=self._settings)
        self._vector_store = vector_store or QdrantVectorStore(settings=self._settings)
        self._default_top_k = self._settings.retrieval_top_k

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedContext]:
        use_limit = limit or self._default_top_k
        query_vector = await self._embedding_client.with_model(embedding_model).embed(query)
        hits = search_qdrant(vector_store=self._vector_store, query_vector=query_vector, limit=use_limit)
        results: list[RetrievedContext] = []
        for hit in hits:
            payload = dict(getattr(hit, "payload", None) or {})
            if not payload:
                continue
            results.append(payload_to_context(payload))
        return results
