from __future__ import annotations

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.retrieval.common import payload_to_context, resolve_collection_embedding_model, search_qdrant
from inference.storage.qdrant_store import QdrantVectorStore
from shared.config import InferenceSettings, get_inference_settings
from shared.contracts.inference import RetrievedContext


class DenseRetriever:
    def __init__(
        self,
        embedding_client: OllamaEmbeddingsClient | None = None,
        vector_store: QdrantVectorStore | None = None,
        settings: InferenceSettings | None = None,
    ) -> None:
        self._settings = settings or get_inference_settings()
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
        resolved_embedding_model = self.resolve_embedding_model(embedding_model)
        query_vector = await self._embedding_client.with_model(resolved_embedding_model).embed(query)
        hits = search_qdrant(vector_store=self._vector_store, query_vector=query_vector, limit=use_limit)
        results: list[RetrievedContext] = []
        for hit in hits:
            payload = dict(getattr(hit, "payload", None) or {})
            if not payload:
                continue
            results.append(payload_to_context(payload))
        return results

    def get_default_embedding_model(self) -> str:
        return self._embedding_client.model

    def resolve_embedding_model(self, requested_embedding_model: str | None = None) -> str:
        return resolve_collection_embedding_model(
            vector_store=self._vector_store,
            requested_embedding_model=requested_embedding_model,
        )
