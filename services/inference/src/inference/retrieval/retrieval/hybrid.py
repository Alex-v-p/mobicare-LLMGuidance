from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.retrieval.graph import ChunkGraphAugmenter
from inference.retrieval.sparse import SparseKeywordRetriever
from inference.storage.qdrant_store import MissingCollectionError, QdrantVectorStore
from shared.contracts.inference import RetrievedContext


class RetrievalCollectionNotReadyError(RuntimeError):
    pass


@dataclass(slots=True)
class HybridRetrievalResult:
    items: list[RetrievedContext]
    metadata: dict[str, Any]


class HybridRetriever:
    def __init__(
        self,
        embedding_client: OllamaEmbeddingsClient | None = None,
        vector_store: QdrantVectorStore | None = None,
        sparse_retriever: SparseKeywordRetriever | None = None,
        graph_augmenter: ChunkGraphAugmenter | None = None,
    ) -> None:
        self._embedding_client = embedding_client or OllamaEmbeddingsClient()
        self._vector_store = vector_store or QdrantVectorStore()
        self._sparse_retriever = sparse_retriever or SparseKeywordRetriever()
        self._graph_augmenter = graph_augmenter or ChunkGraphAugmenter()

    async def retrieve(
        self,
        *,
        query: str,
        limit: int = 3,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        use_graph_augmentation: bool = False,
        graph_max_extra_nodes: int = 2,
    ) -> HybridRetrievalResult:
        if not self._vector_store.collection_exists():
            raise RetrievalCollectionNotReadyError(
                f"Qdrant collection '{self._vector_store.collection_name}' does not exist yet. Run document ingestion first."
            )
        if not self._vector_store.collection_has_points():
            raise RetrievalCollectionNotReadyError(
                f"Qdrant collection '{self._vector_store.collection_name}' is empty. Run document ingestion first."
            )

        corpus_payloads = self._vector_store.get_all_payloads()
        if not corpus_payloads:
            raise RetrievalCollectionNotReadyError(
                f"Qdrant collection '{self._vector_store.collection_name}' is empty. Run document ingestion first."
            )

        query_vector = await self._embedding_client.embed(query)
        try:
            dense_hits = self._vector_store.search(query_vector=query_vector, limit=max(limit * 4, 10))
        except MissingCollectionError as exc:
            raise RetrievalCollectionNotReadyError(str(exc)) from exc

        sparse_hits = self._sparse_retriever.search(query=query, documents=corpus_payloads, limit=max(limit * 4, 10))

        dense_norm = self._normalize_dense_hits(dense_hits)
        sparse_norm = self._normalize_sparse_hits(sparse_hits)

        fused: dict[str, dict[str, Any]] = {}
        for chunk_id, item in dense_norm.items():
            fused[chunk_id] = {
                **item,
                "score": dense_weight * item["score"],
                "dense_score": item["score"],
                "sparse_score": 0.0,
            }

        for chunk_id, item in sparse_norm.items():
            existing = fused.get(chunk_id)
            sparse_component = sparse_weight * item["score"]
            if existing is None:
                fused[chunk_id] = {
                    **item,
                    "score": sparse_component,
                    "dense_score": 0.0,
                    "sparse_score": item["score"],
                }
            else:
                existing["score"] += sparse_component
                existing["sparse_score"] = item["score"]

        ranked_payloads = [
            item["payload"]
            for item in sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:limit]
        ]

        graph_metadata = {"graph_augmented": False, "graph_nodes_added": 0, "graph_edges_used": []}
        if use_graph_augmentation:
            items, graph_metadata = self._graph_augmenter.expand(
                query=query,
                ranked_payloads=ranked_payloads,
                corpus_payloads=corpus_payloads,
                max_extra_nodes=graph_max_extra_nodes,
            )
        else:
            items = [self._to_context(payload) for payload in ranked_payloads]

        return HybridRetrievalResult(
            items=items,
            metadata={
                "retrieval_mode": "hybrid",
                "dense_candidates": len(dense_hits),
                "sparse_candidates": len(sparse_hits),
                "hybrid_dense_weight": dense_weight,
                "hybrid_sparse_weight": sparse_weight,
                **graph_metadata,
            },
        )

    def _normalize_dense_hits(self, hits: list[Any]) -> dict[str, dict[str, Any]]:
        raw: list[tuple[dict[str, Any], float]] = []
        for hit in hits:
            payload = dict(getattr(hit, "payload", None) or {})
            if not payload:
                continue
            score = float(getattr(hit, "score", 0.0) or 0.0)
            raw.append((payload, score))
        return self._normalize_payload_scores(raw)

    def _normalize_sparse_hits(self, hits: list[Any]) -> dict[str, dict[str, Any]]:
        raw = [(dict(hit.payload), float(hit.score)) for hit in hits if getattr(hit, "payload", None)]
        return self._normalize_payload_scores(raw)

    def _normalize_payload_scores(self, items: list[tuple[dict[str, Any], float]]) -> dict[str, dict[str, Any]]:
        if not items:
            return {}
        max_score = max(score for _, score in items) or 1.0
        min_score = min(score for _, score in items)
        denom = max(max_score - min_score, 1e-9)
        normalized: dict[str, dict[str, Any]] = {}
        for payload, score in items:
            chunk_id = str(payload.get("chunk_id") or payload.get("source_id") or id(payload))
            normalized[chunk_id] = {
                "payload": payload,
                "score": (score - min_score) / denom if max_score != min_score else 1.0,
            }
        return normalized

    def _to_context(self, payload: dict[str, Any]) -> RetrievedContext:
        return RetrievedContext(
            source_id=str(payload.get("source_id") or payload.get("chunk_id") or "unknown"),
            title=str(payload.get("title") or payload.get("object_name") or "Untitled"),
            snippet=str(payload.get("text") or ""),
        )
