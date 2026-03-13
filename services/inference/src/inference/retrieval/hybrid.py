from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.retrieval.common import (
    ensure_collection_ready,
    payload_identity,
    payload_to_context,
    search_qdrant,
)
from inference.retrieval.graph import ChunkGraphAugmenter
from inference.retrieval.sparse import SparseKeywordRetriever
from inference.storage.qdrant_store import QdrantVectorStore


@dataclass(slots=True)
class HybridRetrievalResult:
    items: list
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
        self._cached_corpus_payloads: list[dict[str, Any]] | None = None
        self._cached_corpus_count: int | None = None

    async def retrieve(
        self,
        *,
        query: str,
        limit: int = 3,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        use_graph_augmentation: bool = False,
        graph_max_extra_nodes: int = 2,
        embedding_model: str | None = None,
    ) -> HybridRetrievalResult:
        ensure_collection_ready(self._vector_store)
        corpus_payloads = self._get_corpus_payloads()
        if not corpus_payloads:
            raise RuntimeError(
                f"Qdrant collection '{self._vector_store.collection_name}' is empty. Run document ingestion first."
            )

        query_vector = await self._embedding_client.with_model(embedding_model).embed(query)
        dense_limit = max(limit * 4, 10)
        dense_hits = search_qdrant(vector_store=self._vector_store, query_vector=query_vector, limit=dense_limit)
        sparse_hits = self._sparse_retriever.search(query=query, documents=corpus_payloads, limit=dense_limit)

        fused = self._fuse_results(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        ranked_payloads = [item["payload"] for item in sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:limit]]

        graph_metadata = {"graph_augmented": False, "graph_nodes_added": 0, "graph_edges_used": []}
        if use_graph_augmentation:
            items, graph_metadata = self._graph_augmenter.expand(
                query=query,
                ranked_payloads=ranked_payloads,
                corpus_payloads=corpus_payloads,
                max_extra_nodes=graph_max_extra_nodes,
            )
        else:
            items = [payload_to_context(payload) for payload in ranked_payloads]

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

    def _get_corpus_payloads(self) -> list[dict[str, Any]]:
        count = self._vector_store.count_points()
        if self._cached_corpus_payloads is not None and self._cached_corpus_count == count:
            return self._cached_corpus_payloads
        self._cached_corpus_payloads = self._vector_store.get_all_payloads()
        self._cached_corpus_count = count
        return self._cached_corpus_payloads

    def _fuse_results(
        self,
        *,
        dense_hits: list[Any],
        sparse_hits: list[Any],
        dense_weight: float,
        sparse_weight: float,
    ) -> dict[str, dict[str, Any]]:
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
        return fused

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
            chunk_id = payload_identity(payload)
            normalized[chunk_id] = {
                "payload": payload,
                "score": (score - min_score) / denom if max_score != min_score else 1.0,
            }
        return normalized
