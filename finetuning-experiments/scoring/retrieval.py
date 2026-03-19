from __future__ import annotations

import math
from typing import Any

_SOURCE_WEIGHTS = {
    "direct_evidence": 1.0,
    "partial_direct_evidence": 0.8,
    "supporting": 0.5,
    "tangential": 0.2,
    "irrelevant": 0.0,
}
_STRICT_LABELS = {"direct_evidence", "partial_direct_evidence"}


def _build_chunk_label_map(source_mapping: dict[str, Any] | None) -> dict[str, str]:
    mapping = source_mapping or {}
    source_list = mapping.get("source_list") or {}
    label_map: dict[str, str] = {}

    if source_list:
        for label, items in source_list.items():
            for item in items or []:
                for chunk_id in item.get("chunk_ids") or []:
                    label_map[str(chunk_id)] = str(label)
        return label_map

    for item in mapping.get("matches") or []:
        for chunk_id in item.get("chunk_ids") or []:
            label_map[str(chunk_id)] = "direct_evidence"
    return label_map


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def _soft_ndcg(weights: list[float], ideal_weights: list[float]) -> float:
    dcg = sum(weight * _discount(rank) for rank, weight in enumerate(weights, start=1))
    ideal = sum(weight * _discount(rank) for rank, weight in enumerate(sorted(ideal_weights, reverse=True), start=1))
    return dcg / ideal if ideal > 0 else 0.0


def score_retrieval(source_mapping: dict[str, Any] | None, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    label_map = _build_chunk_label_map(source_mapping)
    ranked_chunk_ids = [str(chunk.get("chunk_id")) for chunk in retrieved_chunks if chunk.get("chunk_id")]
    ranked_labels = [label_map.get(chunk_id, "irrelevant") for chunk_id in ranked_chunk_ids]
    ranked_weights = [_SOURCE_WEIGHTS.get(label, 0.0) for label in ranked_labels]

    strict_chunk_ids = {chunk_id for chunk_id, label in label_map.items() if label in _STRICT_LABELS}
    first_rank = None
    for index, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in strict_chunk_ids:
            first_rank = index
            break

    def hit_at(k: int) -> float:
        return 1.0 if any(chunk_id in strict_chunk_ids for chunk_id in ranked_chunk_ids[:k]) else 0.0

    source_list = (source_mapping or {}).get("source_list") or {}
    strict_items = list(source_list.get("direct_evidence") or []) + list(source_list.get("partial_direct_evidence") or [])
    overlap_scores = [float(item.get("metadata", {}).get("passage_coverage", 0.0) or 0.0) for item in strict_items]
    semantic_scores = [float(item.get("semantic_score", 0.0) or 0.0) for item in strict_items]
    ideal_weights = [_SOURCE_WEIGHTS.get(label, 0.0) for label, items in source_list.items() for _ in (items or [])]

    return {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "strict_success": bool(first_rank),
        "average_overlap_score": _mean(overlap_scores),
        "average_semantic_score": _mean(semantic_scores),
        "weighted_relevance_score": _mean(ranked_weights),
        "soft_ndcg": _soft_ndcg(ranked_weights, ideal_weights),
        "retrieved_chunk_ids": ranked_chunk_ids,
        "retrieved_chunk_labels": ranked_labels,
        "strict_chunk_ids": sorted(strict_chunk_ids),
        "source_weight_scheme": dict(_SOURCE_WEIGHTS),
    }
