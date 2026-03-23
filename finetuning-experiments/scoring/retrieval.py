from __future__ import annotations

import math
import re
from typing import Any

_SOURCE_WEIGHTS = {
    "direct_evidence": 1.0,
    "partial_direct_evidence": 0.8,
    "supporting": 0.5,
    "tangential": 0.2,
    "irrelevant": 0.0,
}
_STRICT_LABELS = {"direct_evidence", "partial_direct_evidence"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


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


def _snippet_similarity(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_RE.findall((left or "").lower()))
    right_tokens = set(_TOKEN_RE.findall((right or "").lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _deduplicated_relevance(retrieved_chunks: list[dict[str, Any]], label_map: dict[str, str]) -> tuple[float, float, int]:
    kept_weights: list[float] = []
    duplicate_count = 0
    previous_snippets: list[str] = []
    for chunk in retrieved_chunks:
        snippet = str(chunk.get("snippet") or "")
        label = label_map.get(str(chunk.get("chunk_id")), "irrelevant")
        weight = _SOURCE_WEIGHTS.get(label, 0.0)
        is_duplicate = any(_snippet_similarity(snippet, prev) >= 0.75 for prev in previous_snippets)
        if is_duplicate:
            duplicate_count += 1
            weight *= 0.35
        kept_weights.append(weight)
        previous_snippets.append(snippet)
    raw = _mean(kept_weights)
    diversity = 1.0 - (duplicate_count / len(retrieved_chunks)) if retrieved_chunks else 1.0
    adjusted = min(1.0, (raw * 0.8) + (diversity * 0.2))
    return raw, adjusted, duplicate_count


def _lenient_success(ranked_labels: list[str]) -> float:
    top3 = ranked_labels[:3]
    top5 = ranked_labels[:5]
    if any(label == "direct_evidence" for label in top3):
        return 1.0
    if any(label in _STRICT_LABELS for label in top3) and any(label == "supporting" for label in top5):
        return 0.9
    if sum(1 for label in top5 if label in _STRICT_LABELS) >= 1:
        return 0.75
    if sum(1 for label in top5 if label in {"supporting", * _STRICT_LABELS}) >= 2:
        return 0.6
    if any(label == "supporting" for label in top3):
        return 0.4
    return 0.0


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
    retrieved_overlap_scores = [float(chunk.get("overlap_score", 0.0) or 0.0) for chunk in retrieved_chunks]
    retrieved_semantic_scores = [float(chunk.get("semantic_score", 0.0) or 0.0) for chunk in retrieved_chunks]
    ideal_weights = [_SOURCE_WEIGHTS.get(label, 0.0) for label, items in source_list.items() for _ in (items or [])]
    raw_relevance, adjusted_relevance, duplicate_count = _deduplicated_relevance(retrieved_chunks, label_map)
    lenient_success = _lenient_success(ranked_labels)

    return {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "strict_success": bool(first_rank),
        "lenient_success_score": lenient_success,
        "strict_hit_rank": first_rank,
        "average_overlap_score": _mean(overlap_scores),
        "average_semantic_score": _mean(semantic_scores),
        "retrieved_average_overlap_score": _mean(retrieved_overlap_scores),
        "retrieved_average_semantic_score": _mean(retrieved_semantic_scores),
        "weighted_relevance_score": adjusted_relevance,
        "weighted_relevance_raw": raw_relevance,
        "duplicate_chunk_count": duplicate_count,
        "duplicate_chunk_rate": (duplicate_count / len(ranked_chunk_ids)) if ranked_chunk_ids else 0.0,
        "context_diversity_score": 1.0 - ((duplicate_count / len(ranked_chunk_ids)) if ranked_chunk_ids else 0.0),
        "soft_ndcg": _soft_ndcg(ranked_weights, ideal_weights),
        "retrieved_chunk_ids": ranked_chunk_ids,
        "retrieved_chunk_count": len(ranked_chunk_ids),
        "retrieved_chunk_labels": ranked_labels,
        "strict_chunk_ids": sorted(strict_chunk_ids),
        "source_weight_scheme": dict(_SOURCE_WEIGHTS),
    }
