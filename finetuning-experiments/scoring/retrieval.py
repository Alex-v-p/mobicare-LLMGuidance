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
_RELEVANT_LABELS = _STRICT_LABELS | {"supporting"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _chunk_id_aliases(chunk_id: Any) -> set[str]:
    raw = str(chunk_id or "").strip()
    if not raw:
        return set()
    aliases = {raw, raw.lower()}
    if "#" in raw:
        suffix = raw.split("#", 1)[1].strip()
        if suffix:
            aliases.add(suffix)
            aliases.add(suffix.lower())
    basename = raw.rsplit("/", 1)[-1]
    aliases.add(basename)
    aliases.add(basename.lower())
    return {alias for alias in aliases if alias}


def _build_chunk_label_map(source_mapping: dict[str, Any] | None) -> dict[str, str]:
    mapping = source_mapping or {}
    source_list = mapping.get("source_list") or {}
    label_map: dict[str, str] = {}

    def register(chunk_id: Any, label: str) -> None:
        for alias in _chunk_id_aliases(chunk_id):
            label_map[alias] = str(label)

    if source_list:
        for label, items in source_list.items():
            for item in items or []:
                for chunk_id in item.get("chunk_ids") or []:
                    register(chunk_id, str(label))
        return label_map

    for item in mapping.get("matches") or []:
        for chunk_id in item.get("chunk_ids") or []:
            register(chunk_id, "direct_evidence")
    return label_map


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_or_none(values: list[float]) -> float | None:
    return _mean(values) if values else None


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


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_chunk_metric(chunk: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _coerce_float(chunk.get(key))
        if value is not None:
            return value
    return None


def _metric_availability_rate(values: list[float], *, total: int) -> float:
    if total <= 0:
        return 0.0
    return len(values) / total


def _deduplicated_relevance(retrieved_chunks: list[dict[str, Any]], label_map: dict[str, str]) -> tuple[float, float, int]:
    kept_weights: list[float] = []
    duplicate_count = 0
    previous_snippets: list[str] = []
    for chunk in retrieved_chunks:
        snippet = str(chunk.get("snippet") or "")
        chunk_id = next(iter(_chunk_id_aliases(chunk.get("chunk_id"))), "")
        label = label_map.get(chunk_id, "irrelevant")
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
    if sum(1 for label in top5 if label in _RELEVANT_LABELS) >= 2:
        return 0.6
    if any(label == "supporting" for label in top3):
        return 0.4
    return 0.0


def score_retrieval(source_mapping: dict[str, Any] | None, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    mapping = source_mapping or {}
    metadata = dict(mapping.get("metadata") or {})
    applicable = bool(metadata.get("applicable", True))
    skipped_reason = metadata.get("skipped_reason")

    label_map = _build_chunk_label_map(mapping)
    ranked_chunk_ids = [str(chunk.get("chunk_id")) for chunk in retrieved_chunks if chunk.get("chunk_id")]
    ranked_labels = [
        next((label_map.get(alias) for alias in _chunk_id_aliases(chunk_id) if label_map.get(alias) is not None), "irrelevant")
        for chunk_id in ranked_chunk_ids
    ]
    ranked_weights = [_SOURCE_WEIGHTS.get(label, 0.0) for label in ranked_labels]

    strict_chunk_ids = {chunk_id for chunk_id, label in label_map.items() if label in _STRICT_LABELS}
    relevant_chunk_ids = {chunk_id for chunk_id, label in label_map.items() if label in _RELEVANT_LABELS}

    first_strict_rank = None
    first_relevant_rank = None
    for index, chunk_id in enumerate(ranked_chunk_ids, start=1):
        aliases = _chunk_id_aliases(chunk_id)
        if first_strict_rank is None and aliases & strict_chunk_ids:
            first_strict_rank = index
        if first_relevant_rank is None and aliases & relevant_chunk_ids:
            first_relevant_rank = index
        if first_strict_rank is not None and first_relevant_rank is not None:
            break

    def hit_at(k: int, targets: set[str]) -> float:
        for chunk_id in ranked_chunk_ids[:k]:
            if _chunk_id_aliases(chunk_id) & targets:
                return 1.0
        return 0.0

    source_list = mapping.get("source_list") or {}
    strict_items = list(source_list.get("direct_evidence") or []) + list(source_list.get("partial_direct_evidence") or [])
    overlap_scores = [float(item.get("metadata", {}).get("passage_coverage", 0.0) or 0.0) for item in strict_items]
    semantic_scores = [float(item.get("semantic_score", 0.0) or 0.0) for item in strict_items]

    retrieved_overlap_scores = [
        value for value in (_extract_chunk_metric(chunk, "overlap_score", "retrieval_overlap_score") for chunk in retrieved_chunks)
        if value is not None
    ]
    retrieved_semantic_scores = [
        value for value in (_extract_chunk_metric(chunk, "semantic_score", "retrieval_semantic_score") for chunk in retrieved_chunks)
        if value is not None
    ]
    retrieved_ranking_scores = [
        value for value in (_extract_chunk_metric(chunk, "retrieval_ranking_score", "ranking_score") for chunk in retrieved_chunks)
        if value is not None
    ]
    retrieved_query_term_overlaps = [
        value for value in (_extract_chunk_metric(chunk, "retrieval_query_term_overlap", "query_term_overlap") for chunk in retrieved_chunks)
        if value is not None
    ]
    retrieved_hf_overlaps = [
        value for value in (_extract_chunk_metric(chunk, "retrieval_heart_failure_overlap", "heart_failure_overlap") for chunk in retrieved_chunks)
        if value is not None
    ]
    retrieved_clinical_overlaps = [
        value for value in (_extract_chunk_metric(chunk, "retrieval_clinical_term_overlap", "clinical_term_overlap") for chunk in retrieved_chunks)
        if value is not None
    ]

    ideal_weights = [_SOURCE_WEIGHTS.get(label, 0.0) for label, items in source_list.items() for _ in (items or [])]
    raw_relevance, adjusted_relevance, duplicate_count = _deduplicated_relevance(retrieved_chunks, label_map)
    lenient_success = _lenient_success(ranked_labels)
    retrieved_count = len(retrieved_chunks)

    return {
        "applicable": applicable,
        "skipped_reason": skipped_reason,
        "hit_at_1": hit_at(1, relevant_chunk_ids) if applicable else None,
        "hit_at_3": hit_at(3, relevant_chunk_ids) if applicable else None,
        "hit_at_5": hit_at(5, relevant_chunk_ids) if applicable else None,
        "mrr": ((1.0 / first_relevant_rank) if first_relevant_rank else 0.0) if applicable else None,
        "strict_hit_at_1": hit_at(1, strict_chunk_ids) if applicable else None,
        "strict_hit_at_3": hit_at(3, strict_chunk_ids) if applicable else None,
        "strict_hit_at_5": hit_at(5, strict_chunk_ids) if applicable else None,
        "strict_mrr": ((1.0 / first_strict_rank) if first_strict_rank else 0.0) if applicable else None,
        "strict_success": bool(first_strict_rank) if applicable else None,
        "relevance_success": bool(first_relevant_rank) if applicable else None,
        "lenient_success_score": lenient_success if applicable else None,
        "strict_hit_rank": first_strict_rank if applicable else None,
        "relevance_hit_rank": first_relevant_rank if applicable else None,
        "average_overlap_score": _mean(overlap_scores) if applicable else None,
        "average_semantic_score": _mean(semantic_scores) if applicable else None,
        "retrieved_average_overlap_score": _mean_or_none(retrieved_overlap_scores) if applicable else None,
        "retrieved_average_semantic_score": _mean_or_none(retrieved_semantic_scores) if applicable else None,
        "retrieved_overlap_score_observed_count": len(retrieved_overlap_scores) if applicable else None,
        "retrieved_semantic_score_observed_count": len(retrieved_semantic_scores) if applicable else None,
        "retrieved_overlap_score_available_rate": _metric_availability_rate(retrieved_overlap_scores, total=retrieved_count) if applicable else None,
        "retrieved_semantic_score_available_rate": _metric_availability_rate(retrieved_semantic_scores, total=retrieved_count) if applicable else None,
        "retrieved_average_ranking_score": _mean_or_none(retrieved_ranking_scores) if applicable else None,
        "retrieved_average_query_term_overlap": _mean_or_none(retrieved_query_term_overlaps) if applicable else None,
        "retrieved_average_heart_failure_overlap": _mean_or_none(retrieved_hf_overlaps) if applicable else None,
        "retrieved_average_clinical_term_overlap": _mean_or_none(retrieved_clinical_overlaps) if applicable else None,
        "retrieved_ranking_score_available_rate": _metric_availability_rate(retrieved_ranking_scores, total=retrieved_count) if applicable else None,
        "weighted_relevance_score": adjusted_relevance if applicable else None,
        "weighted_relevance_raw": raw_relevance if applicable else None,
        "duplicate_chunk_count": duplicate_count if applicable else None,
        "duplicate_chunk_rate": ((duplicate_count / len(ranked_chunk_ids)) if ranked_chunk_ids else 0.0) if applicable else None,
        "context_diversity_score": (1.0 - ((duplicate_count / len(ranked_chunk_ids)) if ranked_chunk_ids else 0.0)) if applicable else None,
        "soft_ndcg": _soft_ndcg(ranked_weights, ideal_weights) if applicable else None,
        "retrieved_chunk_ids": ranked_chunk_ids,
        "retrieved_chunk_count": len(ranked_chunk_ids),
        "retrieved_chunk_labels": ranked_labels,
        "strict_chunk_ids": sorted(strict_chunk_ids),
        "relevant_chunk_ids": sorted(relevant_chunk_ids),
        "source_weight_scheme": dict(_SOURCE_WEIGHTS),
    }
