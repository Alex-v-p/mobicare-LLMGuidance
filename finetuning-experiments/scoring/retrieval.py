from __future__ import annotations

from typing import Any


def score_retrieval(expected_matches: list[dict[str, Any]], retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    expected_sets = [tuple(match.get("chunk_ids", [])) for match in expected_matches]
    expected_flat = {chunk_id for group in expected_sets for chunk_id in group}
    ranked_chunk_ids = [chunk.get("chunk_id") for chunk in retrieved_chunks if chunk.get("chunk_id")]

    first_rank = None
    for index, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in expected_flat:
            first_rank = index
            break

    def hit_at(k: int) -> float:
        return 1.0 if any(chunk_id in expected_flat for chunk_id in ranked_chunk_ids[:k]) else 0.0

    overlap_scores = [float(match.get("metadata", {}).get("passage_coverage", 0.0)) for match in expected_matches]
    semantic_scores = [float(match.get("semantic_score", 0.0)) for match in expected_matches]

    return {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "strict_success": bool(first_rank),
        "average_overlap_score": (sum(overlap_scores) / len(overlap_scores)) if overlap_scores else 0.0,
        "average_semantic_score": (sum(semantic_scores) / len(semantic_scores)) if semantic_scores else 0.0,
        "retrieved_chunk_ids": ranked_chunk_ids,
        "expected_chunk_ids": sorted(expected_flat),
    }
