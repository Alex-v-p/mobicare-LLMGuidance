from __future__ import annotations

from typing import Any

_CORE_DEFAULTS: dict[str, float] = {
    "retrieval.hit_at_1": 0.0,
    "retrieval.hit_at_3": 0.0,
    "retrieval.hit_at_5": 0.0,
    "retrieval.mrr": 0.0,
    "retrieval.strict_success_rate": 0.0,
    "retrieval.average_overlap_score": 0.0,
    "retrieval.average_semantic_score": 0.0,
    "generation.average_answer_similarity": 0.0,
    "generation.average_required_fact_recall": 0.0,
    "generation.forbidden_fact_violation_rate": 0.0,
    "generation.average_faithfulness_to_gold_passage": 0.0,
    "generation.average_faithfulness_to_retrieved_context": 0.0,
    "generation.hallucination_rate": 0.0,
    "generation.exact_pass_rate": 0.0,
    "latency.average": 0.0,
    "latency.p50": 0.0,
    "latency.p95": 0.0,
    "latency.p99": 0.0,
    "latency.min": 0.0,
    "latency.max": 0.0,
}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default



def normalize_run_metrics(
    retrieval_summary: dict[str, Any] | None,
    generation_summary: dict[str, Any] | None,
    api_summary: dict[str, Any] | None,
) -> dict[str, float]:
    retrieval = retrieval_summary or {}
    generation = generation_summary or {}
    latency = api_summary or {}

    normalized = dict(_CORE_DEFAULTS)
    normalized.update(
        {
            "retrieval.hit_at_1": _as_float(retrieval.get("hit_at_1"), normalized["retrieval.hit_at_1"]),
            "retrieval.hit_at_3": _as_float(retrieval.get("hit_at_3"), normalized["retrieval.hit_at_3"]),
            "retrieval.hit_at_5": _as_float(retrieval.get("hit_at_5"), normalized["retrieval.hit_at_5"]),
            "retrieval.mrr": _as_float(retrieval.get("mrr"), normalized["retrieval.mrr"]),
            "retrieval.strict_success_rate": _as_float(retrieval.get("strict_success_rate"), normalized["retrieval.strict_success_rate"]),
            "retrieval.average_overlap_score": _as_float(retrieval.get("average_overlap_score"), normalized["retrieval.average_overlap_score"]),
            "retrieval.average_semantic_score": _as_float(retrieval.get("average_semantic_score"), normalized["retrieval.average_semantic_score"]),
            "generation.average_answer_similarity": _as_float(generation.get("average_answer_similarity"), normalized["generation.average_answer_similarity"]),
            "generation.average_required_fact_recall": _as_float(generation.get("average_required_fact_recall"), normalized["generation.average_required_fact_recall"]),
            "generation.forbidden_fact_violation_rate": _as_float(generation.get("forbidden_fact_violation_rate"), normalized["generation.forbidden_fact_violation_rate"]),
            "generation.average_faithfulness_to_gold_passage": _as_float(generation.get("average_faithfulness_to_gold_passage"), normalized["generation.average_faithfulness_to_gold_passage"]),
            "generation.average_faithfulness_to_retrieved_context": _as_float(generation.get("average_faithfulness_to_retrieved_context"), normalized["generation.average_faithfulness_to_retrieved_context"]),
            "generation.hallucination_rate": _as_float(generation.get("hallucination_rate"), normalized["generation.hallucination_rate"]),
            "generation.exact_pass_rate": _as_float(generation.get("exact_pass_rate"), normalized["generation.exact_pass_rate"]),
            "latency.average": _as_float(latency.get("average"), normalized["latency.average"]),
            "latency.p50": _as_float(latency.get("p50"), normalized["latency.p50"]),
            "latency.p95": _as_float(latency.get("p95"), normalized["latency.p95"]),
            "latency.p99": _as_float(latency.get("p99"), normalized["latency.p99"]),
            "latency.min": _as_float(latency.get("min"), normalized["latency.min"]),
            "latency.max": _as_float(latency.get("max"), normalized["latency.max"]),
        }
    )
    return normalized
