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
    "retrieval.weighted_relevance_score": 0.0,
    "retrieval.soft_ndcg": 0.0,
    "generation.average_answer_similarity": 0.0,
    "generation.average_required_fact_recall": 0.0,
    "generation.forbidden_fact_violation_rate": 0.0,
    "generation.average_faithfulness_to_gold_passage": 0.0,
    "generation.average_faithfulness_to_retrieved_context": 0.0,
    "generation.hallucination_rate": 0.0,
    "generation.average_hallucination_unsupported_token_count": 0.0,
    "generation.average_warning_count": 0.0,
    "generation.verification_pass_rate": 0.0,
    "generation.exact_pass_rate": 0.0,
    "latency.average_ms": 0.0,
    "latency.p50_ms": 0.0,
    "latency.p95_ms": 0.0,
    "latency.p99_ms": 0.0,
    "latency.min_ms": 0.0,
    "latency.max_ms": 0.0,
    "latency.failure_rate": 0.0,
    "latency.timeout_rate": 0.0,
    "latency.completion_rate": 0.0,
    "latency.queue_delay_ms": 0.0,
    "latency.execution_duration_ms": 0.0,
}



def _coerce_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0



def normalize_run_metrics(
    retrieval_summary: dict[str, Any] | None,
    generation_summary: dict[str, Any] | None,
    api_summary: dict[str, Any] | None,
) -> dict[str, float]:
    retrieval_summary = retrieval_summary or {}
    generation_summary = generation_summary or {}
    api_summary = api_summary or {}
    primary_latency = api_summary.get("endpoint_summaries", {}).get("guidance_endpoint") or api_summary
    queue_delay = primary_latency.get("queue_delay") or {}
    execution_duration = primary_latency.get("execution_duration") or {}

    normalized = dict(_CORE_DEFAULTS)
    normalized.update(
        {
            "retrieval.hit_at_1": _coerce_float(retrieval_summary.get("hit_at_1")),
            "retrieval.hit_at_3": _coerce_float(retrieval_summary.get("hit_at_3")),
            "retrieval.hit_at_5": _coerce_float(retrieval_summary.get("hit_at_5")),
            "retrieval.mrr": _coerce_float(retrieval_summary.get("mrr")),
            "retrieval.strict_success_rate": _coerce_float(retrieval_summary.get("strict_success_rate")),
            "retrieval.average_overlap_score": _coerce_float(retrieval_summary.get("average_overlap_score")),
            "retrieval.average_semantic_score": _coerce_float(retrieval_summary.get("average_semantic_score")),
            "retrieval.weighted_relevance_score": _coerce_float(retrieval_summary.get("weighted_relevance_score")),
            "retrieval.soft_ndcg": _coerce_float(retrieval_summary.get("soft_ndcg")),
            "generation.average_answer_similarity": _coerce_float(generation_summary.get("average_answer_similarity")),
            "generation.average_required_fact_recall": _coerce_float(generation_summary.get("average_required_fact_recall")),
            "generation.forbidden_fact_violation_rate": _coerce_float(generation_summary.get("forbidden_fact_violation_rate")),
            "generation.average_faithfulness_to_gold_passage": _coerce_float(generation_summary.get("average_faithfulness_to_gold_passage")),
            "generation.average_faithfulness_to_retrieved_context": _coerce_float(generation_summary.get("average_faithfulness_to_retrieved_context")),
            "generation.hallucination_rate": _coerce_float(generation_summary.get("hallucination_rate")),
            "generation.average_hallucination_unsupported_token_count": _coerce_float(generation_summary.get("average_hallucination_unsupported_token_count")),
            "generation.average_warning_count": _coerce_float(generation_summary.get("average_warning_count")),
            "generation.verification_pass_rate": _coerce_float(generation_summary.get("verification_pass_rate")),
            "generation.exact_pass_rate": _coerce_float(generation_summary.get("exact_pass_rate")),
            "latency.average_ms": _coerce_float(primary_latency.get("average")) * 1000.0,
            "latency.p50_ms": _coerce_float(primary_latency.get("p50")) * 1000.0,
            "latency.p95_ms": _coerce_float(primary_latency.get("p95")) * 1000.0,
            "latency.p99_ms": _coerce_float(primary_latency.get("p99")) * 1000.0,
            "latency.min_ms": _coerce_float(primary_latency.get("min")) * 1000.0,
            "latency.max_ms": _coerce_float(primary_latency.get("max")) * 1000.0,
            "latency.failure_rate": _coerce_float(primary_latency.get("failure_rate")),
            "latency.timeout_rate": _coerce_float(primary_latency.get("timeout_rate")),
            "latency.completion_rate": _coerce_float(primary_latency.get("completion_rate", 1.0)),
            "latency.queue_delay_ms": _coerce_float(queue_delay.get("average")) * 1000.0,
            "latency.execution_duration_ms": _coerce_float(execution_duration.get("average")) * 1000.0,
        }
    )
    return normalized
