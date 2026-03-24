from __future__ import annotations

from statistics import mean
from typing import Any

from .latency import summarize_latencies, summarize_stage_latencies


def _avg(items: list[float]) -> float:
    return mean(items) if items else 0.0


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg_defined(values: list[Any]) -> float:
    numeric = [item for item in (_to_float(value) for value in values) if item is not None]
    return _avg(numeric)


def summarize_results(per_case_results: list[dict[str, Any]]) -> dict[str, Any]:
    completed_cases = [item for item in per_case_results if item.get("status") != "failed"]
    retrieval_items = [item["retrieval_scores"] for item in per_case_results if "retrieval_scores" in item]
    generation_items = [item["generation_scores"] for item in per_case_results if "generation_scores" in item]
    deterministic_items = [
        item for item in generation_items
        if (item.get("deterministic_rubric") or {}).get("applicable", (item.get("deterministic_rubric") or {}).get("enabled", True))
    ]

    stage_lists = [list((item.get("telemetry") or {}).get("stages") or []) for item in per_case_results]
    derived_timings = [dict((item.get("telemetry") or {}).get("derived") or {}) for item in per_case_results]
    total_latencies = [
        float(derived.get("total_duration_ms")) / 1000.0
        for derived in derived_timings
        if derived.get("total_duration_ms") is not None
    ]
    if not total_latencies:
        total_latencies = [float(item.get("timings", {}).get("total_latency_seconds", 0.0)) for item in per_case_results]

    queue_latencies = [
        float(derived.get("queue_delay_ms")) / 1000.0
        for derived in derived_timings
        if derived.get("queue_delay_ms") is not None
    ]
    execution_latencies = [
        float(derived.get("execution_duration_ms")) / 1000.0
        for derived in derived_timings
        if derived.get("execution_duration_ms") is not None
    ]

    retrieval_summary = {
        "completed_case_count": len(completed_cases),
        "failed_case_count": sum(1 for item in per_case_results if item.get("status") == "failed"),
        "case_count": len(retrieval_items),
        "hit_at_1": _avg([float(x.get("hit_at_1", 0.0)) for x in retrieval_items]),
        "hit_at_3": _avg([float(x.get("hit_at_3", 0.0)) for x in retrieval_items]),
        "hit_at_5": _avg([float(x.get("hit_at_5", 0.0)) for x in retrieval_items]),
        "mrr": _avg([float(x.get("mrr", 0.0)) for x in retrieval_items]),
        "strict_success_rate": _avg([1.0 if x.get("strict_success") else 0.0 for x in retrieval_items]),
        "lenient_success_score": _avg([float(x.get("lenient_success_score", 0.0)) for x in retrieval_items]),
        "average_overlap_score": _avg([float(x.get("average_overlap_score", 0.0)) for x in retrieval_items]),
        "average_semantic_score": _avg([float(x.get("average_semantic_score", 0.0)) for x in retrieval_items]),
        "weighted_relevance_score": _avg([float(x.get("weighted_relevance_score", 0.0)) for x in retrieval_items]),
        "soft_ndcg": _avg([float(x.get("soft_ndcg", 0.0)) for x in retrieval_items]),
        "retrieved_average_overlap_score": _avg([float(x.get("retrieved_average_overlap_score", 0.0)) for x in retrieval_items]),
        "retrieved_average_semantic_score": _avg([float(x.get("retrieved_average_semantic_score", 0.0)) for x in retrieval_items]),
        "duplicate_chunk_rate": _avg([float(x.get("duplicate_chunk_rate", 0.0)) for x in retrieval_items]),
        "context_diversity_score": _avg([float(x.get("context_diversity_score", 0.0)) for x in retrieval_items]),
    }
    warning_counts = [len(item.get("warnings") or []) for item in per_case_results]
    verification_items = [item.get("verification") or {} for item in per_case_results]
    generation_summary = {
        "case_count": len(generation_items),
        "deterministic_applicable_case_count": len(deterministic_items),
        "observation_only_case_count": sum(1 for x in generation_items if x.get("evaluation_profile") == "observation_only"),
        "average_answer_similarity": _avg([float(x.get("answer_similarity", 0.0)) for x in generation_items]),
        "average_answer_quality_score": _avg_defined([x.get("answer_quality_score") for x in deterministic_items]),
        "average_deterministic_rubric_score": _avg_defined([(x.get("deterministic_rubric") or {}).get("score", x.get("answer_quality_score")) for x in deterministic_items]),
        "average_judge_score": _avg_defined([x.get("judge_score") for x in generation_items]),
        "average_llm_judge_score": _avg_defined([(x.get("llm_judge") or {}).get("score", x.get("judge_score")) for x in generation_items]),
        "average_reference_token_f1": _avg([float(x.get("reference_token_f1", 0.0)) for x in generation_items]),
        "llm_judge_enabled_rate": _avg([1.0 if (x.get("llm_judge") or {}).get("enabled") else 0.0 for x in generation_items]),
        "llm_judge_error_rate": _avg([1.0 if str((x.get("llm_judge") or {}).get("error") or "").strip() else 0.0 for x in generation_items]),
        "average_required_fact_recall": _avg([float(x.get("required_fact_recall", 0.0)) for x in generation_items]),
        "forbidden_fact_violation_rate": _avg([1.0 if float(x.get("forbidden_fact_violations", 0.0)) > 0 else 0.0 for x in generation_items]),
        "average_faithfulness_to_gold_passage": _avg_defined([x.get("faithfulness_to_gold_passage") for x in deterministic_items]),
        "average_groundedness_score": _avg([float(x.get("groundedness_score", 0.0)) for x in generation_items]),
        "average_faithfulness_to_retrieved_context": _avg([float(x.get("faithfulness_to_retrieved_context", 0.0)) for x in generation_items]),
        "average_hallucination_unsupported_token_count": _avg([float(x.get("hallucination_unsupported_token_count", 0.0)) for x in generation_items]),
        "hallucination_rate": _avg([1.0 if float(x.get("hallucination_unsupported_token_count", 0.0)) > 0 else 0.0 for x in generation_items]),
        "average_warning_count": _avg([float(count) for count in warning_counts]),
        "warning_case_rate": _avg([1.0 if count > 0 else 0.0 for count in warning_counts]),
        "verification_pass_rate": _avg([1.0 if str(item.get("verdict") or "").lower() in {"pass", "passed", "ok", "success"} else 0.0 for item in verification_items]),
        "exact_pass_rate": _avg([1.0 if x.get("exact_pass") else 0.0 for x in deterministic_items]),
    }
    successful_total_latencies = [float(item.get("timings", {}).get("total_latency_seconds", 0.0)) for item in completed_cases]
    if derived_timings:
        successful_total_latencies = [
            float((item.get("telemetry") or {}).get("derived", {}).get("total_duration_ms")) / 1000.0
            for item in completed_cases
            if (item.get("telemetry") or {}).get("derived", {}).get("total_duration_ms") is not None
        ] or successful_total_latencies
    api_summary = summarize_latencies(total_latencies, policy="all", outlier_policy="keep_all")
    api_summary.update(
        {
            "case_count": len(total_latencies),
            "success_only": summarize_latencies(successful_total_latencies, policy="success_only", outlier_policy="keep_all"),
            "queue_delay": summarize_latencies(queue_latencies, policy="success_only", outlier_policy="keep_all"),
            "execution_duration": summarize_latencies(execution_latencies, policy="success_only", outlier_policy="keep_all"),
            "failure_count": sum(1 for item in per_case_results if item.get("status") == "failed"),
            "success_count": len(completed_cases),
            "stage_latency_summary": summarize_stage_latencies(stage_lists),
        }
    )
    return {
        "retrieval_summary": retrieval_summary,
        "generation_summary": generation_summary,
        "api_summary": api_summary,
    }
