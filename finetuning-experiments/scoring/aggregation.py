from __future__ import annotations

from statistics import mean
from typing import Any

from .latency import summarize_latencies


def _avg(items: list[float]) -> float:
    return mean(items) if items else 0.0


def summarize_results(per_case_results: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval_items = [item["retrieval_scores"] for item in per_case_results if "retrieval_scores" in item]
    generation_items = [item["generation_scores"] for item in per_case_results if "generation_scores" in item]
    total_latencies = [float(item.get("timings", {}).get("total_latency_seconds", 0.0)) for item in per_case_results]

    retrieval_summary = {
        "case_count": len(retrieval_items),
        "hit_at_1": _avg([float(x.get("hit_at_1", 0.0)) for x in retrieval_items]),
        "hit_at_3": _avg([float(x.get("hit_at_3", 0.0)) for x in retrieval_items]),
        "hit_at_5": _avg([float(x.get("hit_at_5", 0.0)) for x in retrieval_items]),
        "mrr": _avg([float(x.get("mrr", 0.0)) for x in retrieval_items]),
        "strict_success_rate": _avg([1.0 if x.get("strict_success") else 0.0 for x in retrieval_items]),
        "average_overlap_score": _avg([float(x.get("average_overlap_score", 0.0)) for x in retrieval_items]),
        "average_semantic_score": _avg([float(x.get("average_semantic_score", 0.0)) for x in retrieval_items]),
    }
    generation_summary = {
        "case_count": len(generation_items),
        "average_answer_similarity": _avg([float(x.get("answer_similarity", 0.0)) for x in generation_items]),
        "average_required_fact_recall": _avg([float(x.get("required_fact_recall", 0.0)) for x in generation_items]),
        "forbidden_fact_violation_rate": _avg([1.0 if float(x.get("forbidden_fact_violations", 0.0)) > 0 else 0.0 for x in generation_items]),
        "average_faithfulness_to_gold_passage": _avg([float(x.get("faithfulness_to_gold_passage", 0.0)) for x in generation_items]),
        "average_faithfulness_to_retrieved_context": _avg([float(x.get("faithfulness_to_retrieved_context", 0.0)) for x in generation_items]),
        "hallucination_rate": _avg([1.0 if float(x.get("hallucination_unsupported_token_count", 0.0)) > 0 else 0.0 for x in generation_items]),
        "exact_pass_rate": _avg([1.0 if x.get("exact_pass") else 0.0 for x in generation_items]),
    }
    api_summary = summarize_latencies(total_latencies)
    api_summary["case_count"] = len(total_latencies)
    return {
        "retrieval_summary": retrieval_summary,
        "generation_summary": generation_summary,
        "api_summary": api_summary,
    }
