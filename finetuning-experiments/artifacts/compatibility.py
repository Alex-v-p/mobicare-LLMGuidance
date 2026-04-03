from __future__ import annotations

import hashlib
import json
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def first_defined(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _patch_stage_latency_summary(summary: dict[str, Any]) -> dict[str, Any]:
    patched: dict[str, Any] = {}
    for stage_name, value in (summary or {}).items():
        if not isinstance(value, dict):
            patched[stage_name] = value
            continue
        stage = dict(value)
        completed_count = safe_int(stage.get("completed_count"))
        included_count = safe_int(first_defined(stage.get("included_count"), stage.get("timed_count")))
        count = safe_int(stage.get("count"))
        if count <= 0:
            count = max(completed_count, included_count)
        stage.setdefault("timed_count", included_count)
        stage.setdefault("untimed_count", max(0, count - safe_int(stage.get("timed_count"))))
        stage["count"] = count
        patched[stage_name] = stage
    return patched




def backfill_retrieval_summary_fields(retrieval: dict[str, Any]) -> dict[str, Any]:
    retrieval = dict(retrieval or {})
    legacy = retrieval.get("weighted_relevance_score")
    v2 = first_defined(retrieval.get("weighted_relevance_score_v2"))
    display = first_defined(retrieval.get("weighted_relevance_display"), v2, legacy)
    retrieval.setdefault("weighted_relevance_score", legacy)
    retrieval.setdefault("weighted_relevance_score_v2", v2)
    retrieval.setdefault("weighted_relevance_display", display)
    retrieval.setdefault("weighted_relevance_score_source", "legacy_v1" if legacy is not None else None)
    retrieval.setdefault("weighted_relevance_score_v2_source", "relevance_gated_v2" if v2 is not None else None)
    retrieval.setdefault(
        "weighted_relevance_display_source",
        "relevance_gated_v2" if retrieval.get("weighted_relevance_score_v2") is not None else retrieval.get("weighted_relevance_score_source"),
    )
    return retrieval

def backfill_generation_summary_fields(generation: dict[str, Any]) -> dict[str, Any]:
    generation = dict(generation or {})
    deterministic = first_defined(
        generation.get("average_deterministic_rubric_score"),
        generation.get("average_answer_quality_score"),
        generation.get("average_judge_score"),
    )
    llm = first_defined(generation.get("average_llm_judge_score"))
    effective = first_defined(
        generation.get("average_effective_generation_score"),
        generation.get("average_primary_generation_score"),
    )
    deterministic_cases = safe_int(generation.get("deterministic_applicable_case_count"))
    if effective is None:
        if deterministic_cases == 0 and llm is not None:
            effective = llm
        else:
            effective = first_defined(deterministic, llm)
    generation.setdefault("average_answer_quality_score", deterministic)
    generation.setdefault("average_deterministic_rubric_score", deterministic)
    generation.setdefault("average_judge_score", deterministic)
    generation.setdefault("average_llm_judge_score", llm)
    generation.setdefault("average_effective_generation_score", effective)
    generation.setdefault("average_primary_generation_score", effective)
    generation.setdefault("average_verification_alignment_score", generation.get("average_verification_alignment_score"))
    generation.setdefault("verification_alignment_rate", generation.get("verification_alignment_rate"))
    generation.setdefault("verification_alignment_applicable_case_count", generation.get("verification_alignment_applicable_case_count"))
    generation.setdefault("average_verification_intrinsic_quality_score", generation.get("average_verification_intrinsic_quality_score"))
    return generation


def resolve_run_generation_score(generation: dict[str, Any]) -> tuple[Any, Any, Any]:
    generation = backfill_generation_summary_fields(generation)
    deterministic = generation.get("average_deterministic_rubric_score")
    llm = generation.get("average_llm_judge_score")
    effective = first_defined(
        generation.get("average_effective_generation_score"),
        generation.get("average_primary_generation_score"),
        deterministic,
        llm,
    )
    return deterministic, llm, effective


def build_config_overview(payload: dict[str, Any]) -> dict[str, Any]:
    existing = payload.get("config_overview")
    if isinstance(existing, dict) and existing:
        return existing
    config = payload.get("config") or {}
    execution = config.get("execution") or {}
    return {
        "ingestion": dict(config.get("ingestion") or {}),
        "inference": dict(config.get("inference") or {}),
        "source_mapping": dict(config.get("source_mapping") or {}),
        "evaluation": dict(config.get("evaluation") or {}),
        "execution": {
            "api_test": dict(execution.get("api_test") or {}),
            "environment": dict(execution.get("environment") or {}),
        },
    }


def build_telemetry_summary(payload: dict[str, Any]) -> dict[str, Any]:
    existing = payload.get("telemetry_summary")
    if isinstance(existing, dict) and existing:
        summary = dict(existing)
    else:
        api_summary = payload.get("api_summary") or {}
        endpoint_summaries = api_summary.get("endpoint_summaries") or {}
        primary_api = endpoint_summaries.get("guidance_endpoint") or api_summary
        summary = {
            "success_only": dict(primary_api.get("success_only") or {}),
            "queue_delay": dict(primary_api.get("queue_delay") or {}),
            "execution_duration": dict(primary_api.get("execution_duration") or {}),
            "stage_latency_summary": dict(primary_api.get("stage_latency_summary") or {}),
            "endpoint_summaries": dict(endpoint_summaries),
            "failure_taxonomy": dict(api_summary.get("failure_taxonomy") or {}),
        }
    summary["stage_latency_summary"] = _patch_stage_latency_summary(dict(summary.get("stage_latency_summary") or {}))
    return summary


def normalize_run_row_payload(payload: dict[str, Any], run: dict[str, Any] | None = None) -> dict[str, Any]:
    run_meta = run or {}
    retrieval = backfill_retrieval_summary_fields(payload.get("retrieval_summary") or {})
    generation = backfill_generation_summary_fields(payload.get("generation_summary") or {})
    api = payload.get("api_summary") or {}
    primary_api = (api.get("endpoint_summaries") or {}).get("guidance_endpoint") or api
    ingestion = payload.get("ingestion_summary") or {}
    telemetry = build_telemetry_summary(payload)
    source_mapping = payload.get("source_mapping_summary") or {}
    normalized = payload.get("normalized_metrics") or {}
    config_overview = build_config_overview(payload)
    ingestion_config = config_overview.get("ingestion") or {}
    inference_config = config_overview.get("inference") or {}
    source_mapping_config = config_overview.get("source_mapping") or {}

    avg_deterministic_raw, avg_llm_raw, avg_effective_raw = resolve_run_generation_score(generation)
    avg_faithfulness_raw = first_defined(
        generation.get("average_groundedness_score"),
        generation.get("average_faithfulness_to_retrieved_context"),
        generation.get("average_faithfulness_to_context"),
        generation.get("average_faithfulness_score"),
    )

    return {
        "run_id": safe_str(payload.get("run_id") or run_meta.get("run_id")),
        "label": safe_str(payload.get("label")),
        "datetime": safe_str(payload.get("datetime")),
        "dataset_version": safe_str(payload.get("dataset_version")),
        "documents_version": safe_str(payload.get("documents_version")),
        "summary_path": run_meta.get("summary_path"),
        "artifact_path": run_meta.get("artifact_path"),
        "artifact_type": safe_str(payload.get("artifact_type")),
        "artifact_version": safe_str(payload.get("artifact_version")),
        "case_count": safe_int(payload.get("case_count") or len(payload.get("per_case_results") or [])),
        "chunking_strategy": safe_str(ingestion_config.get("chunking_strategy") or get_nested(payload, "config", "ingestion", "chunking_strategy")),
        "retrieval_mode": safe_str(inference_config.get("retrieval_mode") or get_nested(payload, "config", "inference", "retrieval_mode")),
        "llm_model": safe_str(inference_config.get("llm_model") or get_nested(payload, "config", "inference", "llm_model")),
        "prompt_label": safe_str(inference_config.get("prompt_engineering_label") or get_nested(payload, "config", "inference", "prompt_engineering_label")),
        "pipeline_variant": safe_str(first_defined(inference_config.get("pipeline_variant"), get_nested(payload, "config", "inference", "pipeline_variant"), default="standard")),
        "query_rewriting": bool(inference_config.get("enable_query_rewriting") or get_nested(payload, "config", "inference", "enable_query_rewriting", default=False)),
        "verification": bool(inference_config.get("enable_response_verification") or get_nested(payload, "config", "inference", "enable_response_verification", default=False)),
        "graph_augmentation": bool(inference_config.get("use_graph_augmentation") or get_nested(payload, "config", "inference", "use_graph_augmentation", default=False)),
        "second_pass_mapping": bool(source_mapping_config.get("llm_second_pass_enabled") or get_nested(payload, "config", "source_mapping", "llm_second_pass_enabled", default=False)),
        "hit@1": safe_float(retrieval.get("hit_at_1")),
        "hit@3": safe_float(retrieval.get("hit_at_3")),
        "hit@5": safe_float(retrieval.get("hit_at_5")),
        "mrr": safe_float(retrieval.get("mrr")),
        "strict_hit@1": safe_float(retrieval.get("strict_hit_at_1")),
        "strict_hit@3": safe_float(retrieval.get("strict_hit_at_3")),
        "strict_hit@5": safe_float(retrieval.get("strict_hit_at_5")),
        "strict_mrr": safe_float(retrieval.get("strict_mrr")),
        "weighted_relevance": safe_float(retrieval.get("weighted_relevance_score")),
        "weighted_relevance_v2": safe_float(retrieval.get("weighted_relevance_score_v2"), default=float("nan")),
        "weighted_relevance_display": safe_float(retrieval.get("weighted_relevance_display", retrieval.get("weighted_relevance_score"))),
        "lenient_success_score": safe_float(retrieval.get("lenient_success_score")),
        "duplicate_chunk_rate": safe_float(retrieval.get("duplicate_chunk_rate")),
        "context_diversity_score": safe_float(retrieval.get("context_diversity_score")),
        "soft_ndcg": safe_float(retrieval.get("soft_ndcg")),
        "retrieved_overlap_available_rate": safe_float(retrieval.get("retrieved_overlap_score_available_rate")),
        "retrieved_semantic_available_rate": safe_float(retrieval.get("retrieved_semantic_score_available_rate")),
        "retrieved_ranking_available_rate": safe_float(retrieval.get("retrieved_ranking_score_available_rate")),
        "retrieved_avg_ranking_score": safe_float(retrieval.get("retrieved_average_ranking_score")),
        "avg_answer_similarity": safe_float(generation.get("average_answer_similarity")),
        "avg_answer_quality": safe_float(first_defined(generation.get("average_answer_quality_score"), avg_deterministic_raw)),
        "avg_deterministic_rubric": safe_float(avg_deterministic_raw),
        "avg_judge_score": safe_float(generation.get("average_judge_score")),
        "avg_llm_judge_score": safe_float(avg_llm_raw),
        "avg_effective_generation_score": safe_float(avg_effective_raw),
        "avg_primary_generation_score": safe_float(generation.get("average_primary_generation_score")),
        "avg_fact_recall": safe_float(generation.get("average_required_fact_recall")),
        "avg_faithfulness": safe_float(avg_faithfulness_raw),
        "exact_pass_rate": safe_float(generation.get("exact_pass_rate")),
        "verification_pass_rate": safe_float(generation.get("verification_pass_rate")),
        "avg_verification_alignment_score": safe_float(generation.get("average_verification_alignment_score"), default=float("nan")),
        "verification_alignment_rate": safe_float(generation.get("verification_alignment_rate"), default=float("nan")),
        "avg_verification_intrinsic_quality_score": safe_float(generation.get("average_verification_intrinsic_quality_score"), default=float("nan")),
        "forbidden_violation_rate": safe_float(generation.get("forbidden_fact_violation_rate")),
        "hallucination_rate": safe_float(generation.get("hallucination_rate")),
        "avg_latency": safe_float(primary_api.get("average")),
        "p50_latency": safe_float(primary_api.get("p50")),
        "p95_latency": safe_float(primary_api.get("p95")),
        "p99_latency": safe_float(primary_api.get("p99")),
        "api_failure_rate": safe_float(primary_api.get("failure_rate")),
        "api_timeout_rate": safe_float(primary_api.get("timeout_rate")),
        "api_completion_rate": safe_float(primary_api.get("completion_rate")),
        "queue_delay_avg": safe_float(get_nested(telemetry, "queue_delay", "average")),
        "execution_duration_avg": safe_float(get_nested(telemetry, "execution_duration", "average")),
        "chunks_created": safe_float(first_defined(ingestion.get("chunks_created"), get_nested(ingestion, "raw_endpoint_result", "chunks_created"))),
        "vectors_upserted": safe_float(first_defined(ingestion.get("vectors_upserted"), get_nested(ingestion, "raw_endpoint_result", "vectors_upserted"))),
        "avg_chunk_length": safe_float(get_nested(ingestion, "ingestion_metrics", "average_chunk_length")),
        "page_coverage_ratio": safe_float(get_nested(ingestion, "ingestion_metrics", "page_coverage_ratio")),
        "source_map_cases": safe_int(first_defined(source_mapping.get("case_count"), len(source_mapping.get("case_chunk_assignments") or []))),
        "direct_evidence_total": safe_int(get_nested(source_mapping, "label_totals", "direct_evidence")),
        "normalized.strict_success_rate": safe_float(normalized.get("retrieval.strict_success_rate")),
        "normalized.avg_answer_similarity": safe_float(normalized.get("generation.average_answer_similarity")),
        "normalized.avg_answer_quality": safe_float(normalized.get("generation.average_answer_quality_score")),
        "normalized.avg_deterministic_rubric": safe_float(normalized.get("generation.average_deterministic_rubric_score")),
        "normalized.avg_judge_score": safe_float(normalized.get("generation.average_judge_score")),
        "normalized.avg_llm_judge_score": safe_float(normalized.get("generation.average_llm_judge_score")),
        "normalized.avg_effective_generation_score": safe_float(normalized.get("generation.average_effective_generation_score")),
        "normalized.avg_verification_alignment_score": safe_float(
            normalized.get("generation.average_verification_alignment_score"),
            default=float("nan"),
        ),
        "normalized.verification_alignment_rate": safe_float(
            normalized.get("generation.verification_alignment_rate"),
            default=float("nan"),
        ),
        "normalized.weighted_relevance_v2": safe_float(
            first_defined(normalized.get("retrieval.weighted_relevance_score_v2"), retrieval.get("weighted_relevance_score_v2")),
            default=float("nan"),
        ),
        "normalized.weighted_relevance_display": safe_float(
            first_defined(normalized.get("retrieval.weighted_relevance_display"), retrieval.get("weighted_relevance_display"), retrieval.get("weighted_relevance_score"))
        ),
        "normalized.avg_latency": safe_float(normalized.get("latency.average_ms")),
    }


def stable_digest(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
