from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from artifacts.loader import list_run_artifacts, list_run_summaries, load_run_artifact
from utils.json import read_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = ROOT / "artifacts" / "runs"


@st.cache_data(show_spinner=False)
def discover_runs(output_dir: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    root = Path(output_dir)
    summary_paths = list_run_summaries(root)
    artifact_paths = list_run_artifacts(root)
    artifact_by_run_id = {path.stem: str(path) for path in artifact_paths}

    runs: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    for path in summary_paths:
        payload = load_run_artifact(path)
        run_id = str(payload.get("run_id") or path.name.replace(".summary.json", ""))
        seen_run_ids.add(run_id)
        runs.append(
            {
                "run_id": run_id,
                "summary_path": str(path),
                "artifact_path": artifact_by_run_id.get(run_id),
                "payload": payload,
                "is_summary": True,
            }
        )

    for path in artifact_paths:
        run_id = path.stem
        if run_id in seen_run_ids:
            continue
        payload = load_run_artifact(path)
        runs.append(
            {
                "run_id": run_id,
                "summary_path": None,
                "artifact_path": str(path),
                "payload": payload,
                "is_summary": False,
            }
        )

    runs.sort(key=lambda item: str((item.get("payload") or {}).get("datetime") or ""), reverse=True)
    return runs, artifact_by_run_id


@st.cache_data(show_spinner=False)
def load_full_run(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return load_run_artifact(path)


@st.cache_data(show_spinner=False)
def discover_campaigns(output_dir: str) -> list[dict[str, Any]]:
    root = Path(output_dir) / "_campaigns"
    if not root.exists():
        return []
    campaigns: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*.campaign.json"), reverse=True):
        payload = read_json(path)
        payload["artifact_path"] = str(path)
        campaigns.append(payload)
    campaigns.sort(key=lambda item: str(item.get("datetime") or ""), reverse=True)
    return campaigns


@st.cache_data(show_spinner=False)
def load_source_maps(output_dir: str, run_id: str) -> dict[str, Any]:
    path = Path(output_dir) / f"{run_id}.source_maps.json"
    if not path.exists():
        return {}
    return read_json(path)



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



def normalize_run_row(run: dict[str, Any]) -> dict[str, Any]:
    payload = run.get("payload") or {}
    retrieval = payload.get("retrieval_summary") or {}
    generation = payload.get("generation_summary") or {}
    api = payload.get("api_summary") or {}
    primary_api = (api.get("endpoint_summaries") or {}).get("guidance_endpoint") or api
    ingestion = payload.get("ingestion_summary") or {}
    telemetry = payload.get("telemetry_summary") or {}
    source_mapping = payload.get("source_mapping_summary") or {}
    normalized = payload.get("normalized_metrics") or {}
    config_overview = payload.get("config_overview") or {}
    ingestion_config = (config_overview.get("ingestion") or {})
    inference_config = (config_overview.get("inference") or {})
    source_mapping_config = (config_overview.get("source_mapping") or {})

    return {
        "run_id": safe_str(payload.get("run_id") or run.get("run_id")),
        "label": safe_str(payload.get("label")),
        "datetime": safe_str(payload.get("datetime")),
        "dataset_version": safe_str(payload.get("dataset_version")),
        "documents_version": safe_str(payload.get("documents_version")),
        "summary_path": run.get("summary_path"),
        "artifact_path": run.get("artifact_path"),
        "artifact_type": safe_str(payload.get("artifact_type")),
        "case_count": safe_int(payload.get("case_count") or len(payload.get("per_case_results") or [])),
        "chunking_strategy": safe_str(ingestion_config.get("chunking_strategy") or get_nested(payload, "config", "ingestion", "chunking_strategy")),
        "retrieval_mode": safe_str(inference_config.get("retrieval_mode") or get_nested(payload, "config", "inference", "retrieval_mode")),
        "llm_model": safe_str(inference_config.get("llm_model") or get_nested(payload, "config", "inference", "llm_model")),
        "prompt_label": safe_str(inference_config.get("prompt_engineering_label") or get_nested(payload, "config", "inference", "prompt_engineering_label")),
        "query_rewriting": bool(inference_config.get("enable_query_rewriting") or get_nested(payload, "config", "inference", "enable_query_rewriting", default=False)),
        "verification": bool(inference_config.get("enable_response_verification") or get_nested(payload, "config", "inference", "enable_response_verification", default=False)),
        "graph_augmentation": bool(inference_config.get("use_graph_augmentation") or get_nested(payload, "config", "inference", "use_graph_augmentation", default=False)),
        "second_pass_mapping": bool(source_mapping_config.get("llm_second_pass_enabled") or get_nested(payload, "config", "source_mapping", "llm_second_pass_enabled", default=False)),
        "hit@1": safe_float(retrieval.get("hit_at_1")),
        "hit@3": safe_float(retrieval.get("hit_at_3")),
        "hit@5": safe_float(retrieval.get("hit_at_5")),
        "mrr": safe_float(retrieval.get("mrr")),
        "weighted_relevance": safe_float(retrieval.get("weighted_relevance_score")),
        "lenient_success_score": safe_float(retrieval.get("lenient_success_score")),
        "duplicate_chunk_rate": safe_float(retrieval.get("duplicate_chunk_rate")),
        "context_diversity_score": safe_float(retrieval.get("context_diversity_score")),
        "soft_ndcg": safe_float(retrieval.get("soft_ndcg")),
        "avg_answer_similarity": safe_float(generation.get("average_answer_similarity")),
        "avg_answer_quality": safe_float(generation.get("average_answer_quality_score")),
        "avg_deterministic_rubric": safe_float(generation.get("average_deterministic_rubric_score") or generation.get("average_answer_quality_score")),
        "avg_judge_score": safe_float(generation.get("average_judge_score")),
        "avg_llm_judge_score": safe_float(generation.get("average_llm_judge_score") or generation.get("average_judge_score")),
        "avg_fact_recall": safe_float(generation.get("average_required_fact_recall")),
        "avg_faithfulness": safe_float(generation.get("average_faithfulness_score") or generation.get("average_faithfulness_to_context")),
        "exact_pass_rate": safe_float(generation.get("exact_pass_rate")),
        "verification_pass_rate": safe_float(generation.get("verification_pass_rate")),
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
        "chunks_created": safe_float(ingestion.get("chunks_created")),
        "vectors_upserted": safe_float(ingestion.get("vectors_upserted")),
        "avg_chunk_length": safe_float(get_nested(ingestion, "ingestion_metrics", "average_chunk_length")),
        "page_coverage_ratio": safe_float(get_nested(ingestion, "ingestion_metrics", "page_coverage_ratio")),
        "source_map_cases": safe_int(source_mapping.get("case_count")),
        "direct_evidence_total": safe_int(get_nested(source_mapping, "label_totals", "direct_evidence")),
        "normalized.strict_success_rate": safe_float(normalized.get("retrieval.strict_success_rate")),
        "normalized.avg_answer_similarity": safe_float(normalized.get("generation.average_answer_similarity")),
        "normalized.avg_answer_quality": safe_float(normalized.get("generation.average_answer_quality_score")),
        "normalized.avg_deterministic_rubric": safe_float(normalized.get("generation.average_deterministic_rubric_score")),
        "normalized.avg_judge_score": safe_float(normalized.get("generation.average_judge_score")),
        "normalized.avg_llm_judge_score": safe_float(normalized.get("generation.average_llm_judge_score")),
        "normalized.avg_latency": safe_float(normalized.get("latency.average_ms")),
    }



def build_runs_dataframe(runs: list[dict[str, Any]]) -> pd.DataFrame:
    if not runs:
        return pd.DataFrame()
    frame = pd.DataFrame([normalize_run_row(run) for run in runs])
    if "datetime" in frame.columns:
        frame["datetime_parsed"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.sort_values(by="datetime_parsed", ascending=False)
    return frame



def metric_card(label: str, value: Any, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)



def build_case_dataframe(artifact: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in artifact.get("per_case_results") or []:
        retrieval_scores = case.get("retrieval_scores") or {}
        generation_scores = case.get("generation_scores") or {}
        timings = case.get("timings") or {}
        derived = (case.get("telemetry") or {}).get("derived") or {}
        source_list = case.get("source_list") or {}
        rows.append(
            {
                "case_id": safe_str(case.get("case_id")),
                "question": safe_str(case.get("question")),
                "status": safe_str(case.get("status") or "completed"),
                "answerability": safe_str(case.get("answerability")),
                "retrieval_hit@1": safe_float(retrieval_scores.get("hit_at_1")),
                "retrieval_hit@3": safe_float(retrieval_scores.get("hit_at_3")),
                "mrr": safe_float(retrieval_scores.get("mrr")),
                "weighted_relevance_score": safe_float(retrieval_scores.get("weighted_relevance_score")),
                "lenient_success_score": safe_float(retrieval_scores.get("lenient_success_score")),
                "duplicate_chunk_rate": safe_float(retrieval_scores.get("duplicate_chunk_rate")),
                "answer_similarity": safe_float(generation_scores.get("answer_similarity")),
                "answer_quality_score": safe_float(generation_scores.get("answer_quality_score")),
                "deterministic_rubric_score": safe_float(get_nested(generation_scores, "deterministic_rubric", "score", default=generation_scores.get("answer_quality_score"))),
                "judge_score": safe_float(generation_scores.get("judge_score")),
                "llm_judge_score": safe_float(get_nested(generation_scores, "llm_judge", "score", default=generation_scores.get("judge_score"))),
                "fact_recall": safe_float(generation_scores.get("required_fact_recall")),
                "faithfulness": safe_float(generation_scores.get("faithfulness_score") or generation_scores.get("faithfulness_to_context")),
                "hallucination_rate": safe_float(generation_scores.get("hallucination_rate")),
                "exact_pass": safe_float(generation_scores.get("exact_pass")),
                "warning_count": len(case.get("warnings") or []),
                "retrieved_chunk_count": len(case.get("retrieved_chunks") or []),
                "direct_evidence_count": len(source_list.get("direct_evidence") or []),
                "supporting_count": len(source_list.get("supporting") or []),
                "total_latency_ms": safe_float(timings.get("total_latency_ms") or derived.get("total_duration_ms")),
                "queue_delay_ms": safe_float(derived.get("queue_delay_ms")),
                "execution_duration_ms": safe_float(derived.get("execution_duration_ms")),
                "api_failure_category": safe_str((case.get("raw_endpoint_result") or {}).get("error")),
                "generated_answer": safe_str(case.get("generated_answer")),
            }
        )
    return pd.DataFrame(rows)
