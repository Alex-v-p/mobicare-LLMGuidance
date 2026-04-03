from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from artifacts.compatibility import first_defined, get_nested, normalize_run_row_payload, safe_float, safe_int, safe_str
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



def normalize_run_row(run: dict[str, Any]) -> dict[str, Any]:
    payload = run.get("payload") or {}
    return normalize_run_row_payload(payload, run)


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

        deterministic_score_raw = first_defined(
            get_nested(generation_scores, "deterministic_rubric", "score"),
            generation_scores.get("answer_quality_score"),
            generation_scores.get("judge_score"),
        )
        llm_score_raw = first_defined(
            generation_scores.get("llm_judge_score"),
            get_nested(generation_scores, "llm_judge", "score"),
        )
        evaluation_profile = safe_str(generation_scores.get("evaluation_profile"))
        if evaluation_profile == "observation_only" and llm_score_raw is not None:
            display_generation_score_raw = llm_score_raw
        else:
            display_generation_score_raw = first_defined(deterministic_score_raw, llm_score_raw)

        latency_ms_raw = first_defined(
            timings.get("total_duration_ms"),
            timings.get("total_latency_ms"),
            derived.get("total_duration_ms"),
        )
        if latency_ms_raw is None and timings.get("total_latency_seconds") is not None:
            latency_ms_raw = safe_float(timings.get("total_latency_seconds")) * 1000.0

        faithfulness_raw = first_defined(
            generation_scores.get("groundedness_score"),
            generation_scores.get("faithfulness_to_retrieved_context"),
            generation_scores.get("faithfulness_to_context"),
            generation_scores.get("faithfulness_score"),
        )
        hallucination_raw = first_defined(
            generation_scores.get("hallucination_rate"),
            1.0 if safe_float(generation_scores.get("hallucination_unsupported_token_count")) > 0 else 0.0,
        )

        rows.append(
            {
                "case_id": safe_str(case.get("case_id")),
                "question": safe_str(case.get("question")),
                "status": safe_str(case.get("status") or "completed"),
                "answerability": safe_str(case.get("answerability")),
                "evaluation_profile": evaluation_profile,
                "retrieval_hit@1": safe_float(retrieval_scores.get("hit_at_1")),
                "retrieval_hit@3": safe_float(retrieval_scores.get("hit_at_3")),
                "strict_hit@3": safe_float(retrieval_scores.get("strict_hit_at_3")),
                "mrr": safe_float(retrieval_scores.get("mrr")),
                "strict_mrr": safe_float(retrieval_scores.get("strict_mrr")),
                "weighted_relevance_score": safe_float(retrieval_scores.get("weighted_relevance_score")),
                "lenient_success_score": safe_float(retrieval_scores.get("lenient_success_score")),
                "duplicate_chunk_rate": safe_float(retrieval_scores.get("duplicate_chunk_rate")),
                "retrieved_overlap_available_rate": safe_float(retrieval_scores.get("retrieved_overlap_score_available_rate")),
                "retrieved_semantic_available_rate": safe_float(retrieval_scores.get("retrieved_semantic_score_available_rate")),
                "retrieved_avg_ranking_score": safe_float(retrieval_scores.get("retrieved_average_ranking_score")),
                "answer_similarity": safe_float(first_defined(generation_scores.get("answer_similarity"), generation_scores.get("reference_token_f1"))),
                "answer_quality_score": safe_float(first_defined(generation_scores.get("answer_quality_score"), deterministic_score_raw)),
                "deterministic_rubric_score": safe_float(deterministic_score_raw),
                "judge_score": safe_float(first_defined(generation_scores.get("judge_score"), deterministic_score_raw)),
                "llm_judge_score": safe_float(llm_score_raw),
                "generation_score_display": safe_float(display_generation_score_raw),
                "fact_recall": safe_float(generation_scores.get("required_fact_recall")),
                "faithfulness": safe_float(faithfulness_raw),
                "hallucination_rate": safe_float(hallucination_raw),
                "exact_pass": safe_float(generation_scores.get("exact_pass")),
                "warning_count": len(case.get("warnings") or []),
                "retrieved_chunk_count": len(case.get("retrieved_chunks") or []),
                "direct_evidence_count": len(source_list.get("direct_evidence") or []),
                "supporting_count": len(source_list.get("supporting") or []),
                "total_latency_ms": safe_float(latency_ms_raw),
                "queue_delay_ms": safe_float(derived.get("queue_delay_ms")),
                "execution_duration_ms": safe_float(derived.get("execution_duration_ms")),
                "api_failure_category": safe_str((case.get("raw_endpoint_result") or {}).get("error")),
                "generated_answer": safe_str(case.get("generated_answer")),
            }
        )
    return pd.DataFrame(rows)
