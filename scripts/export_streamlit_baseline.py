from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FINETUNING_ROOT = REPO_ROOT / "finetuning-experiments"
if str(FINETUNING_ROOT) not in sys.path:
    sys.path.insert(0, str(FINETUNING_ROOT))

from artifacts.compatibility import normalize_run_row_payload, stable_digest  # noqa: E402
from artifacts.loader import list_run_artifacts, list_run_summaries, load_run_artifact  # noqa: E402
from utils.json import write_json  # noqa: E402


BASELINE_FIELDS = [
    "run_id",
    "label",
    "artifact_type",
    "artifact_version",
    "datetime",
    "dataset_version",
    "documents_version",
    "case_count",
    "chunking_strategy",
    "retrieval_mode",
    "llm_model",
    "prompt_label",
    "pipeline_variant",
    "query_rewriting",
    "verification",
    "graph_augmentation",
    "second_pass_mapping",
    "hit@1",
    "hit@3",
    "hit@5",
    "mrr",
    "strict_hit@3",
    "strict_mrr",
    "weighted_relevance",
    "lenient_success_score",
    "duplicate_chunk_rate",
    "context_diversity_score",
    "soft_ndcg",
    "avg_answer_similarity",
    "avg_answer_quality",
    "avg_deterministic_rubric",
    "avg_judge_score",
    "avg_llm_judge_score",
    "avg_effective_generation_score",
    "avg_fact_recall",
    "avg_faithfulness",
    "exact_pass_rate",
    "verification_pass_rate",
    "forbidden_violation_rate",
    "hallucination_rate",
    "avg_latency",
    "p95_latency",
    "api_failure_rate",
    "api_timeout_rate",
    "api_completion_rate",
    "queue_delay_avg",
    "execution_duration_avg",
    "chunks_created",
    "vectors_upserted",
    "avg_chunk_length",
    "page_coverage_ratio",
    "source_map_cases",
    "direct_evidence_total",
]


def _case_digest(payload: dict[str, Any]) -> str:
    compact_cases: list[dict[str, Any]] = []
    for case in payload.get("per_case_results") or []:
        compact_cases.append(
            {
                "case_id": case.get("case_id"),
                "status": case.get("status"),
                "answerability": case.get("answerability"),
                "retrieval_scores": case.get("retrieval_scores") or {},
                "generation_scores": case.get("generation_scores") or {},
                "timings": case.get("timings") or {},
                "warning_count": len(case.get("warnings") or []),
                "retrieved_chunk_count": len(case.get("retrieved_chunks") or []),
            }
        )
    return stable_digest(compact_cases)



def _build_snapshot_entry(run_meta: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_run_row_payload(payload, run_meta)
    baseline = {key: normalized.get(key) for key in BASELINE_FIELDS}
    return {
        "run_id": normalized.get("run_id"),
        "artifact_type": normalized.get("artifact_type"),
        "artifact_version": normalized.get("artifact_version"),
        "summary_path": run_meta.get("summary_path"),
        "artifact_path": run_meta.get("artifact_path"),
        "baseline_fields": baseline,
        "baseline_digest": stable_digest(baseline),
        "case_digest": _case_digest(payload),
    }



def main() -> int:
    parser = argparse.ArgumentParser(description="Export a backward-compatibility baseline for Streamlit run loading.")
    parser.add_argument(
        "--runs-root",
        default=str(FINETUNING_ROOT / "artifacts" / "runs"),
        help="Directory containing run artifacts and summaries.",
    )
    parser.add_argument(
        "--output",
        default=str(FINETUNING_ROOT / "artifacts" / "baselines" / "streamlit_catalog_snapshot.json"),
        help="Where to write the baseline snapshot.",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    summary_paths = list_run_summaries(runs_root)
    artifact_paths = list_run_artifacts(runs_root)
    artifact_by_run_id = {path.stem: str(path) for path in artifact_paths}

    snapshot_runs: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    for path in summary_paths:
        payload = load_run_artifact(path)
        run_id = str(payload.get("run_id") or path.name.replace(".summary.json", ""))
        seen_run_ids.add(run_id)
        run_meta = {
            "run_id": run_id,
            "summary_path": str(path),
            "artifact_path": artifact_by_run_id.get(run_id),
        }
        snapshot_runs.append(_build_snapshot_entry(run_meta, payload))

    for path in artifact_paths:
        run_id = path.stem
        if run_id in seen_run_ids:
            continue
        payload = load_run_artifact(path)
        run_meta = {
            "run_id": run_id,
            "summary_path": None,
            "artifact_path": str(path),
        }
        snapshot_runs.append(_build_snapshot_entry(run_meta, payload))

    snapshot_runs.sort(key=lambda item: str(item.get("baseline_fields", {}).get("datetime") or ""), reverse=True)

    manifest = {
        "snapshot_type": "streamlit_run_catalog_baseline",
        "snapshot_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root),
        "run_count": len(snapshot_runs),
        "runs": snapshot_runs,
    }
    write_json(args.output, manifest)
    print(json.dumps({"output": args.output, "run_count": len(snapshot_runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
