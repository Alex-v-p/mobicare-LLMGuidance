from __future__ import annotations

import argparse
import logging
from typing import Any

from cli.rendering import compact_number, emit_json, format_kv_block, format_table, load_full_artifact, load_summary_or_artifact

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a benchmark run artifact or summary.")
    parser.add_argument("target", help="Run ID, artifact path, or summary path.")
    parser.add_argument("--runs-dir", default="./artifacts/runs", help="Directory containing run artifacts.")
    parser.add_argument("--case-id", default=None, help="Optional case id to inspect in detail.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--full", action="store_true", help="Load the full artifact instead of summary output.")
    parser.add_argument("--verbose", action="store_true")
    return parser



def _find_case(artifact: dict[str, Any], case_id: str) -> dict[str, Any] | None:
    for item in artifact.get("per_case_results") or []:
        if item.get("case_id") == case_id:
            return item
    return None



def _render_summary_text(path: str, payload: dict[str, Any], source_kind: str) -> str:
    config_overview = payload.get("config_overview") or {}
    ingestion = config_overview.get("ingestion") or {}
    inference = config_overview.get("inference") or {}
    lines = [
        format_kv_block(
            "Run",
            [
                ("source", path),
                ("source_kind", source_kind),
                ("run_id", payload.get("run_id")),
                ("label", payload.get("label")),
                ("datetime", payload.get("datetime")),
                ("dataset_version", payload.get("dataset_version")),
                ("documents_version", payload.get("documents_version")),
                ("case_count", payload.get("case_count")),
            ],
        ),
        "",
        format_kv_block(
            "Config overview",
            [
                ("cleaning_strategy", ingestion.get("cleaning_strategy")),
                ("chunking_strategy", ingestion.get("chunking_strategy")),
                ("top_k", inference.get("top_k")),
                ("retrieval_mode", inference.get("retrieval_mode")),
                ("llm_model", inference.get("llm_model")),
                ("prompt_label", inference.get("prompt_engineering_label")),
                ("verification", inference.get("enable_response_verification")),
                ("regeneration", inference.get("enable_regeneration")),
            ],
        ),
        "",
    ]

    normalized = payload.get("normalized_metrics") or {}
    if normalized:
        key_metrics = sorted(normalized.items())
        rows = [[key, compact_number(value)] for key, value in key_metrics]
        lines.extend(["Normalized metrics", format_table(["metric", "value"], rows), ""])

    retrieval = payload.get("retrieval_summary") or {}
    generation = payload.get("generation_summary") or {}
    telemetry = payload.get("telemetry_summary") or {}
    lines.append(
        format_kv_block(
            "Summaries",
            [
                ("retrieval.strict_success_rate", compact_number(retrieval.get("strict_success_rate"))),
                ("retrieval.hit_at_3", compact_number(retrieval.get("hit_at_3"))),
                ("generation.avg_answer_similarity", compact_number(generation.get("average_answer_similarity"))),
                ("generation.exact_pass_rate", compact_number(generation.get("exact_pass_rate"))),
                ("latency.p95_ms", compact_number((telemetry.get("success_only") or {}).get("p95"))),
                ("latency.avg_ms", compact_number((telemetry.get("success_only") or {}).get("average"))),
            ],
        )
    )
    return "\n".join(lines)



def _render_case_text(case: dict[str, Any]) -> str:
    retrieval_scores = case.get("retrieval_scores") or {}
    answer_scores = case.get("answer_scores") or {}
    telemetry = case.get("telemetry") or {}
    stages = telemetry.get("stages") or []
    stage_rows = [
        [stage.get("name"), stage.get("status"), compact_number(stage.get("duration_ms"))]
        for stage in stages
    ]
    sections = [
        format_kv_block(
            "Case",
            [
                ("case_id", case.get("case_id")),
                ("status", case.get("status")),
                ("question", case.get("question")),
                ("generated_answer", case.get("generated_answer")),
            ],
        ),
        "",
        format_kv_block(
            "Retrieval scores",
            [(key, compact_number(value)) for key, value in sorted(retrieval_scores.items())],
        ),
        "",
        format_kv_block(
            "Answer scores",
            [(key, compact_number(value)) for key, value in sorted(answer_scores.items())],
        ),
    ]
    if stage_rows:
        sections.extend(["", "Telemetry stages", format_table(["stage", "status", "duration_ms"], stage_rows)])
    return "\n".join(sections)



def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.full or args.case_id:
        path, artifact = load_full_artifact(args.target, runs_dir=args.runs_dir)
        if args.case_id:
            case = _find_case(artifact, args.case_id)
            if not case:
                raise ValueError(f"Case not found in run artifact: {args.case_id}")
            if args.format == "json":
                emit_json(case)
            else:
                print(_render_case_text(case))
            return
        if args.format == "json":
            emit_json(artifact)
        else:
            print(_render_summary_text(str(path), artifact, "artifact"))
        return

    path, payload, source_kind = load_summary_or_artifact(args.target, runs_dir=args.runs_dir)
    if args.format == "json":
        emit_json(payload)
    else:
        print(_render_summary_text(str(path), payload, source_kind))


if __name__ == "__main__":
    main()
