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
    parser = argparse.ArgumentParser(description="Compare two or more benchmark runs.")
    parser.add_argument("targets", nargs="+", help="Run IDs and/or artifact paths to compare.")
    parser.add_argument("--runs-dir", default="./artifacts/runs", help="Directory containing run artifacts.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--case-id", default=None, help="Optional case id for case-level comparison.")
    parser.add_argument("--full", action="store_true", help="Load full artifacts for every target.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Optional explicit normalized metric keys to show.")
    parser.add_argument("--verbose", action="store_true")
    return parser



def _load_entries(targets: list[str], runs_dir: str, full: bool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for target in targets:
        if full:
            path, artifact = load_full_artifact(target, runs_dir=runs_dir)
            summary = {
                "path": str(path),
                "payload": artifact,
                "kind": "artifact",
            }
        else:
            path, payload, kind = load_summary_or_artifact(target, runs_dir=runs_dir)
            summary = {"path": str(path), "payload": payload, "kind": kind}
        entries.append(summary)
    return entries



def _metric_keys(entries: list[dict[str, Any]], requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    keys = sorted(
        {
            key
            for entry in entries
            for key in ((entry["payload"].get("normalized_metrics") or {}).keys())
        }
    )
    return keys



def _find_case(artifact: dict[str, Any], case_id: str) -> dict[str, Any] | None:
    for item in artifact.get("per_case_results") or []:
        if item.get("case_id") == case_id:
            return item
    return None



def _build_case_rows(entries: list[dict[str, Any]], case_id: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for entry in entries:
        case = _find_case(entry["payload"], case_id)
        if not case:
            rows.append([entry["payload"].get("label") or entry["payload"].get("run_id"), "missing", "", "", ""])
            continue
        retrieval = case.get("retrieval_scores") or {}
        answer = case.get("answer_scores") or {}
        rows.append(
            [
                entry["payload"].get("label") or entry["payload"].get("run_id"),
                str(case.get("status") or "completed"),
                compact_number(retrieval.get("strict_success")),
                compact_number(answer.get("answer_similarity")),
                compact_number(answer.get("exact_pass")),
            ]
        )
    return rows



def _render_text(entries: list[dict[str, Any]], metric_keys: list[str], case_id: str | None) -> str:
    baseline = entries[0]["payload"]
    overview_rows = []
    for entry in entries:
        payload = entry["payload"]
        config = payload.get("config_overview") or payload.get("config") or {}
        ingestion = config.get("ingestion") or {}
        inference = config.get("inference") or {}
        overview_rows.append(
            [
                payload.get("label") or payload.get("run_id"),
                payload.get("run_id"),
                ingestion.get("chunking_strategy"),
                inference.get("retrieval_mode"),
                inference.get("llm_model"),
                inference.get("prompt_engineering_label"),
            ]
        )

    lines = [
        format_kv_block(
            "Comparison",
            [
                ("baseline", baseline.get("label") or baseline.get("run_id")),
                ("run_count", len(entries)),
                ("case_level_compare", case_id or "disabled"),
            ],
        ),
        "",
        "Run overview",
        format_table(["label", "run_id", "chunking", "retrieval", "llm_model", "prompt"], overview_rows),
        "",
    ]

    metric_rows = []
    for metric in metric_keys:
        row = [metric]
        base_value = (baseline.get("normalized_metrics") or {}).get(metric)
        for entry in entries:
            value = (entry["payload"].get("normalized_metrics") or {}).get(metric)
            row.append(compact_number(value))
        if len(entries) > 1 and isinstance(base_value, (int, float)):
            for entry in entries[1:]:
                value = (entry["payload"].get("normalized_metrics") or {}).get(metric)
                delta = None if not isinstance(value, (int, float)) else value - base_value
                row.append(compact_number(delta))
        metric_rows.append(row)

    headers = ["metric"] + [entry["payload"].get("label") or entry["payload"].get("run_id") for entry in entries]
    if len(entries) > 1:
        headers += [f"Δ vs {headers[1]} ({entry['payload'].get('label') or entry['payload'].get('run_id')})" for entry in entries[1:]]
    lines.extend(["Normalized metric comparison", format_table(headers, metric_rows)])

    if case_id:
        lines.extend([
            "",
            f"Case comparison: {case_id}",
            format_table(["run", "status", "strict_success", "answer_similarity", "exact_pass"], _build_case_rows(entries, case_id)),
        ])
    return "\n".join(lines)



def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    use_full = args.full or bool(args.case_id)
    entries = _load_entries(args.targets, runs_dir=args.runs_dir, full=use_full)
    metric_keys = _metric_keys(entries, args.metrics)

    if args.format == "json":
        emit_json(
            {
                "runs": [entry["payload"] for entry in entries],
                "metric_keys": metric_keys,
                "case_id": args.case_id,
            }
        )
    else:
        print(_render_text(entries, metric_keys, args.case_id))


if __name__ == "__main__":
    main()
