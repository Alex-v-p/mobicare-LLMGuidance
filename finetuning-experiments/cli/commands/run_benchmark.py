from __future__ import annotations

import argparse
import logging
from pathlib import Path

from artifacts.loader import load_run_artifact
from artifacts.summaries import build_run_summary
from cli.rendering import emit_json, format_kv_block
from configs.loader import load_run_config
from configs.validator import validate_run_config
from runners.benchmark_runner import run_benchmark

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an end-to-end benchmark experiment.")
    parser.add_argument("--config", required=True, help="Path to the benchmark config JSON file.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the config and exit without running.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--verbose", action="store_true")
    return parser



def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    config = load_run_config(args.config)
    validate_run_config(config)

    if args.validate_only:
        payload = {
            "status": "validated",
            "config_path": str(Path(args.config).resolve()),
            "label": config.label,
            "dataset_path": config.dataset_path,
            "output_dir": config.execution.output_dir,
        }
        if args.format == "json":
            emit_json(payload)
        else:
            print(
                format_kv_block(
                    "Benchmark config validated",
                    payload.items(),
                )
            )
        return

    path = run_benchmark(config)
    artifact = load_run_artifact(path)
    summary = build_run_summary(artifact)
    output = {
        "status": "completed",
        "artifact_path": str(path),
        "summary_path": str(path).replace(".json", ".summary.json"),
        "run_id": artifact.get("run_id"),
        "label": artifact.get("label"),
        "normalized_metrics": artifact.get("normalized_metrics") or {},
        "summary": summary,
    }
    if args.format == "json":
        emit_json(output)
    else:
        print(
            "\n".join(
                [
                    format_kv_block(
                        "Benchmark completed",
                        [
                            ("artifact_path", output["artifact_path"]),
                            ("summary_path", output["summary_path"]),
                            ("run_id", output["run_id"]),
                            ("label", output["label"]),
                        ],
                    ),
                    "",
                    format_kv_block(
                        "Top normalized metrics",
                        sorted((artifact.get("normalized_metrics") or {}).items()),
                    ),
                ]
            )
        )


if __name__ == "__main__":
    main()
