from __future__ import annotations

import argparse
import logging

from cli.rendering import emit_json, format_kv_block
from runners.campaign_runner import run_campaign
from utils.json import read_json

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a campaign sweep of benchmark experiments.")
    parser.add_argument("--config", required=True, help="Path to the campaign config JSON file.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--verbose", action="store_true")
    return parser



def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    path = run_campaign(args.config)
    payload = read_json(path)
    output = {
        "status": "completed",
        "campaign_artifact_path": str(path),
        "label": payload.get("label"),
        "run_count": payload.get("run_count"),
        "failed_run_count": payload.get("failed_run_count"),
        "campaign_summary": payload.get("campaign_summary") or {},
    }
    if args.format == "json":
        emit_json(output)
    else:
        print(
            "\n".join(
                [
                    format_kv_block(
                        "Campaign completed",
                        [
                            ("campaign_artifact_path", output["campaign_artifact_path"]),
                            ("label", output["label"]),
                            ("run_count", output["run_count"]),
                            ("failed_run_count", output["failed_run_count"]),
                        ],
                    ),
                    "",
                    format_kv_block(
                        "Average normalized metrics",
                        sorted(((output.get("campaign_summary") or {}).get("average_normalized_metrics") or {}).items()),
                    ),
                ]
            )
        )


if __name__ == "__main__":
    main()
