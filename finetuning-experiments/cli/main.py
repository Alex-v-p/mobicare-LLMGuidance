from __future__ import annotations

import argparse

from cli.commands.build_source_maps import main as build_source_maps_main
from cli.commands.compare_runs import main as compare_runs_main
from cli.commands.inspect_run import main as inspect_run_main
from cli.commands.run_benchmark import main as run_benchmark_main
from cli.commands.run_campaign import main as run_campaign_main
from dataset_generation.cli import main as generate_dataset_main



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiments-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate-dataset", help="Generate a benchmark dataset from a PDF document.")
    subparsers.add_parser("build-source-maps", help="Build derived source lists for a chunking strategy.")
    subparsers.add_parser("run-benchmark", help="Run an end-to-end benchmark experiment.")
    subparsers.add_parser("run-campaign", help="Run a matrix campaign of benchmark experiments.")
    subparsers.add_parser("inspect-run", help="Inspect a run artifact or summary.")
    subparsers.add_parser("compare-runs", help="Compare two or more benchmark runs.")
    return parser



def main() -> None:
    parser = build_parser()
    args, remainder = parser.parse_known_args()
    if args.command == "generate-dataset":
        generate_dataset_main(remainder)
    elif args.command == "build-source-maps":
        build_source_maps_main(remainder)
    elif args.command == "run-benchmark":
        run_benchmark_main(remainder)
    elif args.command == "run-campaign":
        run_campaign_main(remainder)
    elif args.command == "inspect-run":
        inspect_run_main(remainder)
    elif args.command == "compare-runs":
        compare_runs_main(remainder)


if __name__ == "__main__":
    main()
