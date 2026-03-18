from __future__ import annotations

import argparse
import logging

from configs.loader import load_run_config
from configs.validator import validate_run_config
from runners.benchmark_runner import run_benchmark


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an end-to-end benchmark experiment.")
    parser.add_argument("--config", required=True, help="Path to the benchmark config JSON file.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    config = load_run_config(args.config)
    validate_run_config(config)
    path = run_benchmark(config)
    print(path)


if __name__ == "__main__":
    main()
