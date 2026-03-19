from __future__ import annotations

import argparse
import logging

from runners.campaign_runner import run_campaign



def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a campaign sweep of benchmark experiments.")
    parser.add_argument("--config", required=True, help="Path to the campaign config JSON file.")
    parser.add_argument("--verbose", action="store_true")
    return parser



def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    print(run_campaign(args.config))


if __name__ == "__main__":
    main()
