from __future__ import annotations

import argparse

from dataset_generation.cli import main as generate_dataset_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiments-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate-dataset", help="Generate a benchmark dataset from a PDF document.")
    return parser


def main() -> None:
    parser = build_parser()
    args, remainder = parser.parse_known_args()
    if args.command == "generate-dataset":
        generate_dataset_main(remainder)


if __name__ == "__main__":
    main()
