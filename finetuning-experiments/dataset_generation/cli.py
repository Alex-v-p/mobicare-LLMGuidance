from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config import DatasetGenerationSettings
from .generator import DatasetGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a benchmark dataset from a PDF document.")
    parser.add_argument("--input-pdf", required=True, help="Input PDF file path")
    parser.add_argument("--output", required=True, help="Output dataset JSON path")
    parser.add_argument("--config", help="Optional JSON config file")
    parser.add_argument("--dataset-size", type=int, help="Number of benchmark cases to generate")
    parser.add_argument("--provider", choices=["ollama", "openai_compatible"], help="LLM provider for dataset generation")
    parser.add_argument("--model", help="Model name for the selected endpoint")
    parser.add_argument("--base-url", help="Base URL for the selected API, e.g. http://localhost:11434 for Ollama")
    parser.add_argument("--api-key", help="API key for OpenAI-compatible endpoints")
    parser.add_argument("--mix", help="JSON string with question type proportions")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--temperature", type=float, help="Generation temperature")
    parser.add_argument("--min-words", type=int, help="Minimum words per extracted passage")
    parser.add_argument("--max-words", type=int, help="Maximum words per extracted passage")
    return parser


def parse_settings(args: argparse.Namespace) -> DatasetGenerationSettings:
    settings = DatasetGenerationSettings.from_json(args.config) if args.config else DatasetGenerationSettings()
    settings.input_path = args.input_pdf
    settings.output_path = args.output

    if args.dataset_size is not None:
        settings.generation.dataset_size = args.dataset_size
    if args.provider:
        settings.generation.provider = args.provider
    if args.model:
        settings.generation.model = args.model
    if args.base_url:
        settings.generation.api_base_url = args.base_url
    if args.api_key:
        settings.generation.api_key = args.api_key
    if args.mix:
        settings.generation.mix = json.loads(args.mix)
    if args.seed is not None:
        settings.generation.seed = args.seed
    if args.temperature is not None:
        settings.generation.temperature = args.temperature
    if args.min_words is not None:
        settings.extraction.min_words = args.min_words
    if args.max_words is not None:
        settings.extraction.max_words = args.max_words

    return settings


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = parse_settings(args)
    generator = DatasetGenerator(settings)
    cases = generator.build_dataset(settings.input_path)
    generator.save_dataset(cases, settings.output_path)
    generator.print_summary(cases)
    print(f"Saved dataset to {settings.output_path}")


if __name__ == "__main__":
    main()
