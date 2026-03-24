from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_settings
from .generator import DatasetGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    settings = load_settings(args.config, args.input_pdf, args.output, args.output_jsonl)
    generator = DatasetGenerator(settings)
    cases = generator.build_dataset(Path(settings.input_path))
    generator.save_dataset(cases, settings.output_path, settings.output_jsonl_path)
    print(f"Generated {len(cases)} cases")
    print(f"JSON: {settings.output_path}")
    print(f"JSONL: {settings.output_jsonl_path or str(Path(settings.output_path).with_suffix('.jsonl'))}")


if __name__ == "__main__":
    main()
