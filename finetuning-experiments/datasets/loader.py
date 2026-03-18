from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.json import read_json

from .schema import BenchmarkCase
from .validator import validate_benchmark_dataset



def load_benchmark_dataset(path: str | Path) -> dict[str, Any]:
    raw_dataset = read_json(path)
    validate_benchmark_dataset(raw_dataset)
    return raw_dataset



def load_cases_from_jsonl(path: str | Path) -> list[BenchmarkCase]:
    file_path = Path(path)
    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            case = BenchmarkCase(**read_json_string(line))
        except ValueError as exc:
            raise ValueError(f"Invalid JSONL entry in {file_path} at line {line_number}: {exc}") from exc
        cases.append(case)
    return cases



def read_json_string(raw_text: str) -> dict[str, Any]:
    import json
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("entry must be a JSON object")
    return payload
