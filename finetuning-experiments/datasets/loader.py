from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import BenchmarkCase


def load_benchmark_dataset(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_cases_from_jsonl(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(BenchmarkCase(**json.loads(line)))
    return cases
