from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from artifacts.loader import load_run_artifact
from artifacts.summaries import build_run_summary
from utils.json import read_json


def emit_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def format_kv_block(title: str, items: Iterable[tuple[str, Any]]) -> str:
    lines = [title]
    for key, value in items:
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    string_rows = [["" if value is None else str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def _render_row(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    divider = "-+-".join("-" * width for width in widths)
    lines = [_render_row(headers), divider]
    lines.extend(_render_row(row) for row in string_rows)
    return "\n".join(lines)


def resolve_artifact_path(target: str | Path, runs_dir: str | Path = "./artifacts/runs") -> Path:
    candidate = Path(target)
    if candidate.exists():
        return candidate.resolve()

    root = Path(runs_dir)
    preferred = [
        root / f"{target}.summary.json",
        root / f"{target}.json",
        root / target,
    ]
    for path in preferred:
        if path.exists():
            return path.resolve()

    matches = sorted(root.glob(f"*{target}*.summary.json")) + sorted(root.glob(f"*{target}*.json"))
    matches = [path for path in matches if path.is_file()]
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(f"Could not resolve run artifact or summary for target: {target}")
    raise FileNotFoundError(
        "Multiple matching artifacts found for target "
        f"{target}: {', '.join(path.name for path in matches[:8])}"
    )


def load_summary_or_artifact(target: str | Path, runs_dir: str | Path = "./artifacts/runs") -> tuple[Path, dict[str, Any], str]:
    path = resolve_artifact_path(target, runs_dir=runs_dir)
    payload = read_json(path)
    artifact_type = payload.get("artifact_type")
    if artifact_type == "run_summary" or str(path).endswith(".summary.json"):
        return path, payload, "summary"
    if artifact_type == "run":
        return path, build_run_summary(load_run_artifact(path)), "artifact"
    return path, payload, artifact_type or "unknown"


def load_full_artifact(target: str | Path, runs_dir: str | Path = "./artifacts/runs") -> tuple[Path, dict[str, Any]]:
    path = resolve_artifact_path(target, runs_dir=runs_dir)
    if str(path).endswith(".summary.json"):
        path = path.with_name(path.name.replace(".summary.json", ".json"))
    return path, load_run_artifact(path)


def compact_number(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)
