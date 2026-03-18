from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_artifact(output_dir: str, run_id: str, payload: dict[str, Any]) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
