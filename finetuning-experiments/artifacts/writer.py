from __future__ import annotations

from pathlib import Path
from typing import Any

from artifacts.summaries import write_run_summary
from utils.json import write_json



def write_run_artifact(output_dir: str, run_id: str, payload: dict[str, Any]) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = write_json(root / f"{run_id}.json", payload)
    write_run_summary(root, run_id, payload)
    return path
