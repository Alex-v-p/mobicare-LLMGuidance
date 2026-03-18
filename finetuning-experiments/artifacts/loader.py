from __future__ import annotations

from pathlib import Path
from typing import Any

from artifacts.migration import migrate_artifact
from utils.json import read_json



def list_run_artifacts(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    return sorted((path for path in root.glob("*.json") if not path.name.endswith(".summary.json")), reverse=True)



def list_run_summaries(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    return sorted(root.glob("*.summary.json"), reverse=True)



def load_run_artifact(path: str | Path) -> dict[str, Any]:
    return migrate_artifact(read_json(path))
