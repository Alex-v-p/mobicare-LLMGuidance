from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def list_run_artifacts(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    return sorted(root.glob("*.json"), reverse=True)


def load_run_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
