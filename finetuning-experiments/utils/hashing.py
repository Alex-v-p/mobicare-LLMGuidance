from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hexdigest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fingerprint(value: Any, *, prefix: str | None = None) -> str:
    digest = sha256_hexdigest(stable_json_dumps(value))
    return f"{prefix}:{digest}" if prefix else digest
