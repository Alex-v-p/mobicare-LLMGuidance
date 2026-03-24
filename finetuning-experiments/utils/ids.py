from __future__ import annotations

import re
from datetime import datetime, timezone

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")



def slugify(value: str, *, default: str = "run") -> str:
    normalized = _NON_ALNUM_RE.sub("_", value.lower()).strip("_")
    return normalized or default



def build_run_id(label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{slugify(label)}"
