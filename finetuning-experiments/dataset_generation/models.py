from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExtractedPassage:
    passage_id: str
    document_id: str
    document_name: str
    text: str
    page: int | None = None
    block_index: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
