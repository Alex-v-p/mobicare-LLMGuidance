from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ExtractedPassage:
    passage_id: str
    text: str
    normalized_text: str
    document_id: str
    document_name: str
    page: int
    block_index: int
    section_title: str | None = None
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)
