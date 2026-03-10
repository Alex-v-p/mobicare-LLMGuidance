from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceDocument:
    source_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextChunk:
    chunk_id: str
    source_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
