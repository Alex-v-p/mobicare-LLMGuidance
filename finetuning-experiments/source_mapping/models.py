from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RankedChunkMatch:
    chunk_ids: list[str]
    combined_score: float
    lexical_score: float
    semantic_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_ids": self.chunk_ids,
            "combined_score": round(self.combined_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CaseChunkAssignment:
    case_id: str
    mapping_label: str
    matches: list[RankedChunkMatch]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mapping_label": self.mapping_label,
            "matches": [match.to_dict() for match in self.matches],
        }
