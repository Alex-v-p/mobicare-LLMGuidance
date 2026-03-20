from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

StrictEvidenceLabel = Literal["direct_evidence", "partial_direct_evidence"]
SoftEvidenceLabel = Literal["supporting", "tangential", "irrelevant"]
EvidenceLabel = Literal[
    "direct_evidence",
    "partial_direct_evidence",
    "supporting",
    "tangential",
    "irrelevant",
]

STRICT_LABELS: tuple[StrictEvidenceLabel, ...] = ("direct_evidence", "partial_direct_evidence")
SOFT_LABELS: tuple[SoftEvidenceLabel, ...] = ("supporting", "tangential", "irrelevant")
ALL_LABELS: tuple[EvidenceLabel, ...] = STRICT_LABELS + SOFT_LABELS


@dataclass(slots=True)
class SourceEvidenceItem:
    chunk_ids: list[str]
    label: EvidenceLabel
    combined_score: float
    lexical_score: float
    semantic_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "chunk_ids": self.chunk_ids,
            "label": self.label,
            "combined_score": round(self.combined_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "metadata": self.metadata,
        }
        if self.llm_label:
            payload["llm_label"] = self.llm_label
        return payload


@dataclass(slots=True)
class CaseSourceMapping:
    case_id: str
    mapping_label: str
    strategy: str
    source_list: dict[str, list[SourceEvidenceItem]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        serialized = {
            label: [item.to_dict() for item in self.source_list.get(label, [])]
            for label in ALL_LABELS
        }
        strict_matches = serialized["direct_evidence"] + serialized["partial_direct_evidence"]
        return {
            "case_id": self.case_id,
            "mapping_label": self.mapping_label,
            "strategy": self.strategy,
            "source_list": serialized,
            "matches": strict_matches,
            "metadata": self.metadata,
        }


def legacy_matches_to_source_list(matches: list[dict[str, Any]] | None) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {label: [] for label in ALL_LABELS}
    for match in matches or []:
        item = dict(match)
        item.setdefault("label", "direct_evidence")
        buckets[item["label"]].append(item)
    return buckets
