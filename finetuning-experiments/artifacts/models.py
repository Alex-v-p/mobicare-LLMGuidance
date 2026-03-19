from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CURRENT_ARTIFACT_VERSION = "2.3"


@dataclass(slots=True)
class RunArtifact:
    artifact_type: str
    artifact_version: str
    run_id: str
    label: str
    datetime: str
    dataset_version: str | None
    documents_version: str | None
    notes: str = ""
    change_note: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    ingestion_summary: dict[str, Any] = field(default_factory=dict)
    source_mapping_summary: dict[str, Any] = field(default_factory=dict)
    retrieval_summary: dict[str, Any] = field(default_factory=dict)
    generation_summary: dict[str, Any] = field(default_factory=dict)
    api_summary: dict[str, Any] = field(default_factory=dict)
    normalized_metrics: dict[str, Any] = field(default_factory=dict)
    per_case_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunSummaryArtifact:
    artifact_type: str
    artifact_version: str
    run_id: str
    label: str
    datetime: str
    dataset_version: str | None
    documents_version: str | None
    notes: str = ""
    change_note: str = ""
    case_count: int = 0
    config_overview: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    normalized_metrics: dict[str, Any] = field(default_factory=dict)
    retrieval_summary: dict[str, Any] = field(default_factory=dict)
    generation_summary: dict[str, Any] = field(default_factory=dict)
    api_summary: dict[str, Any] = field(default_factory=dict)
    ingestion_summary: dict[str, Any] = field(default_factory=dict)
    source_mapping_summary: dict[str, Any] = field(default_factory=dict)
    telemetry_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
