from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BenchmarkCase:
    id: str
    question: str
    question_type: str
    patient_variables: dict[str, Any]
    gold_passage_id: str
    gold_passage_text: str
    gold_passage_normalized: str
    gold_passage_hash: str
    anchor_start_text: str
    anchor_end_text: str
    source_document_id: str
    source_document_name: str
    source_page: int | None
    source_block_index: int | None
    reference_answer: str
    required_facts: list[str] = field(default_factory=list)
    forbidden_facts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    passage_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
