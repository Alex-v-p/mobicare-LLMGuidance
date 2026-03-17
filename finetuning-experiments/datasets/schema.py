from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BenchmarkCase:
    id: str
    question: str
    patient_variables: dict[str, Any]
    gold_passage_id: str
    gold_passage_text: str
    page: int | None
    reference_answer: str
    required_facts: list[str] = field(default_factory=list)
    forbidden_facts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
