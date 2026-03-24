from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BenchmarkCase:
    id: str
    question: str
    reference_answer: str
    dataset_version: str = "unknown"
    question_type: str = "unknown"
    reasoning_type: str = "unknown"
    difficulty: str = "unknown"
    answerability: str = "answerable"
    expected_behavior: str = "answer_question"
    expected_abstention_style: str | None = None
    case_weight: float = 1.0
    review_status: str = "approved"
    patient_variables: dict[str, Any] = field(default_factory=dict)
    gold_passage_id: str | None = None
    gold_passage_text: str | None = None
    gold_passage_normalized: str | None = None
    gold_passage_hash: str | None = None
    anchor_start_text: str | None = None
    anchor_end_text: str | None = None
    source_document_id: str | None = None
    source_document_name: str | None = None
    source_page: int | None = None
    source_block_index: int | None = None
    required_facts: list[str] = field(default_factory=list)
    forbidden_facts: list[str] = field(default_factory=list)
    query_variants: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    retrieval_hints: dict[str, Any] = field(default_factory=dict)
    unanswerable_reason: str | None = None
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    passage_metadata: dict[str, Any] = field(default_factory=dict)
    hallucination_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset_version": self.dataset_version,
            "question": self.question,
            "question_type": self.question_type,
            "reasoning_type": self.reasoning_type,
            "difficulty": self.difficulty,
            "answerability": self.answerability,
            "expected_behavior": self.expected_behavior,
            "expected_abstention_style": self.expected_abstention_style,
            "case_weight": self.case_weight,
            "review_status": self.review_status,
            "patient_variables": self.patient_variables,
            "reference_answer": self.reference_answer,
            "required_facts": self.required_facts,
            "forbidden_facts": self.forbidden_facts,
            "query_variants": self.query_variants,
            "tags": self.tags,
            "retrieval_hints": self.retrieval_hints,
            "unanswerable_reason": self.unanswerable_reason,
            "gold_passage_id": self.gold_passage_id,
            "gold_passage_text": self.gold_passage_text,
            "gold_passage_normalized": self.gold_passage_normalized,
            "gold_passage_hash": self.gold_passage_hash,
            "anchor_start_text": self.anchor_start_text,
            "anchor_end_text": self.anchor_end_text,
            "source_document_id": self.source_document_id,
            "source_document_name": self.source_document_name,
            "source_page": self.source_page,
            "source_block_index": self.source_block_index,
            "generation_metadata": self.generation_metadata,
            "passage_metadata": self.passage_metadata,
            "hallucination_metadata": self.hallucination_metadata,
        }
