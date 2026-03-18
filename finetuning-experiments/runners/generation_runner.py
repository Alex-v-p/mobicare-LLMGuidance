from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from datasets.schema import BenchmarkCase
from scoring.generation import score_generation


@dataclass(slots=True)
class GenerationStageResult:
    generated_answer: str
    generation_scores: dict[str, Any]
    verification: dict[str, Any] | None
    warnings: list[Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_generation_stage(case: BenchmarkCase, guidance_record: dict[str, Any], retrieved_chunks: list[dict[str, Any]]) -> GenerationStageResult:
    generated_answer = str(guidance_record.get("answer") or "")
    return GenerationStageResult(
        generated_answer=generated_answer,
        generation_scores=score_generation(case.to_dict(), generated_answer, retrieved_chunks),
        verification=guidance_record.get("verification"),
        warnings=list(guidance_record.get("warnings") or []),
        metadata=dict(guidance_record.get("metadata") or {}),
    )
