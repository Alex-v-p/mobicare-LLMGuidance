from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from adapters.guidance_payloads import normalize_guidance_record
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
    normalized = normalize_guidance_record(guidance_record)
    generated_answer = str(normalized.get("answer") or "")
    metadata = dict(normalized.get("metadata") or {})
    metadata.setdefault("response_shape", normalized.get("endpoint_envelope", {}).get("response_shape"))
    diagnostics = dict(normalized.get("diagnostics") or {})
    if diagnostics:
        metadata.setdefault("diagnostics", diagnostics)
    return GenerationStageResult(
        generated_answer=generated_answer,
        generation_scores=score_generation(case.to_dict(), generated_answer, retrieved_chunks),
        verification=normalized.get("verification"),
        warnings=list(normalized.get("warnings") or []),
        metadata=metadata,
    )
