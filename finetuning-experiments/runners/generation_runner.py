from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from adapters.guidance_payloads import normalize_guidance_record
from adapters.llm_judge import evaluate_llm_judge
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


def run_generation_stage(
    case: BenchmarkCase,
    guidance_record: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    evaluation_config: Any | None = None,
) -> GenerationStageResult:
    normalized = normalize_guidance_record(guidance_record)
    generated_answer = str(normalized.get("answer") or "")
    metadata = dict(normalized.get("metadata") or {})
    metadata.setdefault("response_shape", normalized.get("endpoint_envelope", {}).get("response_shape"))
    diagnostics = dict(normalized.get("diagnostics") or {})
    if diagnostics:
        metadata.setdefault("diagnostics", diagnostics)

    rubric_config = getattr(evaluation_config, "deterministic_rubric", None) if evaluation_config is not None else None
    generation_scores = score_generation(case.to_dict(), generated_answer, retrieved_chunks, normalized.get("verification"), rubric_config)

    llm_judge_config = getattr(evaluation_config, "llm_judge", None) if evaluation_config is not None else None
    if llm_judge_config is not None and getattr(llm_judge_config, "enabled", False):
        payload = {
            "question": case.question,
            "patient_variables": case.patient_variables,
            "reference_answer": case.reference_answer,
            "required_facts": case.required_facts,
            "forbidden_facts": case.forbidden_facts,
            "retrieved_context": "\n\n".join(
                str(item.get("snippet") or "") for item in retrieved_chunks
            ) if getattr(llm_judge_config, "include_retrieved_context", True) else "",
            "generated_answer": generated_answer,
            "deterministic_rubric": generation_scores.get("deterministic_rubric") or {},
            "verification": normalized.get("verification") or {},
            "evaluation_profile": generation_scores.get("evaluation_profile"),
        }
        try:
            generation_scores["llm_judge"] = evaluate_llm_judge(llm_judge_config, payload)
            generation_scores["llm_judge_score"] = generation_scores["llm_judge"].get("score")
            generation_scores["llm_judge_grade"] = generation_scores["llm_judge"].get("overall_grade")
        except Exception as exc:  # noqa: BLE001
            if getattr(llm_judge_config, "fail_open", True):
                generation_scores["llm_judge"] = {
                    "enabled": True,
                    "error": str(exc),
                    "score": None,
                    "overall_grade": None,
                    "dimensions": {},
                    "strengths": [],
                    "weaknesses": [],
                    "reasoning_summary": "",
                    "model": getattr(llm_judge_config, "model", None),
                    "raw_response": "",
                }
                generation_scores["llm_judge_score"] = None
                generation_scores["llm_judge_grade"] = None
            else:
                raise

    return GenerationStageResult(
        generated_answer=generated_answer,
        generation_scores=generation_scores,
        verification=normalized.get("verification"),
        warnings=list(normalized.get("warnings") or []),
        metadata=metadata,
    )
