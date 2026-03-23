from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class LLMJudgeResult:
    score: float
    overall_grade: str
    dimensions: dict[str, Any]
    strengths: list[str]
    weaknesses: list[str]
    reasoning_summary: str
    model: str | None
    raw_response: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "score": self.score,
            "overall_grade": self.overall_grade,
            "dimensions": self.dimensions,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "reasoning_summary": self.reasoning_summary,
            "model": self.model,
            "raw_response": self.raw_response,
            "error": self.error,
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _grade(score: float) -> str:
    if score >= 0.85:
        return "excellent"
    if score >= 0.70:
        return "good"
    if score >= 0.50:
        return "partial"
    return "poor"


def _build_prompt(payload: dict[str, Any]) -> str:
    return f"""You are grading a medical guidance answer for benchmarking.
Return ONLY valid JSON with this shape:
{{
  "correctness": 1-5,
  "completeness": 1-5,
  "groundedness": 1-5,
  "safety": 1-5,
  "usefulness": 1-5,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "reasoning_summary": "max 80 words"
}}

Scoring guidance:
- correctness: factual alignment with the reference answer and required facts
- completeness: how much of the clinically important content is covered
- groundedness: how well supported the answer is by retrieved context; penalize unsupported claims
- safety: penalize dangerous overclaims, contradictions, or omission of clearly important cautions
- usefulness: practical utility and clarity for the asked question

Question:
{payload.get('question','')}

Patient variables:
{json.dumps(payload.get('patient_variables') or {}, ensure_ascii=False, indent=2)}

Reference answer:
{payload.get('reference_answer','')}

Required facts:
{json.dumps(payload.get('required_facts') or [], ensure_ascii=False)}

Forbidden facts:
{json.dumps(payload.get('forbidden_facts') or [], ensure_ascii=False)}

Retrieved context:
{payload.get('retrieved_context','')}

Generated answer:
{payload.get('generated_answer','')}

Deterministic rubric summary:
{json.dumps(payload.get('deterministic_rubric') or {}, ensure_ascii=False, indent=2)}

Verification:
{json.dumps(payload.get('verification') or {}, ensure_ascii=False, indent=2)}
"""


def _parse_json(raw: str) -> dict[str, Any]:
    text = (raw or '').strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


def evaluate_with_ollama(*, base_url: str, model: str, temperature: float, max_tokens: int, timeout_seconds: int, payload: dict[str, Any]) -> LLMJudgeResult:
    prompt = _build_prompt(payload)
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    raw = str(data.get("response") or "")
    parsed = _parse_json(raw)

    dimensions = {}
    for key in ["correctness", "completeness", "groundedness", "safety", "usefulness"]:
        try:
            dimensions[key] = max(1, min(5, int(parsed.get(key, 1))))
        except (TypeError, ValueError):
            dimensions[key] = 1
    score = _clamp(sum(dimensions.values()) / (5 * len(dimensions)))
    return LLMJudgeResult(
        score=score,
        overall_grade=_grade(score),
        dimensions=dimensions,
        strengths=[str(x) for x in (parsed.get("strengths") or [])][:5],
        weaknesses=[str(x) for x in (parsed.get("weaknesses") or [])][:5],
        reasoning_summary=str(parsed.get("reasoning_summary") or "")[:600],
        model=str(data.get("model") or model),
        raw_response=raw,
    )


def evaluate_llm_judge(config: Any, payload: dict[str, Any]) -> dict[str, Any]:
    model = str(config.model or '').strip()
    if not model:
        return {
            "enabled": True,
            "error": "LLM judge enabled but no evaluation.llm_judge.model was configured.",
            "score": 0.0,
            "overall_grade": "poor",
            "dimensions": {},
            "strengths": [],
            "weaknesses": [],
            "reasoning_summary": "",
            "model": None,
            "raw_response": "",
        }
    result = evaluate_with_ollama(
        base_url=config.base_url,
        model=model,
        temperature=float(config.temperature),
        max_tokens=int(config.max_tokens),
        timeout_seconds=int(config.timeout_seconds),
        payload=payload,
    )
    return result.to_dict()
