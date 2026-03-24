from __future__ import annotations

import re
from typing import Any

from datasets.observation import is_observation_only_case

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "to", "and", "or", "of", "in", "for", "on", "with", "is", "are", "was", "were", "by", "that",
    "this", "these", "those", "from", "as", "at", "be", "it", "its", "into", "than", "then", "their", "there",
}


def _norm(text: str | None) -> str:
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def _informative(text: str | None) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 2]


def _fact_hit(answer_norm: str, fact: str) -> bool:
    fact_norm = _norm(fact)
    return bool(fact_norm and fact_norm in answer_norm)


def _coverage(tokens: list[str], haystack: set[str]) -> float:
    unique = set(tokens)
    return len(unique & haystack) / len(unique) if unique else 0.0


def _set_f1(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return (2 * precision * recall) / (precision + recall) if precision + recall else 0.0


def _verification_score(verification: dict[str, Any] | None) -> float:
    payload = verification or {}
    verdict = str(payload.get("verdict") or "").strip().lower()
    confidence = str(payload.get("confidence") or "").strip().lower()
    if verdict in {"pass", "passed", "ok", "success"}:
        return {"high": 1.0, "medium": 0.9, "low": 0.8}.get(confidence, 0.85)
    if verdict in {"partial", "warning", "mixed"}:
        return {"high": 0.65, "medium": 0.55, "low": 0.45}.get(confidence, 0.55)
    if verdict in {"fail", "failed", "error"}:
        return {"high": 0.1, "medium": 0.2, "low": 0.3}.get(confidence, 0.2)
    return 0.5


def _grade_from_score(score: float) -> str:
    if score >= 0.85:
        return "excellent"
    if score >= 0.7:
        return "good"
    if score >= 0.5:
        return "partial"
    return "poor"



def score_generation(
    case: dict[str, Any],
    answer: str,
    retrieved_context: list[dict[str, Any]],
    verification: dict[str, Any] | None = None,
    rubric_config: Any | None = None,
) -> dict[str, Any]:
    answer_norm = _norm(answer)
    context_norm = _norm(" ".join((item.get("snippet") or "") for item in retrieved_context))
    observation_only = is_observation_only_case(case)

    required = case.get("required_facts", []) or []
    forbidden = case.get("forbidden_facts", []) or []
    required_hits = sum(1 for fact in required if _fact_hit(answer_norm, fact))
    forbidden_hits = sum(1 for fact in forbidden if _fact_hit(answer_norm, fact))

    answer_tokens = _informative(answer)
    answer_token_set = set(answer_tokens)
    reference_token_set = set(_informative(case.get("reference_answer", "")))
    gold_token_set = set(_informative(case.get("gold_passage_text", "")))
    context_tokens = set(_informative(context_norm))

    reference_token_f1 = _set_f1(answer_token_set, reference_token_set)
    gold_token_f1 = _set_f1(answer_token_set, gold_token_set)
    context_token_f1 = _set_f1(answer_token_set, context_tokens)

    faith_gold = _coverage(answer_tokens, gold_token_set)
    faith_context = _coverage(answer_tokens, context_tokens)
    unsupported_tokens = sorted(answer_token_set - context_tokens)
    unsupported = max(0, len(unsupported_tokens))
    unsupported_rate = (unsupported / len(answer_token_set)) if answer_token_set else 0.0
    groundedness = max(0.0, 1.0 - unsupported_rate)
    verification_score = _verification_score(verification)
    required_fact_recall = (required_hits / len(required)) if required else 1.0

    weights = {
        "required_fact": 0.35,
        "reference_alignment": 0.15,
        "gold_alignment": 0.10,
        "context_alignment": 0.15,
        "groundedness": 0.15,
        "verification": 0.10,
    }
    if rubric_config is not None:
        weights = {
            "required_fact": float(getattr(rubric_config, "required_fact_weight", weights["required_fact"])),
            "reference_alignment": float(getattr(rubric_config, "reference_alignment_weight", weights["reference_alignment"])),
            "gold_alignment": float(getattr(rubric_config, "gold_alignment_weight", weights["gold_alignment"])),
            "context_alignment": float(getattr(rubric_config, "context_alignment_weight", weights["context_alignment"])),
            "groundedness": float(getattr(rubric_config, "groundedness_weight", weights["groundedness"])),
            "verification": float(getattr(rubric_config, "verification_weight", weights["verification"])),
        }
    weight_total = sum(weights.values()) or 1.0

    deterministic_applicable = bool(not observation_only and getattr(rubric_config, "enabled", True))
    deterministic_rubric_score: float | None = None
    deterministic_grade: str | None = None
    if deterministic_applicable:
        deterministic_rubric_score = (
            weights["required_fact"] * required_fact_recall
            + weights["reference_alignment"] * reference_token_f1
            + weights["gold_alignment"] * gold_token_f1
            + weights["context_alignment"] * max(faith_context, context_token_f1)
            + weights["groundedness"] * groundedness
            + weights["verification"] * verification_score
        ) / weight_total
        if forbidden_hits:
            deterministic_rubric_score = max(0.0, deterministic_rubric_score - min(0.4, 0.2 * forbidden_hits))
        deterministic_rubric_score = max(0.0, min(1.0, deterministic_rubric_score))
        deterministic_grade = _grade_from_score(deterministic_rubric_score)

    deterministic = {
        "enabled": deterministic_applicable,
        "applicable": deterministic_applicable,
        "skipped_reason": None if deterministic_applicable else "observation_only_case",
        "score": deterministic_rubric_score,
        "grade": deterministic_grade,
        "weights": weights,
        "subscores": {
            "required_fact_recall": required_fact_recall,
            "reference_alignment": reference_token_f1,
            "gold_alignment": gold_token_f1,
            "context_alignment": max(faith_context, context_token_f1),
            "groundedness": groundedness,
            "verification": verification_score,
        },
        "required_fact_hits": required_hits,
        "required_fact_total": len(required),
        "required_fact_misses": max(0, len(required) - required_hits),
        "forbidden_fact_violations": forbidden_hits,
        "forbidden_fact_total": len(forbidden),
    }

    return {
        "evaluation_profile": "observation_only" if observation_only else "standard",
        "deterministic_rubric_applicable": deterministic_applicable,
        "answer_similarity": reference_token_f1,
        "answer_similarity_legacy_note": "Token-F1 against the reference answer. Prefer deterministic_rubric.score and llm_judge.score.",
        "reference_token_f1": reference_token_f1,
        "gold_token_f1": gold_token_f1,
        "context_token_f1": context_token_f1,
        "deterministic_rubric": deterministic,
        "answer_quality_score": deterministic_rubric_score,
        "answer_quality_grade": deterministic_grade,
        "judge_score": deterministic_rubric_score,
        "judge_grade": deterministic_grade,
        "verification_score": verification_score,
        "required_fact_recall": required_fact_recall,
        "required_fact_hits": required_hits,
        "required_fact_total": len(required),
        "required_fact_misses": max(0, len(required) - required_hits),
        "forbidden_fact_violations": forbidden_hits,
        "forbidden_fact_total": len(forbidden),
        "faithfulness_to_gold_passage": faith_gold,
        "faithfulness_to_retrieved_context": faith_context,
        "groundedness_score": groundedness,
        "hallucination_unsupported_token_count": unsupported,
        "hallucination_rate": unsupported_rate,
        "unsupported_tokens": unsupported_tokens[:25],
        "retrieved_context_chunk_count": len(retrieved_context),
        "exact_pass": bool(deterministic_applicable and required_fact_recall >= 0.999 and forbidden_hits == 0 and unsupported == 0),
    }
