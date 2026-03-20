from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "a", "an", "to", "and", "or", "of", "in", "for", "on", "with", "is", "are", "was", "were", "by", "that"}


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


def score_generation(case: dict[str, Any], answer: str, retrieved_context: list[dict[str, Any]]) -> dict[str, Any]:
    answer_norm = _norm(answer)
    reference_norm = _norm(case.get("reference_answer", ""))
    gold_norm = _norm(case.get("gold_passage_text", ""))
    context_norm = _norm(" ".join((item.get("snippet") or "") for item in retrieved_context))

    similarity = SequenceMatcher(a=reference_norm, b=answer_norm).ratio() if reference_norm or answer_norm else 0.0
    required = case.get("required_facts", []) or []
    forbidden = case.get("forbidden_facts", []) or []
    required_hits = sum(1 for fact in required if _fact_hit(answer_norm, fact))
    forbidden_hits = sum(1 for fact in forbidden if _fact_hit(answer_norm, fact))

    answer_tokens = _informative(answer)
    faith_gold = _coverage(answer_tokens, set(_informative(case.get("gold_passage_text", ""))))
    faith_context = _coverage(answer_tokens, set(_informative(context_norm)))
    context_tokens = set(_informative(context_norm))
    unsupported_tokens = sorted(set(answer_tokens) - context_tokens)
    unsupported = max(0, len(unsupported_tokens))
    unsupported_rate = (unsupported / len(set(answer_tokens))) if answer_tokens else 0.0

    return {
        "answer_similarity": similarity,
        "required_fact_recall": (required_hits / len(required)) if required else 1.0,
        "required_fact_hits": required_hits,
        "required_fact_total": len(required),
        "required_fact_misses": max(0, len(required) - required_hits),
        "forbidden_fact_violations": forbidden_hits,
        "forbidden_fact_total": len(forbidden),
        "faithfulness_to_gold_passage": faith_gold,
        "faithfulness_to_retrieved_context": faith_context,
        "hallucination_unsupported_token_count": unsupported,
        "hallucination_rate": unsupported_rate,
        "unsupported_tokens": unsupported_tokens[:25],
        "retrieved_context_chunk_count": len(retrieved_context),
        "exact_pass": similarity >= 0.95 and forbidden_hits == 0 and unsupported == 0,
    }
