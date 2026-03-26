from __future__ import annotations

import re
from typing import Any

from inference.pipeline.support.specialty import to_float
from shared.contracts.inference import RetrievedContext


def has_actionable_guidance(answer: str) -> bool:
    lowered = answer.lower()
    action_terms = [
        "monitor", "review", "assess", "reassess", "trend", "follow-up", "follow up",
        "evaluate", "check", "prioritize", "seek", "consider", "tolerance", "contributors",
    ]
    return any(term in lowered for term in action_terms)


def is_minimal_unknown_fallback_answer(answer: str) -> bool:
    lowered = re.sub(r"\s+", " ", answer.strip().lower())
    accepted_prefixes = (
        "based on the provided context, i don't know",
        "based on the provided context i don't know",
        "i can't give a grounded answer from the provided context",
        "i cannot give a grounded answer from the provided context",
    )
    return any(lowered.startswith(prefix) for prefix in accepted_prefixes)


def build_unknown_fallback_answer() -> str:
    return "Based on the provided context, I don't know."


def looks_like_generic_clinical_fallback(answer: str) -> bool:
    lowered = answer.strip().lower()
    fallback_markers = (
        "the main value abnormalities point to a clinically relevant pattern",
        "interpreted as a whole rather than marker by marker",
        "use the retrieved guidance to prioritize the most abnormal findings first",
        "review these results together with symptoms",
        "key missing context that could change the recommendation",
    )
    return sum(marker in lowered for marker in fallback_markers) >= 2


def normalize_generated_answer(
    answer: str,
    *,
    retrieved_context: list[RetrievedContext],
    patient_variables: dict[str, Any],
) -> str:
    normalized = answer.strip()
    if not normalized:
        return normalized

    replacements = {
        "Evidence-based recommendation": "Direct answer",
        "Main answer": "Direct answer",
        "Document-grounded general guidance": "General advice",
        "Uncertainty and missing data": "Caution",
    }
    for old, new in replacements.items():
        normalized = re.sub(rf"\b{re.escape(old)}\b", new, normalized, flags=re.IGNORECASE)

    for item in retrieved_context:
        for token in filter(None, {item.source_id, item.title}):
            normalized = re.sub(re.escape(token), "the available evidence", normalized, flags=re.IGNORECASE)

    cleanup_patterns = [
        r"the most relevant (document|source|pdf)[^.\n]*[.\n]",
        r"the pdf says[^.\n]*",
        r"the document says[^.\n]*",
        r"the best (document|source|pdf)[^.\n]*",
        r"based on the retrieved context,?",
        r"retrieved context",
        r"this document provides",
        r"the pdf provides",
        r"the available evidence is the [^.\n]*",
        r"###\s*Direct\s*Answer",
        r"###\s*Rationale",
        r"###\s*Caution",
        r"###\s*General\s*Advice",
    ]
    for pattern in cleanup_patterns:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)

    potassium_value = to_float(patient_variables.get("potassium"))
    if potassium_value is not None and potassium_value >= 5.0:
        normalized = re.sub(r"[^.\n]*hypokalemia[^.\n]*[.\n]?", "", normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()
