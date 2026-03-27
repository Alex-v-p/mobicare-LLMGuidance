from __future__ import annotations

import re
from typing import Any

from inference.clinical import ClinicalProfile
from shared.contracts.inference import RetrievedContext

from inference.domain.guidance.constants import (
    _GENERIC_NON_ANSWER_PHRASES,
    _QUESTION_LITERAL_TERMS,
    _STOPWORDS,
)

def extract_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]{3,}", text.lower()) if term not in _STOPWORDS}


def context_key(item: RetrievedContext) -> tuple[str, str | None, str]:
    return (item.source_id, item.chunk_id, item.snippet)


def is_literal_question_mode(
    question: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile | None = None,
) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False

    profile = clinical_profile
    has_patient_context = bool(patient_variables) or bool(profile and (profile.recognized_variables or profile.abnormal_variables))
    patient_language = any(
        term in normalized
        for term in {
            "patient", "symptom", "symptoms", "result", "results", "value", "values", "profile",
            "management", "follow-up", "follow up", "monitor", "monitoring", "safety", "treat",
            "treatment", "prescribe", "medication", "dose", "escalation",
        }
    )
    if not has_patient_context and not patient_language:
        return True
    return any(term in normalized for term in _QUESTION_LITERAL_TERMS) and not patient_language


def is_explicit_question_only_mode(
    question: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile | None = None,
) -> bool:
    normalized = question.strip()
    if not normalized:
        return False

    profile = clinical_profile
    has_patient_context = bool(patient_variables) or bool(profile and (profile.recognized_variables or profile.abnormal_variables))
    return not has_patient_context


def _question_focus_terms(question: str, retrieved_context: list[RetrievedContext]) -> set[str]:
    terms = extract_terms(question)
    combined = " ".join(f"{item.title} {item.snippet}" for item in retrieved_context).lower()
    return {term for term in terms if term in combined and len(term) > 3}


def expected_item_count(question: str) -> int | None:
    lowered = question.lower()
    digit_match = re.search(r"\b([2-9]|10)\b", lowered)
    if digit_match:
        return int(digit_match.group(1))
    for word, value in {
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }.items():
        if re.search(rf"\b{word}\b", lowered):
            return value
    return None


def _expected_item_count(question: str) -> int | None:
    return expected_item_count(question)


def extract_numbered_items(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<!\d)\b([1-9])\.\s*", collapsed)
    items: list[str] = []
    for index in range(1, len(parts), 2):
        if index + 1 >= len(parts):
            break
        cleaned = re.sub(r"\s+", " ", parts[index + 1]).strip(" ;,.-")
        if cleaned:
            items.append(cleaned)
    return items


def select_relevant_context_sentences(question: str, retrieved_context: list[RetrievedContext], *, limit: int = 3) -> list[str]:
    focus_terms = _question_focus_terms(question, retrieved_context) or extract_terms(question)
    scored: list[tuple[int, str]] = []
    for item in retrieved_context:
        for sentence in re.split(r"(?<=[.!?;])\s+", item.snippet):
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if not cleaned:
                continue
            sentence_terms = extract_terms(cleaned)
            overlap = len(focus_terms & sentence_terms)
            scored.append((overlap, cleaned))
    scored.sort(key=lambda entry: (entry[0], len(entry[1])), reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for overlap, sentence in scored:
        if sentence in seen:
            continue
        if overlap <= 0 and selected:
            continue
        selected.append(sentence)
        seen.add(sentence)
        if len(selected) >= limit:
            break
    return selected


def answer_addresses_literal_question(answer: str, question: str, retrieved_context: list[RetrievedContext]) -> bool:
    direct_block = answer.lower().split("2. rationale", 1)[0]
    if any(phrase in direct_block for phrase in _GENERIC_NON_ANSWER_PHRASES):
        return False

    expected_count = expected_item_count(question)
    enumerated_items = extract_numbered_items(" ".join(item.snippet for item in retrieved_context))
    direct_lines = [
        line.strip()
        for line in direct_block.splitlines()
        if line.strip().startswith(("-", "1.", "2.", "3.", "4.", "5."))
    ]
    if expected_count and enumerated_items and len(direct_lines) < min(expected_count, 3):
        return False

    focus_terms = _question_focus_terms(question, retrieved_context)
    if sum(1 for term in focus_terms if term in direct_block) >= 2:
        return True

    anchor_terms = set()
    for item in enumerated_items[: max(expected_count or 0, 3) or 3]:
        item_terms = [term for term in extract_terms(item) if len(term) > 4]
        anchor_terms.update(item_terms[:2])
    if anchor_terms and any(term in direct_block for term in anchor_terms):
        return True

    sentence_matches = select_relevant_context_sentences(question, retrieved_context, limit=2)
    return any(len(extract_terms(sentence) & extract_terms(direct_block)) >= 3 for sentence in sentence_matches)


def answer_addresses_explicit_question(answer: str, question: str, retrieved_context: list[RetrievedContext]) -> bool:
    direct_block = answer.lower().split("2. rationale", 1)[0]
    if any(phrase in direct_block for phrase in _GENERIC_NON_ANSWER_PHRASES):
        return False

    focus_terms = _question_focus_terms(question, retrieved_context) or extract_terms(question)
    focus_terms = {term for term in focus_terms if len(term) > 3}
    if sum(1 for term in focus_terms if term in direct_block) >= min(2, max(1, len(focus_terms))):
        return True

    sentence_matches = select_relevant_context_sentences(question, retrieved_context, limit=3)
    if any(len(extract_terms(sentence) & extract_terms(direct_block)) >= 3 for sentence in sentence_matches):
        return True

    enumerated_items = extract_numbered_items(" ".join(item.snippet for item in retrieved_context))
    if enumerated_items:
        anchor_terms: set[str] = set()
        for item in enumerated_items[:4]:
            anchor_terms.update(term for term in extract_terms(item) if len(term) > 4)
        if anchor_terms and any(term in direct_block for term in list(anchor_terms)[:6]):
            return True

    return False
