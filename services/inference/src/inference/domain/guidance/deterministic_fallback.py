from __future__ import annotations

from typing import Any

from inference.clinical import ClinicalProfile
from inference.domain.guidance.advice_builder import (
    build_caution_lines,
    build_direct_answer_lines,
    build_general_advice,
    build_rationale_lines,
)
from inference.domain.guidance.answer_normalizer import build_unknown_fallback_answer
from inference.pipeline.support.question_analysis import (
    expected_item_count,
    extract_numbered_items,
    is_explicit_question_only_mode,
    is_literal_question_mode,
    select_relevant_context_sentences,
)
from inference.pipeline.support.specialty import detected_clusters, infer_specialty_focus, synthesize_clinical_state
from shared.contracts.inference import RetrievedContext


def build_deterministic_answer(
    *,
    question: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    retrieved_context: list[RetrievedContext],
    context_assessment: Any,
    prefer_unknown_fallback: bool = False,
) -> str:
    if is_literal_question_mode(question, patient_variables, clinical_profile):
        return build_literal_question_answer(
            question=question,
            retrieved_context=retrieved_context,
            context_assessment=context_assessment,
        )
    if is_explicit_question_only_mode(question, patient_variables, clinical_profile):
        return build_context_question_answer(
            question=question,
            retrieved_context=retrieved_context,
            context_assessment=context_assessment,
        )
    if prefer_unknown_fallback:
        return build_unknown_fallback_answer()

    clusters = detected_clusters(clinical_profile)
    specialty = infer_specialty_focus(patient_variables, clinical_profile, retrieved_context)
    synthesis = synthesize_clinical_state(
        patient_variables=patient_variables,
        clinical_profile=clinical_profile,
        retrieved_context=retrieved_context,
        context_assessment=context_assessment,
        specialty=specialty,
    )
    direct_lines = build_direct_answer_lines(
        clusters=clusters,
        patient_variables=patient_variables,
        retrieved_context=retrieved_context,
        specialty=specialty,
        context_assessment=context_assessment,
        synthesis=synthesis,
    )
    rationale_lines = build_rationale_lines(clusters=clusters, specialty=specialty, synthesis=synthesis)
    caution_lines = build_caution_lines(patient_variables, context_assessment, clinical_profile, specialty)
    general_advice = build_general_advice(context_assessment, clinical_profile, specialty)
    lines = [
        "1. Direct answer",
        *direct_lines,
        "",
        "2. Rationale",
        *rationale_lines,
        "",
        "3. Caution",
        *caution_lines,
        "",
        "4. General advice",
        *general_advice,
    ]
    return "\n".join(lines).strip()


def build_literal_question_answer(
    *,
    question: str,
    retrieved_context: list[RetrievedContext],
    context_assessment: Any,
) -> str:
    combined = " ".join(item.snippet for item in retrieved_context)
    enumerated_items = extract_numbered_items(combined)
    item_count = expected_item_count(question)
    if enumerated_items and item_count:
        direct_lines = [f"- {item}." for item in enumerated_items[:item_count]]
    elif enumerated_items:
        direct_lines = [f"- {item}." for item in enumerated_items[:4]]
    else:
        direct_lines = [f"- {sentence}" for sentence in select_relevant_context_sentences(question, retrieved_context, limit=3)]

    if not direct_lines:
        direct_lines = ["- Unavailable from the retrieved evidence."]

    rationale_lines: list[str] = []
    if enumerated_items:
        rationale_lines.append(
            f"- The retrieved excerpts explicitly enumerate {len(enumerated_items[: item_count or len(enumerated_items)])} items relevant to the question."
        )
    for sentence in select_relevant_context_sentences(question, retrieved_context, limit=2):
        rationale_lines.append(f"- {sentence}")
    if not rationale_lines:
        rationale_lines.append("- The available excerpts do not provide enough detail for a stronger grounded answer.")

    caution_lines: list[str] = []
    if not getattr(context_assessment, "sufficient", False):
        caution_lines.append("- I cannot give a fuller answer with confidence because the retrieved excerpts appear partial or incomplete.")
    if combined and combined.rstrip()[-12:].lower().endswith("memb"):
        caution_lines.append("- One excerpt appears truncated, so the final item wording may be incomplete.")
    if not caution_lines:
        caution_lines.append("- I am relying only on the supplied excerpts and may miss detail that appears in adjacent text.")

    general_advice = [
        "- If exact wording matters, retrieve the adjacent excerpt or the full table entry before making the answer more specific.",
    ]
    lines = [
        "1. Direct answer",
        *direct_lines[: max(item_count or 0, 4) or 4],
        "",
        "2. Rationale",
        *rationale_lines[:3],
        "",
        "3. Caution",
        *caution_lines[:2],
        "",
        "4. General advice",
        *general_advice,
    ]
    return "\n".join(lines).strip()


def build_context_question_answer(
    *,
    question: str,
    retrieved_context: list[RetrievedContext],
    context_assessment: Any,
) -> str:
    sentence_matches = select_relevant_context_sentences(question, retrieved_context, limit=3)
    combined = " ".join(item.snippet for item in retrieved_context)
    enumerated_items = extract_numbered_items(combined)

    if enumerated_items:
        direct_lines = [f"- {item}." for item in enumerated_items[:4]]
    elif sentence_matches:
        direct_lines = [f"- {sentence.rstrip('.')} .".replace('  ', ' ') for sentence in sentence_matches]
    else:
        direct_lines = ["- Unavailable from the retrieved evidence."]

    rationale_lines = [f"- {sentence}" for sentence in sentence_matches[:2]]
    if not rationale_lines:
        rationale_lines.append("- The available excerpts do not provide enough detail for a stronger grounded answer.")

    caution_lines: list[str] = []
    if not getattr(context_assessment, "sufficient", False):
        caution_lines.append("- I cannot give a fuller answer with confidence because the retrieved excerpts appear partial or incomplete.")
    if not caution_lines:
        caution_lines.append("- I am relying only on the supplied excerpts and may miss detail that appears in adjacent text.")

    general_advice = [
        "- Retrieve adjacent text or the full table entry before making the answer more specific or treatment-prescriptive.",
    ]
    lines = [
        "1. Direct answer",
        *direct_lines[:4],
        "",
        "2. Rationale",
        *rationale_lines[:3],
        "",
        "3. Caution",
        *caution_lines[:2],
        "",
        "4. General advice",
        *general_advice,
    ]
    return "\n".join(lines).strip()
