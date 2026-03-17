from __future__ import annotations

from .models import ExtractedPassage


ANSWERABLE_PROMPT = """You are creating a benchmark case for a medical RAG system.
Return ONLY valid JSON with keys:
question, question_type, reasoning_type, difficulty, patient_variables, reference_answer, required_facts, forbidden_facts, query_variants, tags, retrieval_hints, hallucination_metadata.
Rules:
- The question MUST be answerable directly from the passage.
- Keep patient_variables empty unless the requested question_type is clinical-scenario.
- Use short, concrete required_facts.
- query_variants should have 0-2 alternative phrasings.
- retrieval_hints must include key_terms, expected_section, document_scope.
- hallucination_metadata must include risk_level and likely_failure_modes.
- Do not invent facts not present in the passage.
- question_type must equal the requested value.
- reasoning_type should be one of single-hop, definition, comparison, contraindication-check, scenario-application.
- difficulty should be easy, medium, or hard.
Requested question_type: {question_type}
Passage section: {section_title}
Passage:
{passage}
"""

UNANSWERABLE_PROMPT = """You are creating an out-of-scope benchmark case for a medical RAG system that only knows one heart-failure guideline supplement.
Return ONLY valid JSON with keys:
question, question_type, reasoning_type, difficulty, patient_variables, forbidden_facts, query_variants, tags, retrieval_hints, unanswerable_reason, hallucination_metadata.
Rules:
- The question MUST NOT be answerable from the heart-failure guideline supplement.
- Use a medical question from another domain or from a clearly absent heart-failure detail.
- No gold passage exists for this case.
- question_type must equal the requested value.
- reasoning_type should usually be abstention, definition, or single-hop.
- difficulty should be easy, medium, or hard.
- retrieval_hints expected_section must be null.
- hallucination_metadata must include risk_level, likely_failure_modes, unsupported_targets.
Requested question_type: {question_type}
Topic seed: {topic}
"""


def build_answerable_prompt(passage: ExtractedPassage, question_type: str) -> str:
    return ANSWERABLE_PROMPT.format(
        question_type=question_type,
        section_title=passage.section_title or "unknown",
        passage=passage.text,
    )


def build_unanswerable_prompt(question_type: str, topic: str) -> str:
    return UNANSWERABLE_PROMPT.format(question_type=question_type, topic=topic)
