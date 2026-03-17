from __future__ import annotations

from .models import ExtractedPassage


QUESTION_TYPE_INSTRUCTIONS = {
    "factual": "Create one direct factual question answerable from the passage alone.",
    "clinical_scenario": (
        "Create one short clinical-style question that remains answerable from the passage alone. "
        "You may frame it as a patient scenario, but do not introduce unsupported facts."
    ),
    "paraphrased_factual": "Create one factual question using different wording than the passage.",
    "slightly_indirect": (
        "Create one slightly indirect but still clear question that requires understanding the passage, "
        "not copying it."
    ),
}


def build_generation_prompt(passage: ExtractedPassage, question_type: str) -> str:
    instruction = QUESTION_TYPE_INSTRUCTIONS[question_type]
    return f"""
Generate one benchmark case from the passage below.

Question type: {question_type}
Instruction: {instruction}

Return a JSON object with exactly these keys:
- question: string
- patient_variables: object (empty object if not needed)
- reference_answer: string
- required_facts: array of 1 to 3 short strings
- forbidden_facts: array of 0 to 2 short strings
- tags: array of short strings

Rules:
- Use only the supplied passage.
- Do not cite document ids or page numbers in the question.
- Do not ask about things not stated in the passage.
- The reference answer must be concise.
- required_facts should be semantic ideas, not exact quotes.
- forbidden_facts should only include clearly unsupported or contradictory ideas if relevant; otherwise return an empty array.
- patient_variables should only include variables explicitly present in the question. Avoid fabricating clinical values.
- tags must be short lowercase topic tags.

Passage:
{passage.text}
""".strip()
