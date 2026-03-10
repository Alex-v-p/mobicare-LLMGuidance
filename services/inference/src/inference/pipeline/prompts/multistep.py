from __future__ import annotations

from typing import Any, Dict, List

from shared.contracts.inference import RetrievedContext


def _patient_lines(patient_variables: Dict[str, Any]) -> list[str]:
    lines = [f"- {key}: {value}" for key, value in sorted(patient_variables.items()) if value is not None]
    return lines or ["- No patient variables were provided."]


def _context_lines(retrieved_context: List[RetrievedContext]) -> list[str]:
    lines = [f"[{item.source_id}] {item.title}: {item.snippet}" for item in retrieved_context]
    return lines or ["No retrieval context was provided."]


def build_query_rewrite_prompt(question: str, patient_variables: Dict[str, Any]) -> str:
    return f"""
        You rewrite user questions into a single retrieval-optimized search query.
        Preserve the original clinical intent.
        Do not answer the question.
        Do not invent patient facts.
        Return only one line starting with: REWRITTEN_QUERY:

        Original question:
        {question}

        Patient variables:
        {chr(10).join(_patient_lines(patient_variables))}
        """.strip()


def build_generation_prompt(
    *,
    question: str,
    patient_variables: Dict[str, Any],
    retrieved_context: List[RetrievedContext],
    rewritten_query: str | None = None,
    verification_feedback: list[str] | None = None,
    attempt_number: int = 1,
) -> str:
    retry_block = ""
    if verification_feedback:
        retry_block = (
            "\nVerification feedback from the previous draft:\n"
            + "\n".join(f"- {item}" for item in verification_feedback)
            + "\nRevise the answer so these issues are resolved.\n"
        )

    rewrite_block = ""
    if rewritten_query and rewritten_query.strip() and rewritten_query.strip() != question.strip():
        rewrite_block = f"\nRetrieval-optimized interpretation of the question:\n{rewritten_query}\n"

    return f"""
        You are a clinical guidance prototype for internal testing only.
        You are NOT making a final medical decision.
        Use the provided retrieved context and patient variables.
        Be cautious, concise, and explicit about uncertainty.
        Do not invent values that are missing.
        Ground the answer in the supplied data only.
        This is generation attempt #{attempt_number}.
        {rewrite_block}{retry_block}
        User question:
        {question}

        Patient variables:
        {chr(10).join(_patient_lines(patient_variables))}

        Retrieved context:
        {chr(10).join(_context_lines(retrieved_context))}

        Respond in plain text with 3 short parts:
        1. A direct answer.
        2. A short rationale tied to the given variables and retrieved context.
        3. A short caution mentioning missing or risky data if relevant.
        """.strip()


def build_verification_prompt(
    *,
    question: str,
    patient_variables: Dict[str, Any],
    retrieved_context: List[RetrievedContext],
    answer: str,
) -> str:
    return f"""
        You are verifying a draft answer for a clinical guidance prototype.
        Check whether the draft is grounded in the supplied question, patient variables, and retrieved context.
        Do not add new medical advice.

        Verification rules:
        - FAIL if the answer introduces unsupported facts, values, or recommendations.
        - FAIL if the answer ignores important supplied patient variables or retrieved context.
        - FAIL if the answer does not follow the requested 3-part structure.
        - FAIL if the answer is empty, vague, or internally contradictory.
        - PASS only if the answer is cautious and adequately supported.

        Return exactly this format:
        VERDICT: PASS or FAIL
        ISSUES:
        - one short issue or "- none"
        CONFIDENCE: HIGH, MEDIUM, or LOW

        Question:
        {question}

        Patient variables:
        {chr(10).join(_patient_lines(patient_variables))}

        Retrieved context:
        {chr(10).join(_context_lines(retrieved_context))}

        Draft answer:
        {answer}
        """.strip()
