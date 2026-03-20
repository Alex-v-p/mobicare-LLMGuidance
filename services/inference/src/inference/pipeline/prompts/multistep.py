from __future__ import annotations

from typing import Any

from inference.clinical import ClinicalProfile
from shared.contracts.inference import RetrievedContext


def _patient_lines(patient_variables: dict[str, Any]) -> list[str]:
    lines = [f"- {key}: {value}" for key, value in sorted(patient_variables.items()) if value is not None]
    return lines or ["- No patient variables were provided."]


def _profile_lines(clinical_profile: ClinicalProfile) -> list[str]:
    lines: list[str] = []
    if clinical_profile.abnormal_variables:
        lines.append("Abnormal / clinically relevant findings:")
        lines.extend(f"- {finding.summary}" for finding in clinical_profile.abnormal_variables)
    elif clinical_profile.recognized_variables:
        lines.append("Recognized variables:")
        lines.extend(f"- {finding.summary}" for finding in clinical_profile.recognized_variables[:5])
    if clinical_profile.unknown_variables:
        lines.append("Variables without configured range interpretation:")
        lines.extend(f"- {key}" for key in clinical_profile.unknown_variables[:10])
    return lines or ["- No interpreted clinical findings were derived from the patient variables."]


def _context_lines(retrieved_context: list[RetrievedContext]) -> list[str]:
    lines = [f"[{item.source_id}] {item.title}: {item.snippet}" for item in retrieved_context]
    return lines or ["No retrieval context was provided."]


def build_query_rewrite_prompt(question: str, patient_variables: dict[str, Any]) -> str:
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
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    retrieved_context: list[RetrievedContext],
    rewritten_query: str | None = None,
    verification_feedback: list[str] | None = None,
    attempt_number: int = 1,
    allow_general_guidance: bool = True,
    context_assessment: Any | None = None,
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
        rewrite_block = f"\nRetrieval-optimized interpretation of the task:\n{rewritten_query}\n"

    assessment_block = ""
    if context_assessment is not None:
        assessment_block = (
            "\nContext sufficiency assessment:\n"
            f"- sufficient: {context_assessment.sufficient}\n"
            f"- confidence: {context_assessment.confidence}\n"
            f"- reasons: {', '.join(context_assessment.reasons) if context_assessment.reasons else 'none'}\n"
        )

    general_guidance_instruction = (
        "Provide a second section with concise, general guidance that is still grounded in the retrieved documents, "
        "but does not over-claim patient-specific precision."
        if allow_general_guidance
        else "If a second general-guidance section would not be grounded, explicitly say that it is unavailable."
    )

    return f"""
        You are a clinical guidance prototype for internal testing only.
        You are NOT making a final medical decision.
        Use the provided retrieved context and patient variables.
        Be cautious, concise, and explicit about uncertainty.
        Do not invent values that are missing.
        If the retrieval context is weak, insufficient, or absent, say so clearly.
        Do not pretend to know recommendations that are not supported by the retrieved context.
        This is generation attempt #{attempt_number}.
        {rewrite_block}{assessment_block}{retry_block}
        User task:
        {question}

        Patient variables:
        {chr(10).join(_patient_lines(patient_variables))}

        Clinical interpretation of patient variables:
        {chr(10).join(_profile_lines(clinical_profile))}

        Retrieved context:
        {chr(10).join(_context_lines(retrieved_context))}

        Respond in plain text with exactly these section headings:
        1. Evidence-based recommendation
        2. Document-grounded general guidance
        3. Uncertainty and missing data

        Section rules:
        - In section 1, answer the task directly, but ONLY to the degree supported by the retrieved context.
        - In section 1, when no explicit question was supplied, infer the likely task from the patient variables and state that you are doing so.
        - In section 2, {general_guidance_instruction}
        - In section 3, list missing patient variables, explain context limitations, and explicitly say "I don't know" when the evidence is insufficient.
        - Never claim a medication, dosage, or intervention is recommended unless it is supported by the retrieved context.
        """.strip()


def build_verification_prompt(
    *,
    question: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    retrieved_context: list[RetrievedContext],
    answer: str,
) -> str:
    return f"""
        You are verifying a draft answer for a clinical guidance prototype.
        Check whether the draft is grounded in the supplied task, patient variables, interpreted findings, and retrieved context.
        Do not add new medical advice.

        Verification rules:
        - FAIL if the answer introduces unsupported facts, values, or recommendations.
        - FAIL if the answer ignores important supplied patient variables or retrieved context.
        - FAIL if the answer does not use the required 3-section structure.
        - FAIL if the answer does not clearly communicate uncertainty when evidence is insufficient.
        - PASS only if the answer is cautious and adequately supported.

        Return exactly this format:
        VERDICT: PASS or FAIL
        ISSUES:
        - one short issue or "- none"
        CONFIDENCE: HIGH, MEDIUM, or LOW

        Task:
        {question}

        Patient variables:
        {chr(10).join(_patient_lines(patient_variables))}

        Interpreted findings:
        {chr(10).join(_profile_lines(clinical_profile))}

        Retrieved context:
        {chr(10).join(_context_lines(retrieved_context))}

        Draft answer:
        {answer}
        """.strip()
