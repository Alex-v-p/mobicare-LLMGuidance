from __future__ import annotations

from typing import Any

from inference.clinical import ClinicalProfile
from shared.contracts.inference import RetrievedContext


DISALLOWED_SOURCE_REFERENCES = [
    "document",
    "pdf",
    "source",
    "chunk",
    "page",
    "retrieved context",
    "most relevant document",
]


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
        lines.extend(f"- {finding.summary}" for finding in clinical_profile.recognized_variables[:6])
    if clinical_profile.informational_variables:
        lines.append("Context variables:")
        lines.extend(f"- {finding.label}={finding.value}" for finding in clinical_profile.informational_variables[:8])
    if clinical_profile.unknown_variables:
        lines.append("Variables without configured interpretation:")
        lines.extend(f"- {key}" for key in clinical_profile.unknown_variables[:10])
    return lines or ["- No interpreted clinical findings were derived from the patient variables."]



def _context_lines(retrieved_context: list[RetrievedContext]) -> list[str]:
    lines = [f"Excerpt {index}: {item.snippet}" for index, item in enumerate(retrieved_context, start=1)]
    return lines or ["No retrieval context was provided."]



def build_query_rewrite_prompt(question: str, patient_variables: dict[str, Any]) -> str:
    return f"""
        You rewrite the task into a single retrieval-optimized search query.
        Preserve the original clinical intent.
        Prefer concrete clinical terms over vague wording.
        Do not answer the task.
        Do not invent patient facts.
        Return only one line starting with: REWRITTEN_QUERY:

        Original task:
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
            "\nProblems found in the previous draft:\n"
            + "\n".join(f"- {item}" for item in verification_feedback)
            + "\nRevise the answer so these problems are fixed.\n"
        )

    rewrite_block = ""
    if rewritten_query and rewritten_query.strip() and rewritten_query.strip() != question.strip():
        rewrite_block = f"\nRetrieval-focused task interpretation:\n{rewritten_query}\n"

    assessment_block = ""
    if context_assessment is not None:
        assessment_block = (
            "\nContext sufficiency assessment:\n"
            f"- sufficient: {context_assessment.sufficient}\n"
            f"- confidence: {context_assessment.confidence}\n"
            f"- reasons: {', '.join(context_assessment.reasons) if context_assessment.reasons else 'none'}\n"
        )

    general_guidance_instruction = (
        "Give short practical guidance that stays grounded in the excerpts and does not over-claim patient-specific precision."
        if allow_general_guidance
        else "If grounded general guidance is not possible, write 'Unavailable from the retrieved evidence.'"
    )

    return f"""
        You are a clinical guidance prototype for internal testing.
        Use only the supplied patient variables, interpreted findings, and excerpted evidence.
        Give a direct answer grounded in the excerpts.
        Do not mention files, PDFs, documents, sources, excerpts, chunks, pages, or which evidence is 'best'.
        Do not say 'the document says', 'the PDF says', or 'the most relevant source'.
        Synthesize the answer as straightforward guidance.
        Be cautious, concise, and explicit about uncertainty.
        Do not invent missing values, diagnoses, medication names, dosages, or interventions.
        If evidence is weak or incomplete, say so clearly.
        This is generation attempt #{attempt_number}.
        {rewrite_block}{assessment_block}{retry_block}
        User task:
        {question}

        Patient variables:
        {chr(10).join(_patient_lines(patient_variables))}

        Clinical interpretation:
        {chr(10).join(_profile_lines(clinical_profile))}

        Grounding excerpts:
        {chr(10).join(_context_lines(retrieved_context))}

        Respond in plain text with exactly these section headings:
        1. Main answer
        2. General guidance
        3. Uncertainty and missing data

        Section rules:
        - Section 1 must answer the task directly in 2-4 short bullet points or sentences.
        - When no explicit question was supplied, infer the likely task from the patient variables, but do not mention retrieval or source selection.
        - Explicitly address the most important abnormal variables when evidence supports them.
        - Section 2: {general_guidance_instruction}
        - Section 3 must list missing patient details, explain evidence limitations, and say "I don't know" when the evidence does not support a stronger conclusion.
        - Never mention any filename, title, page number, or source identifier.
        - Never describe a file or evidence as the 'best' or 'most relevant'.
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
        Check whether the draft is grounded in the supplied task, patient variables, interpreted findings, and evidence excerpts.
        Do not add new medical advice.

        Verification rules:
        - FAIL if the answer introduces unsupported facts, values, or recommendations.
        - FAIL if the answer ignores important supplied patient variables or evidence excerpts.
        - FAIL if the answer does not use the required 3-section structure.
        - FAIL if the answer mentions files, PDFs, sources, chunks, pages, or 'best' evidence.
        - FAIL if the answer does not clearly communicate uncertainty when evidence is insufficient.
        - PASS only if the answer is cautious, direct, and adequately supported.

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

        Evidence excerpts:
        {chr(10).join(_context_lines(retrieved_context))}

        Draft answer:
        {answer}
        """.strip()
