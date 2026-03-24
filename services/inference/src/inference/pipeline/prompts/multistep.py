from __future__ import annotations

from typing import Any

from inference.clinical import ClinicalProfile
from inference.pipeline.prompt_formatting import bullet_block, context_lines, patient_lines, profile_lines
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



def build_query_rewrite_prompt(question: str, patient_variables: dict[str, Any], specialty_focus: Any | None = None) -> str:
    specialty_instruction = (
        "If the case is heart-failure-oriented, prefer search wording such as heart failure, HFrEF, congestion, cardio-renal safety, GDMT, hyperkalaemia, or decompensation when justified by the variables."
        if specialty_focus is not None and specialty_focus.is_heart_failure
        else "Prefer specialty-specific terminology only when the variables clearly support it."
    )
    return f"""
        You rewrite the task into a single retrieval-optimized search query.
        Preserve the original clinical intent.
        Prefer concrete clinical terms over vague wording.
        {specialty_instruction}
        Do not answer the task.
        Do not invent patient facts.
        Return only one line starting with: REWRITTEN_QUERY:

        Original task:
        {question}

        Patient variables:
        {bullet_block(patient_lines(patient_variables))}
        """.strip()


def _render_context_assessment(context_assessment: Any | None) -> str:
    if context_assessment is None:
        return ""
    cluster_text = ", ".join(
        f"{name}={count}" for name, count in context_assessment.cluster_coverage.items()
    ) or "none"
    return (
        "\nContext sufficiency assessment:\n"
        f"- sufficient: {context_assessment.sufficient}\n"
        f"- confidence: {context_assessment.confidence}\n"
        f"- reasons: {', '.join(context_assessment.reasons) if context_assessment.reasons else 'none'}\n"
        f"- cluster coverage: {cluster_text}\n"
    )



def _render_synthesis_block(clinical_synthesis: Any | None, actionable_reasoning: list[str] | None) -> str:
    if clinical_synthesis is None and not actionable_reasoning:
        return ""
    lines = ["\nClinical synthesis:"]
    if clinical_synthesis is not None:
        lines.append(f"- headline: {clinical_synthesis.headline}")
        for item in getattr(clinical_synthesis, "interpretation_points", ())[:4]:
            lines.append(f"- interpretation: {item}")
    if actionable_reasoning:
        lines.append("Actionable reasoning priorities:")
        for item in actionable_reasoning[:4]:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _render_rewrite_block(question: str, rewritten_query: str | None) -> str:
    if not rewritten_query or rewritten_query.strip() == question.strip():
        return ""
    return f"\nRetrieval-focused task interpretation:\n{rewritten_query}\n"



def _render_retry_block(verification_feedback: list[str] | None) -> str:
    if not verification_feedback:
        return ""
    return (
        "\nProblems found in the previous draft:\n"
        + "\n".join(f"- {item}" for item in verification_feedback)
        + "\nRevise the answer so these problems are fixed.\n"
    )



def build_generation_prompt(
    *,
    question: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    retrieved_context: list[RetrievedContext],
    clinical_synthesis: Any | None = None,
    actionable_reasoning: list[str] | None = None,
    rewritten_query: str | None = None,
    verification_feedback: list[str] | None = None,
    attempt_number: int = 1,
    allow_general_guidance: bool = True,
    context_assessment: Any | None = None,
    specialty_focus: Any | None = None,
    literal_question_mode: bool = False,
) -> str:
    general_guidance_instruction = (
        "Give short practical guidance that stays grounded in the excerpts and does not over-claim patient-specific precision."
        if allow_general_guidance
        else "If grounded general guidance is not possible, write 'Unavailable from the retrieved evidence.'"
    )

    specialty_block = "\n".join(f"- {item}" for item in (specialty_focus.prompt_priorities if specialty_focus else []))
    literal_mode_instruction = (
        "Answer the literal user question first. If the question asks for a list, count, or named configurations, enumerate them directly from the excerpts instead of drifting into generic clinical commentary."
        if literal_question_mode
        else ""
    )
    section_one_instruction = (
        "- Section 1 must answer the literal question directly. If the question asks for a list or a fixed number of items, enumerate those items explicitly from the excerpts. Do not replace the answer with a generic clinical summary."
        if literal_question_mode
        else "- Section 1 must answer the task directly in 2-4 short bullet points or sentences. Start with a brief clinical interpretation, then give the most relevant next-step priorities."
    )
    section_two_instruction = (
        "- Section 2 must briefly explain how the excerpts support the direct answer. Prefer short grounded sentences over generic clinical fallback phrasing."
        if literal_question_mode
        else "- Section 2 must briefly connect the answer to the interpreted findings and grounded excerpts, using the clinical synthesis as the main reasoning frame rather than merely repeating raw abnormalities."
    )
    section_three_instruction = (
        "- Section 3 must mention truncation, ambiguity, or incomplete excerpts only when they genuinely limit confidence."
        if literal_question_mode
        else '- Section 3 must explain evidence limitations, mention only the most decision-relevant missing details, and say "I don\'t know" when the evidence does not support a stronger conclusion.'
    )
    section_four_instruction = (
        "- Section 4 should stay minimal and practical; for literal excerpt questions, it can simply note whether adjacent context would help."
        if literal_question_mode
        else f"- Section 4: {general_guidance_instruction}"
    )

    return f"""
        You are a clinical guidance prototype for internal testing.
        Use only the supplied patient variables, interpreted findings, and excerpted evidence.
        Give a direct answer grounded in the excerpts.
        Do not mention files, PDFs, documents, sources, excerpts, chunks, pages, or which evidence is 'best'.
        Do not say 'the document says', 'the PDF says', or 'the most relevant source'.
        Synthesize the answer as straightforward guidance.
        Be cautious, concise, and explicit about uncertainty.
        {literal_mode_instruction}
        Do not invent missing values, diagnoses, medication names, dosages, or interventions.
        If evidence is weak or incomplete, say so clearly.
        Acknowledge every abnormal finding cluster that is present in the interpreted findings, even if the evidence is limited for some of them.
        This is generation attempt #{attempt_number}.
        Dominant clinical lens: {specialty_focus.summary if specialty_focus else "General multi-specialty interpretation."}
        Priority rules for this lens:
        {specialty_block or "- Stay direct, grounded, and cautious."}
        {_render_rewrite_block(question, rewritten_query)}{_render_context_assessment(context_assessment)}{_render_synthesis_block(clinical_synthesis, actionable_reasoning)}{_render_retry_block(verification_feedback)}
        User task:
        {question}

        Patient variables:
        {bullet_block(patient_lines(patient_variables))}

        Clinical interpretation:
        {bullet_block(profile_lines(clinical_profile))}

        Grounding excerpts:
        {bullet_block(context_lines(retrieved_context))}

        Respond in plain text with exactly these section headings:
        1. Direct answer
        2. Rationale
        3. Caution
        4. General advice

        Section rules:
        {section_one_instruction}
        - Section 1 should not become a lab-by-lab summary. Keep raw biomarker restatement to Section 2 unless a value directly changes the recommendation.
        - When no explicit question was supplied, infer the likely task from the patient variables and answer it as a prioritization problem: what matters most now, what needs follow-up soon, and what missing context could change the recommendation.
        - Explicitly address each abnormality cluster when evidence supports it; if coverage is weak, say that in the caution section.
        {section_two_instruction}
        {section_three_instruction}
        {section_four_instruction}
        - Never mention any filename, title, page number, or source identifier.
        - Never describe a file or evidence as the 'best' or 'most relevant'.
        - Never introduce a treatment-specific name unless it already appears in the supplied excerpts.
        - If the excerpts do not support a treatment-specific recommendation, say that clearly instead of filling the answer with generic clinical fallback language.
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
        - FAIL if the answer ignores important supplied patient variables or abnormality clusters.
        - FAIL if the answer does not use the required 4-section structure.
        - FAIL if the answer mentions files, PDFs, sources, chunks, pages, or 'best' evidence.
        - FAIL if the answer contradicts obvious patient facts such as elevated potassium being described as hypokalemia.
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
        {bullet_block(patient_lines(patient_variables))}

        Interpreted findings:
        {bullet_block(profile_lines(clinical_profile))}

        Evidence excerpts:
        {bullet_block(context_lines(retrieved_context))}

        Draft answer:
        {answer}
        """.strip()
