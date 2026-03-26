from __future__ import annotations

from typing import Any

from inference.clinical import ClinicalProfile
from inference.domain.guidance.answer_normalizer import (
    has_actionable_guidance,
    is_minimal_unknown_fallback_answer,
)
from inference.pipeline.prompts.multistep import DISALLOWED_SOURCE_REFERENCES
from inference.pipeline.support.question_analysis import (
    answer_addresses_explicit_question,
    answer_addresses_literal_question,
    is_explicit_question_only_mode,
    is_literal_question_mode,
)
from inference.pipeline.support.specialty import (
    detected_clusters,
    infer_specialty_focus,
    prioritized_clusters,
    to_float,
)
from shared.contracts.inference import RetrievedContext

_CLUSTER_ALIASES = {
    "HF severity and congestion": {"heart failure", "hf", "congestion", "decompensation", "natriuretic"},
    "Cardio-renal and electrolyte safety": {"renal", "kidney", "electrolyte", "potassium", "sodium", "creatinine", "cardio-renal"},
    "Rhythm and conduction": {"rhythm", "conduction", "heart rate", "qrs", "atrial fibrillation"},
    "Anemia and iron status": {"anemia", "iron", "hemoglobin", "ferritin", "transferrin"},
    "Inflammation and injury": {"inflammation", "injury", "crp", "troponin", "inflammatory"},
    "Glycemic and cardiometabolic risk": {"glycemic", "cardiometabolic", "glucose", "hba1c", "diabetes", "metabolic"},
}


def should_force_deterministic_answer(
    *,
    answer: str,
    question: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    retrieved_context: list[RetrievedContext],
    context_assessment: Any,
) -> bool:
    literal_question_mode = is_literal_question_mode(question, patient_variables, clinical_profile)
    explicit_question_only_mode = is_explicit_question_only_mode(question, patient_variables, clinical_profile)
    issues = collect_answer_issues(
        answer=answer,
        question=question,
        patient_variables=patient_variables,
        clinical_profile=clinical_profile,
        retrieved_context=retrieved_context,
    )
    severe_issue_markers = (
        "empty",
        "missing the",
        "too short",
        "document-selection",
        "potassium value",
        "unsupported treatment-specific wording",
        "literal question",
    )
    if any(marker in issue.lower() for issue in issues for marker in severe_issue_markers):
        return True
    if literal_question_mode:
        return not answer_addresses_literal_question(answer, question, retrieved_context)
    if explicit_question_only_mode and not answer_addresses_explicit_question(answer, question, retrieved_context):
        return True
    if not has_actionable_guidance(answer):
        return True
    if not context_assessment.sufficient and len(issues) >= 2:
        return True
    return False


def collect_answer_issues(
    *,
    answer: str,
    question: str = "",
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile | None,
    retrieved_context: list[RetrievedContext],
) -> list[str]:
    issues: list[str] = []
    normalized = answer.strip()
    if is_minimal_unknown_fallback_answer(normalized):
        return []

    lowered = normalized.lower()
    required_sections = ["direct answer", "rationale", "caution", "general advice"]
    if not normalized:
        issues.append("Answer is empty.")
    for section in required_sections:
        if section not in lowered:
            issues.append(f"Answer is missing the '{section}' section.")
    if len(normalized.split()) < 35:
        issues.append("Answer is too short to be useful.")
    if any(term in lowered for term in DISALLOWED_SOURCE_REFERENCES):
        issues.append("Answer mentions sources or document-selection language instead of giving direct guidance.")

    uncertainty_signals = [
        "i don't know",
        "missing",
        "uncertain",
        "limited",
        "cautious",
        "may miss",
        "may be incomplete",
        "relying only",
        "relying on",
        "adjacent text",
        "exact wording matters",
        "retrieve the adjacent excerpt",
        "only on the supplied excerpts",
        "only on the retrieved excerpts",
    ]
    caution_block = lowered.split("4. general advice")[0]
    if all(token not in caution_block for token in uncertainty_signals):
        issues.append("Answer may not clearly communicate uncertainty.")

    direct_answer_block = normalized.lower().split("2. rationale")[0]
    literal_question_mode = is_literal_question_mode(question, patient_variables, clinical_profile)
    explicit_question_only_mode = is_explicit_question_only_mode(question, patient_variables, clinical_profile)
    if literal_question_mode:
        if not answer_addresses_literal_question(answer, question, retrieved_context):
            issues.append("Answer does not directly answer the literal question.")
    elif explicit_question_only_mode:
        if not answer_addresses_explicit_question(answer, question, retrieved_context):
            issues.append("Answer does not directly address the explicit question.")
    elif not has_actionable_guidance(direct_answer_block):
        issues.append("Direct answer summarizes findings but does not provide actionable guidance.")

    potassium_value = to_float(patient_variables.get("potassium"))
    if potassium_value is not None and potassium_value >= 5.0 and "hypokalemia" in lowered:
        issues.append("Answer contradicts the patient potassium value.")

    if clinical_profile is not None:
        specialty = infer_specialty_focus(patient_variables, clinical_profile, retrieved_context)
        focus_clusters = set(prioritized_clusters(clinical_profile, specialty, limit=2))
        for cluster_name, findings in detected_clusters(clinical_profile).items():
            if focus_clusters and cluster_name not in focus_clusters:
                continue
            if any(finding.status in {"low", "high"} for finding in findings):
                mention_targets = {cluster_name.lower(), *(finding.label.lower() for finding in findings[:3])}
                mention_targets.update(_CLUSTER_ALIASES.get(cluster_name, set()))
                if not any(target in lowered for target in mention_targets):
                    issues.append(f"Answer does not acknowledge the '{cluster_name}' abnormality cluster.")

    unsupported_terms = ["aliskiren"]
    if retrieved_context:
        context_text = " ".join(f"{item.title} {item.snippet}" for item in retrieved_context).lower()
        for term in unsupported_terms:
            if term in lowered and term not in context_text:
                issues.append("Answer introduces unsupported treatment-specific wording.")
                break
    return sorted(set(issues))
