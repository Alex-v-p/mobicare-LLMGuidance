from __future__ import annotations

import re
from typing import Any

from inference.clinical import ClinicalFinding, ClinicalProfile
from inference.pipeline.prompts.multistep import DISALLOWED_SOURCE_REFERENCES
from shared.contracts.inference import RetrievedContext

_STOPWORDS = {
    "about", "according", "after", "all", "and", "are", "based", "been", "for", "from",
    "guidance", "has", "have", "into", "most", "not", "that", "the", "their", "this", "treatment",
    "what", "with", "would", "patient", "variables", "management", "follow", "relevant", "document",
    "grounded", "available", "data", "should", "could", "regarding", "focus", "clinical",
}


def extract_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]{3,}", text.lower()) if term not in _STOPWORDS}



def context_key(item: RetrievedContext) -> tuple[str, str | None, str]:
    return (item.source_id, item.chunk_id, item.snippet)



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



def should_force_deterministic_answer(
    *,
    answer: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    context_assessment: Any,
) -> bool:
    issues = collect_answer_issues(
        answer=answer,
        patient_variables=patient_variables,
        clinical_profile=clinical_profile,
        retrieved_context=[],
    )
    if issues or not context_assessment.sufficient:
        return True
    abnormal_clusters = detected_clusters(clinical_profile)
    return any(context_assessment.cluster_coverage.get(cluster, 0) == 0 for cluster in abnormal_clusters)



def collect_answer_issues(
    *,
    answer: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile | None,
    retrieved_context: list[RetrievedContext],
) -> list[str]:
    issues: list[str] = []
    normalized = answer.strip()
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
    if all(token not in lowered for token in ["i don't know", "missing", "uncertain", "limited"]):
        issues.append("Answer may not clearly communicate uncertainty.")

    potassium_value = to_float(patient_variables.get("potassium"))
    if potassium_value is not None and potassium_value >= 5.0 and "hypokalemia" in lowered:
        issues.append("Answer contradicts the patient potassium value.")

    if clinical_profile is not None:
        for cluster_name, findings in detected_clusters(clinical_profile).items():
            if any(finding.status in {"low", "high"} for finding in findings):
                cluster_word = cluster_name.lower().split(" and ")[0]
                mention_targets = {cluster_word, *(finding.label.lower() for finding in findings[:3])}
                if not any(target in lowered for target in mention_targets):
                    issues.append(f"Answer does not acknowledge the '{cluster_name}' abnormality cluster.")

    unsupported_terms = ["mra", "mineralocorticoid receptor antagonist", "aliskiren"]
    if retrieved_context:
        context_text = " ".join(f"{item.title} {item.snippet}" for item in retrieved_context).lower()
        for term in unsupported_terms:
            if term in lowered and term not in context_text:
                issues.append("Answer introduces unsupported treatment-specific wording.")
                break
    return sorted(set(issues))



def build_deterministic_answer(
    *,
    question: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    retrieved_context: list[RetrievedContext],
    context_assessment: Any,
) -> str:
    clusters = detected_clusters(clinical_profile)
    cluster_lines: list[str] = []
    rationale_lines: list[str] = []
    for cluster_name, findings in clusters.items():
        labels = ", ".join(finding_phrase(finding) for finding in findings[:3])
        cluster_lines.append(f"- {cluster_name}: {labels}.")
        rationale_lines.append(f"- {cluster_name}: " + "; ".join(finding.summary for finding in findings[:2]))

    evidence_points = build_evidence_points(retrieved_context, patient_variables)
    if evidence_points:
        cluster_lines.extend(f"- {point}" for point in evidence_points[:2])

    caution_lines = build_caution_lines(patient_variables, context_assessment)
    general_advice = build_general_advice(context_assessment)
    rationale_block = rationale_lines[:4] or [
        "- The answer is based on the interpreted patient findings and the available grounded excerpts."
    ]
    lines = [
        "1. Direct answer",
        *cluster_lines[:4],
        "",
        "2. Rationale",
        *rationale_block,
        "",
        "3. Caution",
        *caution_lines,
        "",
        "4. General advice",
        *general_advice,
    ]
    return "\n".join(lines).strip()



def build_caution_lines(patient_variables: dict[str, Any], context_assessment: Any) -> list[str]:
    caution_lines: list[str] = []
    if context_assessment.reasons:
        caution_lines.append(
            "- I don't know the full conclusion because the retrieved evidence is incomplete for some abnormal findings."
        )
    uncovered = [cluster for cluster, count in context_assessment.cluster_coverage.items() if count == 0]
    if uncovered:
        caution_lines.append(f"- Evidence is limited for: {', '.join(uncovered)}.")

    for item in missing_details(patient_variables)[:3]:
        caution_lines.append(f"- Missing detail: {item}.")

    return caution_lines or [
        "- I don't know the full conclusion because symptoms, medication history, and baseline trends are still missing."
    ]



def build_general_advice(context_assessment: Any) -> list[str]:
    advice = [
        "- Review these results together with symptoms, medication history, and prior laboratory trends.",
        "- Repeat or trend abnormal values when clinically appropriate, especially renal function, potassium, sodium, and inflammatory markers.",
    ]
    if not context_assessment.sufficient:
        advice.append("- Do not over-interpret findings that are not well covered by the retrieved guidance.")
    return advice



def build_evidence_points(retrieved_context: list[RetrievedContext], patient_variables: dict[str, Any]) -> list[str]:
    if not retrieved_context:
        return []
    combined = " ".join(item.snippet for item in retrieved_context).lower()
    points: list[str] = []
    if any(term in combined for term in ["creatinine", "renal function", "electrolytes"]):
        points.append("The grounded guidance supports close monitoring of renal function and electrolytes")
    if "nephrotoxic" in combined:
        points.append("The grounded guidance supports reviewing nephrotoxic medicines or other contributors to renal stress")
    if "potassium" in combined and (to_float(patient_variables.get("potassium")) or 0) >= 5.0:
        points.append("The grounded guidance supports follow-up of elevated potassium rather than assuming it is benign")
    return points



def missing_details(patient_variables: dict[str, Any]) -> list[str]:
    expected = ["symptoms", "medication_history", "baseline_creatinine", "prior_results", "diagnosis"]
    return [item.replace("_", " ") for item in expected if item not in patient_variables]



def finding_phrase(finding: ClinicalFinding) -> str:
    unit = f" {finding.unit}" if finding.unit else ""
    return f"{finding.label} {finding.status} ({finding.value}{unit})"



def to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None



def detected_clusters(clinical_profile: ClinicalProfile) -> dict[str, list[ClinicalFinding]]:
    clusters: dict[str, list[ClinicalFinding]] = {}
    for finding in clinical_profile.abnormal_variables:
        cluster = cluster_for_finding(finding)
        clusters.setdefault(cluster, []).append(finding)
    return clusters



def cluster_for_finding(finding: ClinicalFinding) -> str:
    key = finding.key.lower()
    label = finding.label.lower()
    if key in {"creatinine", "urea", "bun", "egfr", "cysc", "cystatin_c", "potassium"} or "creatin" in label:
        return "Renal function and potassium"
    if key in {"ferritin", "hemoglobin", "hb", "haemoglobin", "hematocrit", "haematocrit", "transferrin"}:
        return "Anemia and iron status"
    if key in {"crp", "hscrp", "hs_crp", "il6", "il_6", "c_reactive_protein"} or "reactive protein" in label:
        return "Inflammation"
    if key in {"sodium", "chloride", "magnesium"}:
        return "Electrolytes"
    if key in {"bnp", "nt_pro_bnp", "troponin", "hstnt", "ef", "qrs"}:
        return "Cardiac status"
    if key in {"hba1c", "glucose"}:
        return "Glycemic status"
    if key in {"cholesterol", "ldl", "hdl", "triglycerides"}:
        return "Lipids"
    return "Other findings"



def context_matches_findings(combined: str, findings: list[ClinicalFinding]) -> bool:
    for finding in findings:
        tokens = {finding.key.lower(), finding.label.lower()}
        if any(token in combined for token in tokens):
            return True
        if finding.key.lower() == "potassium" and "renal function" in combined:
            return True
    return False
