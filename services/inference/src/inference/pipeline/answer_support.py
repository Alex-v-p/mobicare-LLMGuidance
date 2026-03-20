from __future__ import annotations

import re
from dataclasses import dataclass
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

_CARDIAC_KEYS = {
    "bnp", "nt_pro_bnp", "ef", "lvef", "nyha", "edema", "rales", "orthopnea", "jugularvein_01",
    "hepatomegaly", "qrsduur", "heart_rate", "heartrate", "bpsyst", "bpdiast", "sbp", "dbp",
    "afib_bl", "bundeltakblok", "pm_rhythm_ecg", "pacemaker_bl", "valvhd", "ischaemic", "hist_mi",
    "cabg", "angina01", "dosebb_prev", "rasdose_prev", "dosespiro_prev", "loop_dose_prev",
    "sglt2dose_prev", "arnidose_prev", "potassium", "sodium", "crea", "creatinine", "urea", "cysc",
    "cystatin_c", "hstnt", "hscrp", "hs_crp", "crp", "il6", "il_6",
}
_GLYCEMIC_KEYS = {"glucose", "hba1c", "diabetes"}
_RENAL_KEYS = {"crea", "creatinine", "urea", "cysc", "cystatin_c", "kidney_disease", "potassium", "sodium"}
_HEART_FAILURE_KEYWORDS = {
    "heart failure", "hfr ef", "hfref", "hfmr ef", "hfpef", "congestion", "decongestion",
    "gdmt", "euvolaemia", "euvolemia", "hyperkalaemia", "hyperkalemia", "hyponatraemia", "hyponatremia",
    "raas", "ace-i", "arb", "arni", "mra", "sglt2", "diuretic", "ivabradine", "nt-probnp", "bnp",
}


@dataclass(frozen=True, slots=True)
class SpecialtyFocus:
    name: str
    score: int
    summary: str
    retrieval_hints: tuple[str, ...]
    prompt_priorities: tuple[str, ...]

    @property
    def is_heart_failure(self) -> bool:
        return self.name == "heart_failure"


def extract_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]{3,}", text.lower()) if term not in _STOPWORDS}


def context_key(item: RetrievedContext) -> tuple[str, str | None, str]:
    return (item.source_id, item.chunk_id, item.snippet)


def infer_specialty_focus(
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    retrieved_context: list[RetrievedContext] | None = None,
) -> SpecialtyFocus:
    keys = {key.lower() for key in patient_variables}
    recognized = {finding.key.lower() for finding in clinical_profile.recognized_variables}
    all_keys = keys | recognized
    cardiac_score = len(all_keys & _CARDIAC_KEYS)
    renal_score = len(all_keys & _RENAL_KEYS)
    glycemic_score = len(all_keys & _GLYCEMIC_KEYS)

    if retrieved_context:
        corpus = " ".join(f"{item.title} {item.snippet}" for item in retrieved_context).lower()
        if any(keyword in corpus for keyword in _HEART_FAILURE_KEYWORDS):
            cardiac_score += 3

    if cardiac_score >= max(glycemic_score + 1, renal_score, 2):
        return SpecialtyFocus(
            name="heart_failure",
            score=cardiac_score,
            summary="Heart-failure-first interpretation with cardio-renal safety, congestion, rhythm, and escalation awareness.",
            retrieval_hints=(
                "heart failure",
                "HFrEF or HF management",
                "cardio-renal safety",
                "congestion or decongestion",
                "guideline-directed therapy",
            ),
            prompt_priorities=(
                "Prioritize heart-failure syndrome severity, congestion, cardio-renal safety, rhythm/conduction, and medication-safety implications.",
                "Prefer heart-failure phrasing such as congestion, decompensation, GDMT, renal/electrolyte safety, and specialist HF review when the evidence supports it.",
                "Use non-cardiac markers as comorbidities or modifiers of HF risk unless the retrieved evidence clearly centers another specialty.",
            ),
        )
    if glycemic_score >= 2:
        return SpecialtyFocus(
            name="metabolic",
            score=glycemic_score,
            summary="Metabolic interpretation with emphasis on persistent glycemic abnormality and measurement context.",
            retrieval_hints=("glycemic control", "HbA1c", "glucose", "diabetes follow-up"),
            prompt_priorities=(
                "Prioritize persistent glycemic abnormality, confirm measurement context, and mention diabetes history or treatment context when missing.",
            ),
        )
    if renal_score >= 2:
        return SpecialtyFocus(
            name="renal",
            score=renal_score,
            summary="Renal interpretation with electrolyte and medication-safety awareness.",
            retrieval_hints=("renal function", "electrolytes", "kidney safety", "drug safety"),
            prompt_priorities=(
                "Prioritize renal function, electrolytes, and medication-safety implications while keeping the answer cautious.",
            ),
        )
    return SpecialtyFocus(
        name="general",
        score=0,
        summary="General multi-specialty interpretation grounded in the retrieved evidence.",
        retrieval_hints=(),
        prompt_priorities=(
            "Stay specialty-agnostic unless the findings or retrieved evidence clearly indicate a dominant clinical domain.",
        ),
    )


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
                mention_targets = {cluster_name.lower(), *(finding.label.lower() for finding in findings[:3])}
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


def build_deterministic_answer(
    *,
    question: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    retrieved_context: list[RetrievedContext],
    context_assessment: Any,
) -> str:
    clusters = detected_clusters(clinical_profile)
    specialty = infer_specialty_focus(patient_variables, clinical_profile, retrieved_context)
    direct_lines = build_direct_answer_lines(
        clusters=clusters,
        patient_variables=patient_variables,
        retrieved_context=retrieved_context,
        specialty=specialty,
        context_assessment=context_assessment,
    )
    rationale_lines = build_rationale_lines(clusters=clusters, specialty=specialty)
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


def build_direct_answer_lines(
    *,
    clusters: dict[str, list[ClinicalFinding]],
    patient_variables: dict[str, Any],
    retrieved_context: list[RetrievedContext],
    specialty: SpecialtyFocus,
    context_assessment: Any,
) -> list[str]:
    evidence_points = build_evidence_points(retrieved_context, patient_variables, specialty)
    cluster_lines: list[str] = []
    if specialty.is_heart_failure:
        for cluster_name in [
            "HF severity and congestion",
            "Cardio-renal and electrolyte safety",
            "Rhythm and conduction",
            "Anemia and iron status",
            "Inflammation and injury",
            "Glycemic and cardiometabolic risk",
        ]:
            findings = clusters.get(cluster_name)
            if findings:
                cluster_lines.append(f"- {cluster_name}: {', '.join(finding_phrase(finding) for finding in findings[:3])}.")
        if evidence_points:
            cluster_lines.extend(f"- {point}" for point in evidence_points[:2])
        if not context_assessment.sufficient:
            cluster_lines.append("- The safest grounded conclusion is limited because only part of the heart-failure picture is covered by the retrieved guidance.")
        return cluster_lines[:5]

    for cluster_name, findings in list(clusters.items())[:4]:
        cluster_lines.append(f"- {cluster_name}: {', '.join(finding_phrase(finding) for finding in findings[:3])}.")
    if evidence_points:
        cluster_lines.extend(f"- {point}" for point in evidence_points[:2])
    return cluster_lines[:5] or ["- The main abnormalities are summarized above, but stronger conclusions require more complete evidence."]


def build_rationale_lines(*, clusters: dict[str, list[ClinicalFinding]], specialty: SpecialtyFocus) -> list[str]:
    rationale_lines: list[str] = []
    preferred_order = [
        "HF severity and congestion",
        "Cardio-renal and electrolyte safety",
        "Rhythm and conduction",
        "Anemia and iron status",
        "Inflammation and injury",
        "Glycemic and cardiometabolic risk",
        "Kidney function and electrolytes",
        "Glycemic status",
        "Lipids",
    ]
    seen = set()
    for cluster_name in preferred_order + list(clusters.keys()):
        if cluster_name in seen or cluster_name not in clusters:
            continue
        seen.add(cluster_name)
        findings = clusters[cluster_name]
        rationale_lines.append(f"- {cluster_name}: " + "; ".join(finding.summary for finding in findings[:2]))
    return rationale_lines[:4] or [f"- {specialty.summary}"]


def build_caution_lines(
    patient_variables: dict[str, Any],
    context_assessment: Any,
    clinical_profile: ClinicalProfile,
    specialty: SpecialtyFocus,
) -> list[str]:
    caution_lines: list[str] = []
    if context_assessment.reasons:
        caution_lines.append(
            "- I don't know the full conclusion because the retrieved evidence is incomplete for some abnormal findings."
        )
    uncovered = [cluster for cluster, count in context_assessment.cluster_coverage.items() if count == 0]
    if uncovered:
        caution_lines.append(f"- Evidence is limited for: {', '.join(uncovered)}.")
    for item in missing_details(patient_variables, clinical_profile, specialty)[:4]:
        caution_lines.append(f"- Missing detail: {item}.")
    return caution_lines or [
        "- I don't know the full conclusion because symptoms, medication history, and baseline trends are still missing."
    ]


def build_general_advice(context_assessment: Any, clinical_profile: ClinicalProfile, specialty: SpecialtyFocus) -> list[str]:
    if specialty.is_heart_failure:
        advice = [
            "- Review these results together with symptoms, blood pressure or volume status, medication history, and prior laboratory or imaging trends.",
            "- In heart-failure-oriented cases, trend natriuretic peptides, renal function, potassium, sodium, congestion signs, and therapy tolerance when clinically appropriate.",
        ]
        if any(finding.key in {"ef", "lvef", "bnp", "nt_pro_bnp", "nyha"} for finding in clinical_profile.abnormal_variables):
            advice.append("- If the overall picture suggests worsening heart failure, persistent congestion, or poor tolerance of guideline-directed therapy, seek specialist heart-failure review.")
    elif specialty.name == "metabolic":
        advice = [
            "- Review these results together with symptoms, diabetes history, medication use, and whether the glucose value was fasting or random.",
            "- Trend glucose and HbA1c over time rather than over-interpreting a single measurement in isolation.",
        ]
    else:
        advice = [
            "- Review these results together with symptoms, medication history, and prior laboratory trends.",
            "- Repeat or trend abnormal values when clinically appropriate.",
        ]
    if not context_assessment.sufficient:
        advice.append("- Do not over-interpret findings that are not well covered by the retrieved guidance.")
    return advice


def build_evidence_points(
    retrieved_context: list[RetrievedContext],
    patient_variables: dict[str, Any],
    specialty: SpecialtyFocus,
) -> list[str]:
    if not retrieved_context:
        return []
    combined = " ".join(item.snippet for item in retrieved_context).lower()
    points: list[str] = []
    if specialty.is_heart_failure:
        if any(term in combined for term in ["creatinine", "urea", "electrolytes", "renal function"]):
            points.append("The grounded guidance supports close monitoring of renal function and electrolytes in the heart-failure setting")
        if any(term in combined for term in ["nephrotoxic", "nsaid", "salt substitutes", "potassium supplements"]):
            points.append("The grounded guidance supports reviewing nephrotoxic or potassium-raising contributors when cardio-renal safety is a concern")
        if any(term in combined for term in ["congestion", "euvolaemia", "euvolemia", "diuretic"]):
            points.append("The grounded guidance supports interpreting renal and electrolyte changes together with congestion status and diuretic therapy")
        if any(term in combined for term in ["specialist advice", "advanced heart failure", "referral"]):
            points.append("The grounded guidance supports escalation or specialist review when severe or treatment-limiting features accumulate")
    else:
        if any(term in combined for term in ["creatinine", "renal function", "electrolytes"]):
            points.append("The grounded guidance supports close monitoring of renal function and electrolytes")
        if "nephrotoxic" in combined:
            points.append("The grounded guidance supports reviewing nephrotoxic medicines or other contributors to renal stress")
    if "potassium" in combined and (to_float(patient_variables.get("potassium")) or 0) >= 5.0:
        points.append("The grounded guidance supports follow-up of elevated potassium rather than assuming it is benign")
    return points


def missing_details(
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    specialty: SpecialtyFocus,
) -> list[str]:
    keys = {key.lower() for key in patient_variables}
    if specialty.is_heart_failure:
        expected = ["symptoms", "medication_history", "blood_pressure", "volume_status", "baseline_creatinine", "prior_results"]
    elif specialty.name == "metabolic":
        expected = ["symptoms", "diabetes_history", "medication_history", "fasting_status", "prior_hba1c"]
    else:
        expected = ["symptoms", "medication_history", "prior_results", "diagnosis"]
    missing = [item.replace("_", " ") for item in expected if item not in keys]
    if specialty.is_heart_failure and any(f.key in {"ef", "lvef"} for f in clinical_profile.abnormal_variables) and "prior_ef" not in keys:
        missing.append("prior EF")
    return missing


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
    if key in {"bnp", "nt_pro_bnp", "ef", "lvef", "nyha", "edema", "rales", "orthopnea", "jugularvein_01", "hepatomegaly"}:
        return "HF severity and congestion"
    if key in {"crea", "creatinine", "urea", "bun", "egfr", "cysc", "cystatin_c", "potassium", "sodium", "kidney_disease"} or "creatin" in label:
        return "Cardio-renal and electrolyte safety"
    if key in {"heartrate", "heart_rate", "pulse", "qrs", "qrsduur", "afib_bl", "bundeltakblok", "pm_rhythm_ecg", "pacemaker_bl", "sbp", "dbp", "bpsyst", "bpdiast"}:
        return "Rhythm and conduction"
    if key in {"ferritin", "hemoglobin", "hb", "haemoglobin", "hematocrit", "haematocrit", "transferrin"}:
        return "Anemia and iron status"
    if key in {"crp", "hscrp", "hs_crp", "il6", "il_6", "c_reactive_protein", "hstnt", "hs_tnt", "troponin"} or "reactive protein" in label:
        return "Inflammation and injury"
    if key in {"glucose", "hba1c", "cholesterol", "chol", "ldl", "hdl", "triglycerides", "diabetes"}:
        return "Glycemic and cardiometabolic risk"
    if key in {"bmi", "weight"}:
        return "Volume and nutritional context"
    return "Other findings"


def context_matches_findings(combined: str, findings: list[ClinicalFinding]) -> bool:
    for finding in findings:
        tokens = {finding.key.lower(), finding.label.lower()}
        if any(token in combined for token in tokens):
            return True
        if finding.key.lower() in {"potassium", "sodium", "creatinine", "crea"} and any(term in combined for term in {"renal function", "hyperkalaemia", "hyperkalemia", "hyponatraemia", "hyponatremia"}):
            return True
        if finding.key.lower() in {"bnp", "nt_pro_bnp", "ef", "lvef"} and "heart failure" in combined:
            return True
        if finding.key.lower() in {"qrsduur", "afib_bl"} and any(term in combined for term in {"rhythm", "atrial fibrillation", "qrs"}):
            return True
    return False
