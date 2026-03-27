from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference.clinical import ClinicalFinding, ClinicalProfile
from shared.contracts.inference import RetrievedContext

from inference.domain.guidance.constants import (
    _CARDIAC_KEYS,
    _GLYCEMIC_KEYS,
    _HEART_FAILURE_KEYWORDS,
    _HF_FOCUS_CLUSTER_ORDER,
    _RENAL_KEYS,
)

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


@dataclass(frozen=True, slots=True)
class ClinicalSynthesis:
    headline: str
    interpretation_points: tuple[str, ...]
    action_points: tuple[str, ...]
    covered_clusters: tuple[str, ...]


def prioritized_clusters(
    clinical_profile: ClinicalProfile,
    specialty: SpecialtyFocus | None = None,
    *,
    limit: int = 2,
) -> list[str]:
    clusters = detected_clusters(clinical_profile)
    if not clusters:
        return []

    specialty = specialty or infer_specialty_focus({}, clinical_profile)
    ordered = list(_HF_FOCUS_CLUSTER_ORDER) if specialty.is_heart_failure else list(clusters.keys())
    selected = [cluster for cluster in ordered if cluster in clusters][:limit]

    if len(selected) < min(limit, len(clusters)):
        for cluster in clusters:
            if cluster not in selected:
                selected.append(cluster)
            if len(selected) >= min(limit, len(clusters)):
                break
    return selected


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


def synthesize_clinical_state(
    *,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    retrieved_context: list[RetrievedContext],
    context_assessment: Any | None = None,
    specialty: SpecialtyFocus | None = None,
) -> ClinicalSynthesis:
    specialty = specialty or infer_specialty_focus(patient_variables, clinical_profile, retrieved_context)
    clusters = detected_clusters(clinical_profile)
    covered_clusters = tuple(clusters.keys())
    interpretation_points: list[str] = []
    action_points: list[str] = []

    hf_findings = clusters.get("HF severity and congestion", [])
    renal_findings = clusters.get("Cardio-renal and electrolyte safety", [])
    rhythm_findings = clusters.get("Rhythm and conduction", [])
    anemia_findings = clusters.get("Anemia and iron status", [])
    inflammation_findings = clusters.get("Inflammation and injury", [])
    glycemic_findings = clusters.get("Glycemic and cardiometabolic risk", [])

    sodium_low = any(f.key.lower() == "sodium" and f.status == "low" for f in renal_findings)
    potassium_high = any(f.key.lower() == "potassium" and f.status == "high" for f in renal_findings)
    hr_high = any(f.key.lower() in {"heart_rate", "heartrate", "pulse"} and f.status == "high" for f in rhythm_findings)

    if specialty.is_heart_failure and hf_findings:
        interpretation_points.append(
            "The overall profile is compatible with significant heart-failure burden or congestion risk rather than an isolated laboratory abnormality."
        )
        action_points.append(
            "Assess heart-failure severity, congestion or volume status, and tolerance of current therapy as a near-term priority."
        )
    if renal_findings:
        renal_text = "Renal function and electrolyte safety appear clinically important in this case"
        if potassium_high or sodium_low:
            renal_text += ", especially because the abnormalities include"
            qualifiers=[]
            if potassium_high:
                qualifiers.append("elevated potassium")
            if sodium_low:
                qualifiers.append("low sodium")
            renal_text += " " + " and ".join(qualifiers)
        interpretation_points.append(renal_text + ".")
        action_points.append(
            "Monitor renal function, potassium, and sodium closely and review treatment tolerance or contributors that could worsen cardio-renal safety."
        )
    if rhythm_findings and hr_high:
        interpretation_points.append(
            "The elevated heart rate may reflect hemodynamic stress, rhythm-related burden, or decompensation in the broader clinical picture."
        )
        action_points.append(
            "Reassess heart rate or rhythm together with blood pressure, symptoms, and overall hemodynamic status."
        )
    if anemia_findings:
        interpretation_points.append(
            "The low hemoglobin or iron markers suggest anemia or iron deficiency that may worsen symptoms, exercise tolerance, and overall cardiovascular risk."
        )
        action_points.append(
            "Work up iron deficiency or anemia in the broader clinical context instead of treating it as a minor incidental finding."
        )
    if inflammation_findings:
        interpretation_points.append(
            "Inflammatory or injury markers are elevated, but the cause cannot be determined from biomarkers alone."
        )
        action_points.append(
            "Interpret inflammatory or injury markers together with symptoms, examination findings, and the rest of the clinical picture before drawing stronger conclusions."
        )
    if glycemic_findings:
        interpretation_points.append(
            "Glycemic or cardiometabolic abnormalities may be contributing comorbidity rather than the main driver, but they still add to overall risk and follow-up needs."
        )
        action_points.append(
            "Review glycemic and cardiometabolic control as part of follow-up, especially if these findings are persistent or already known."
        )

    if not interpretation_points:
        interpretation_points.append(
            "The main value abnormalities point to a clinically relevant pattern that should be interpreted as a whole rather than marker by marker."
        )
    if not action_points:
        action_points.append(
            "Use the retrieved guidance to prioritize the most abnormal findings first, while keeping uncertainty explicit where evidence is thin."
        )

    headline = interpretation_points[0]
    return ClinicalSynthesis(
        headline=headline,
        interpretation_points=tuple(interpretation_points[:4]),
        action_points=tuple(action_points[:4]),
        covered_clusters=covered_clusters,
    )


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
