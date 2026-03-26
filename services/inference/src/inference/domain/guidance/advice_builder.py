from __future__ import annotations

from typing import Any

from inference.clinical import ClinicalProfile
from inference.domain.guidance.missing_context import missing_details
from inference.domain.guidance.specialty import (
    ClinicalSynthesis,
    SpecialtyFocus,
    finding_phrase,
    prioritized_clusters,
    to_float,
)
from shared.contracts.inference import RetrievedContext


def build_direct_answer_lines(
    *,
    clusters: dict[str, list[ClinicalFinding]],
    patient_variables: dict[str, Any],
    retrieved_context: list[RetrievedContext],
    specialty: SpecialtyFocus,
    context_assessment: Any,
    synthesis: ClinicalSynthesis,
) -> list[str]:
    evidence_points = build_evidence_points(retrieved_context, patient_variables, specialty)
    direct_lines: list[str] = []

    if synthesis.headline:
        direct_lines.append(f"- {synthesis.headline}")

    for point in synthesis.action_points[:3]:
        direct_lines.append(f"- {point}")

    if specialty.is_heart_failure and not synthesis.action_points:
        direct_lines.append("- Prioritize assessment of congestion, cardio-renal safety, blood pressure tolerance, and near-term follow-up needs.")

    if evidence_points:
        direct_lines.extend(f"- {point}" for point in evidence_points[:2])

    if not context_assessment.sufficient:
        direct_lines.append("- Treatment decisions should stay cautious because only part of the clinical picture is directly covered by the retrieved guidance.")

    cluster_order = [
        "HF severity and congestion",
        "Cardio-renal and electrolyte safety",
        "Rhythm and conduction",
        "Anemia and iron status",
        "Inflammation and injury",
        "Glycemic and cardiometabolic risk",
    ] if specialty.is_heart_failure else list(clusters.keys())[:4]
    if len(direct_lines) < 3:
        for cluster_name in cluster_order:
            findings = clusters.get(cluster_name)
            if findings:
                direct_lines.append(f"- {cluster_name}: {', '.join(finding_phrase(finding) for finding in findings[:3])}.")
            if len(direct_lines) >= 5:
                break

    return direct_lines[:5] or ["- The main abnormalities require cautious follow-up and clinical review even though the retrieved guidance is incomplete."]


def build_rationale_lines(
    *,
    clusters: dict[str, list[ClinicalFinding]],
    specialty: SpecialtyFocus,
    synthesis: ClinicalSynthesis | None = None,
) -> list[str]:
    rationale_lines: list[str] = []

    if synthesis is not None:
        headline = synthesis.interpretation_points[:1]
        supporting = synthesis.interpretation_points[1:2]
        for point in headline + supporting:
            rationale_lines.append(f"- {point}")

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
    biomarker_lines: list[str] = []
    for cluster_name in preferred_order + list(clusters.keys()):
        if cluster_name in seen or cluster_name not in clusters:
            continue
        seen.add(cluster_name)
        findings = clusters[cluster_name]
        if not findings:
            continue
        detailed_findings = "; ".join(finding.summary for finding in findings[:2])
        biomarker_lines.append(f"- {cluster_name}: {detailed_findings}")
        if len(biomarker_lines) >= 3:
            break

    if biomarker_lines:
        remaining = max(0, 5 - len(rationale_lines))
        rationale_lines.extend(biomarker_lines[:remaining])

    return rationale_lines[:5] or [f"- {specialty.summary}"]


def build_caution_lines(
    patient_variables: dict[str, Any],
    context_assessment: Any,
    clinical_profile: ClinicalProfile,
    specialty: SpecialtyFocus,
) -> list[str]:
    caution_lines: list[str] = []
    if context_assessment.reasons:
        caution_lines.append(
            "- I cannot make a treatment-specific recommendation with confidence because the retrieved guidance only partially covers the dominant findings."
        )
    focus_clusters = prioritized_clusters(clinical_profile, specialty, limit=2)
    uncovered = [cluster for cluster in focus_clusters if context_assessment.cluster_coverage.get(cluster, 0) == 0]
    if uncovered:
        caution_lines.append(f"- Guidance coverage is limited for: {', '.join(uncovered)}.")
    missing = missing_details(patient_variables, clinical_profile, specialty)
    if missing:
        caution_lines.append(
            f"- Key missing context that could change the recommendation: {', '.join(missing[:3])}."
        )
    return caution_lines or [
        "- I do not have the full history, medication list, and trend data needed for a stronger patient-specific conclusion."
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
