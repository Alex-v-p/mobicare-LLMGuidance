from __future__ import annotations

import re
from typing import Any

from inference.clinical import ClinicalProfile
from inference.pipeline.prompts.multistep import DISALLOWED_SOURCE_REFERENCES
from inference.pipeline.support.question_analysis import (
    _expected_item_count,
    answer_addresses_explicit_question,
    answer_addresses_literal_question,
    extract_numbered_items,
    is_explicit_question_only_mode,
    is_literal_question_mode,
    select_relevant_context_sentences,
)
from inference.pipeline.support.specialty import (
    SpecialtyFocus,
    detected_clusters,
    finding_phrase,
    infer_specialty_focus,
    prioritized_clusters,
    synthesize_clinical_state,
    to_float,
)
from shared.contracts.inference import RetrievedContext

def has_actionable_guidance(answer: str) -> bool:
    lowered = answer.lower()
    action_terms = [
        "monitor", "review", "assess", "reassess", "trend", "follow-up", "follow up",
        "evaluate", "check", "prioritize", "seek", "consider", "tolerance", "contributors",
    ]
    return any(term in lowered for term in action_terms)


def is_minimal_unknown_fallback_answer(answer: str) -> bool:
    lowered = re.sub(r"\s+", " ", answer.strip().lower())
    accepted_prefixes = (
        "based on the provided context, i don't know",
        "based on the provided context i don't know",
        "i can't give a grounded answer from the provided context",
        "i cannot give a grounded answer from the provided context",
    )
    return any(lowered.startswith(prefix) for prefix in accepted_prefixes)


def build_unknown_fallback_answer() -> str:
    return "Based on the provided context, I don't know."


def looks_like_generic_clinical_fallback(answer: str) -> bool:
    lowered = answer.strip().lower()
    fallback_markers = (
        "the main value abnormalities point to a clinically relevant pattern",
        "interpreted as a whole rather than marker by marker",
        "use the retrieved guidance to prioritize the most abnormal findings first",
        "review these results together with symptoms",
        "key missing context that could change the recommendation",
    )
    return sum(marker in lowered for marker in fallback_markers) >= 2


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
                cluster_aliases = {
                    "HF severity and congestion": {"heart failure", "hf", "congestion", "decompensation", "natriuretic"},
                    "Cardio-renal and electrolyte safety": {"renal", "kidney", "electrolyte", "potassium", "sodium", "creatinine", "cardio-renal"},
                    "Rhythm and conduction": {"rhythm", "conduction", "heart rate", "qrs", "atrial fibrillation"},
                    "Anemia and iron status": {"anemia", "iron", "hemoglobin", "ferritin", "transferrin"},
                    "Inflammation and injury": {"inflammation", "injury", "crp", "troponin", "inflammatory"},
                    "Glycemic and cardiometabolic risk": {"glycemic", "cardiometabolic", "glucose", "hba1c", "diabetes", "metabolic"},
                }
                mention_targets.update(cluster_aliases.get(cluster_name, set()))
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
    prefer_unknown_fallback: bool = False,
) -> str:
    if is_literal_question_mode(question, patient_variables, clinical_profile):
        return build_literal_question_answer(
            question=question,
            retrieved_context=retrieved_context,
            context_assessment=context_assessment,
        )
    if is_explicit_question_only_mode(question, patient_variables, clinical_profile):
        return build_context_question_answer(
            question=question,
            retrieved_context=retrieved_context,
            context_assessment=context_assessment,
        )
    if prefer_unknown_fallback:
        return build_unknown_fallback_answer()

    clusters = detected_clusters(clinical_profile)
    specialty = infer_specialty_focus(patient_variables, clinical_profile, retrieved_context)
    synthesis = synthesize_clinical_state(
        patient_variables=patient_variables,
        clinical_profile=clinical_profile,
        retrieved_context=retrieved_context,
        context_assessment=context_assessment,
        specialty=specialty,
    )
    direct_lines = build_direct_answer_lines(
        clusters=clusters,
        patient_variables=patient_variables,
        retrieved_context=retrieved_context,
        specialty=specialty,
        context_assessment=context_assessment,
        synthesis=synthesis,
    )
    rationale_lines = build_rationale_lines(clusters=clusters, specialty=specialty, synthesis=synthesis)
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


def build_literal_question_answer(
    *,
    question: str,
    retrieved_context: list[RetrievedContext],
    context_assessment: Any,
) -> str:
    combined = " ".join(item.snippet for item in retrieved_context)
    enumerated_items = extract_numbered_items(combined)
    expected_count = _expected_item_count(question)
    direct_lines: list[str] = []
    if enumerated_items and expected_count:
        direct_lines = [f"- {item}." for item in enumerated_items[:expected_count]]
    elif enumerated_items:
        direct_lines = [f"- {item}." for item in enumerated_items[:4]]
    else:
        direct_lines = [f"- {sentence}" for sentence in select_relevant_context_sentences(question, retrieved_context, limit=3)]

    if not direct_lines:
        direct_lines = ["- Unavailable from the retrieved evidence."]

    rationale_lines = []
    if enumerated_items:
        rationale_lines.append(f"- The retrieved excerpts explicitly enumerate {len(enumerated_items[: expected_count or len(enumerated_items)])} items relevant to the question.")
    for sentence in select_relevant_context_sentences(question, retrieved_context, limit=2):
        rationale_lines.append(f"- {sentence}")
    if not rationale_lines:
        rationale_lines.append("- The available excerpts do not provide enough detail for a stronger grounded answer.")

    caution_lines = []
    if not getattr(context_assessment, "sufficient", False):
        caution_lines.append("- I cannot give a fuller answer with confidence because the retrieved excerpts appear partial or incomplete.")
    if combined and combined.rstrip()[-12:].lower().endswith("memb"):
        caution_lines.append("- One excerpt appears truncated, so the final item wording may be incomplete.")
    if not caution_lines:
        caution_lines.append("- I am relying only on the supplied excerpts and may miss detail that appears in adjacent text.")

    general_advice = [
        "- If exact wording matters, retrieve the adjacent excerpt or the full table entry before making the answer more specific.",
    ]
    lines = [
        "1. Direct answer",
        *direct_lines[: max(expected_count or 0, 4) or 4],
        "",
        "2. Rationale",
        *rationale_lines[:3],
        "",
        "3. Caution",
        *caution_lines[:2],
        "",
        "4. General advice",
        *general_advice,
    ]
    return "\n".join(lines).strip()


def build_context_question_answer(
    *,
    question: str,
    retrieved_context: list[RetrievedContext],
    context_assessment: Any,
) -> str:
    sentence_matches = select_relevant_context_sentences(question, retrieved_context, limit=3)
    combined = " ".join(item.snippet for item in retrieved_context)
    enumerated_items = extract_numbered_items(combined)

    direct_lines: list[str] = []
    if enumerated_items:
        direct_lines = [f"- {item}." for item in enumerated_items[:4]]
    elif sentence_matches:
        direct_lines = [f"- {sentence.rstrip('.')} .".replace('  ', ' ') for sentence in sentence_matches]
    else:
        direct_lines = ["- Unavailable from the retrieved evidence."]

    rationale_lines = []
    for sentence in sentence_matches[:2]:
        rationale_lines.append(f"- {sentence}")
    if not rationale_lines:
        rationale_lines.append("- The available excerpts do not provide enough detail for a stronger grounded answer.")

    caution_lines = []
    if not getattr(context_assessment, "sufficient", False):
        caution_lines.append("- I cannot give a fuller answer with confidence because the retrieved excerpts appear partial or incomplete.")
    if not caution_lines:
        caution_lines.append("- I am relying only on the supplied excerpts and may miss detail that appears in adjacent text.")

    general_advice = [
        "- Retrieve adjacent text or the full table entry before making the answer more specific or treatment-prescriptive.",
    ]
    lines = [
        "1. Direct answer",
        *direct_lines[:4],
        "",
        "2. Rationale",
        *rationale_lines[:3],
        "",
        "3. Caution",
        *caution_lines[:2],
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

    if len(direct_lines) < 3:
        for cluster_name in [
            "HF severity and congestion",
            "Cardio-renal and electrolyte safety",
            "Rhythm and conduction",
            "Anemia and iron status",
            "Inflammation and injury",
            "Glycemic and cardiometabolic risk",
        ] if specialty.is_heart_failure else list(clusters.keys())[:4]:
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


def missing_details(
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    specialty: SpecialtyFocus,
) -> list[str]:
    keys = {key.lower() for key in patient_variables}
    clusters = detected_clusters(clinical_profile)

    def has_any(*candidates: str) -> bool:
        return any(candidate.lower() in keys for candidate in candidates)

    missing: list[str] = []
    if specialty.is_heart_failure:
        if not has_any("symptoms", "nyha", "orthopnea", "edema", "rales", "angina01"):
            missing.append("current symptoms or functional status")
        if not has_any("blood_pressure", "sbp", "dbp", "bpsyst", "bpdiast"):
            missing.append("blood pressure or hypotension tolerance")
        if not has_any("volume_status", "edema", "orthopnea", "rales", "jugularvein_01", "hepatomegaly", "weight"):
            missing.append("congestion or volume status")
        if not has_any(
            "medication_history",
            "dosebb_prev",
            "rasdose_prev",
            "dosespiro_prev",
            "loop_dose_prev",
            "sglt2dose_prev",
            "arnidose_prev",
        ):
            missing.append("active HF medications or recent dose changes")
        if "Cardio-renal and electrolyte safety" in clusters and not has_any(
            "baseline_creatinine",
            "prior_results",
            "prior_creatinine",
            "prior_egfr",
        ):
            missing.append("baseline renal function or recent laboratory trends")
    elif specialty.name == "metabolic":
        if not has_any("symptoms"):
            missing.append("current symptoms")
        if not has_any("diabetes_history"):
            missing.append("diabetes history")
        if not has_any("medication_history"):
            missing.append("glucose-lowering medication history")
        if not has_any("fasting_status"):
            missing.append("fasting versus random sampling context")
        if not has_any("prior_hba1c", "prior_results"):
            missing.append("prior HbA1c or glucose trends")
    else:
        if not has_any("symptoms"):
            missing.append("current symptoms")
        if not has_any("medication_history"):
            missing.append("medication history")
        if not has_any("prior_results"):
            missing.append("prior laboratory trends")
        if not has_any("diagnosis"):
            missing.append("working diagnosis or clinical context")
    if specialty.is_heart_failure and any(f.key in {"ef", "lvef"} for f in clinical_profile.abnormal_variables) and "prior_ef" not in keys:
        missing.append("prior EF")
    return missing
