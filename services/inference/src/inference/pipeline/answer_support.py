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


_QUESTION_LITERAL_TERMS = {"table", "figure", "supplementary", "appendix", "section", "described", "listed"}
_GENERIC_NON_ANSWER_PHRASES = {
    "clinically relevant pattern",
    "interpreted as a whole rather than marker by marker",
    "prioritize the most abnormal findings first",
    "use the retrieved guidance to prioritize",
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


@dataclass(frozen=True, slots=True)
class ClinicalSynthesis:
    headline: str
    interpretation_points: tuple[str, ...]
    action_points: tuple[str, ...]
    covered_clusters: tuple[str, ...]


_HF_FOCUS_CLUSTER_ORDER = (
    "HF severity and congestion",
    "Cardio-renal and electrolyte safety",
    "Rhythm and conduction",
    "Anemia and iron status",
    "Inflammation and injury",
    "Glycemic and cardiometabolic risk",
)


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


def extract_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]{3,}", text.lower()) if term not in _STOPWORDS}


def context_key(item: RetrievedContext) -> tuple[str, str | None, str]:
    return (item.source_id, item.chunk_id, item.snippet)


def is_literal_question_mode(
    question: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile | None = None,
) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False

    profile = clinical_profile
    has_patient_context = bool(patient_variables) or bool(profile and (profile.recognized_variables or profile.abnormal_variables))
    patient_language = any(
        term in normalized
        for term in {
            "patient", "symptom", "symptoms", "result", "results", "value", "values", "profile",
            "management", "follow-up", "follow up", "monitor", "monitoring", "safety", "treat",
            "treatment", "prescribe", "medication", "dose", "escalation",
        }
    )
    if not has_patient_context and not patient_language:
        return True
    return any(term in normalized for term in _QUESTION_LITERAL_TERMS) and not patient_language


def is_explicit_question_only_mode(
    question: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile | None = None,
) -> bool:
    normalized = question.strip()
    if not normalized:
        return False

    profile = clinical_profile
    has_patient_context = bool(patient_variables) or bool(profile and (profile.recognized_variables or profile.abnormal_variables))
    return not has_patient_context


def _question_focus_terms(question: str, retrieved_context: list[RetrievedContext]) -> set[str]:
    terms = extract_terms(question)
    combined = " ".join(f"{item.title} {item.snippet}" for item in retrieved_context).lower()
    return {term for term in terms if term in combined and len(term) > 3}


def _expected_item_count(question: str) -> int | None:
    lowered = question.lower()
    digit_match = re.search(r"\b([2-9]|10)\b", lowered)
    if digit_match:
        return int(digit_match.group(1))
    for word, value in {
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }.items():
        if re.search(rf"\b{word}\b", lowered):
            return value
    return None


def extract_numbered_items(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<!\d)\b([1-9])\.\s*", collapsed)
    items: list[str] = []
    for index in range(1, len(parts), 2):
        if index + 1 >= len(parts):
            break
        cleaned = re.sub(r"\s+", " ", parts[index + 1]).strip(" ;,.-")
        if cleaned:
            items.append(cleaned)
    return items


def select_relevant_context_sentences(question: str, retrieved_context: list[RetrievedContext], *, limit: int = 3) -> list[str]:
    focus_terms = _question_focus_terms(question, retrieved_context) or extract_terms(question)
    scored: list[tuple[int, str]] = []
    for item in retrieved_context:
        for sentence in re.split(r"(?<=[.!?;])\s+", item.snippet):
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if not cleaned:
                continue
            sentence_terms = extract_terms(cleaned)
            overlap = len(focus_terms & sentence_terms)
            scored.append((overlap, cleaned))
    scored.sort(key=lambda entry: (entry[0], len(entry[1])), reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for overlap, sentence in scored:
        if sentence in seen:
            continue
        if overlap <= 0 and selected:
            continue
        selected.append(sentence)
        seen.add(sentence)
        if len(selected) >= limit:
            break
    return selected


def answer_addresses_literal_question(answer: str, question: str, retrieved_context: list[RetrievedContext]) -> bool:
    direct_block = answer.lower().split("2. rationale", 1)[0]
    if any(phrase in direct_block for phrase in _GENERIC_NON_ANSWER_PHRASES):
        return False

    expected_count = _expected_item_count(question)
    enumerated_items = extract_numbered_items(" ".join(item.snippet for item in retrieved_context))
    direct_lines = [
        line.strip()
        for line in direct_block.splitlines()
        if line.strip().startswith(("-", "1.", "2.", "3.", "4.", "5."))
    ]
    if expected_count and enumerated_items and len(direct_lines) < min(expected_count, 3):
        return False

    focus_terms = _question_focus_terms(question, retrieved_context)
    if sum(1 for term in focus_terms if term in direct_block) >= 2:
        return True

    anchor_terms = set()
    for item in enumerated_items[: max(expected_count or 0, 3) or 3]:
        item_terms = [term for term in extract_terms(item) if len(term) > 4]
        anchor_terms.update(item_terms[:2])
    if anchor_terms and any(term in direct_block for term in anchor_terms):
        return True

    sentence_matches = select_relevant_context_sentences(question, retrieved_context, limit=2)
    return any(len(extract_terms(sentence) & extract_terms(direct_block)) >= 3 for sentence in sentence_matches)


def answer_addresses_explicit_question(answer: str, question: str, retrieved_context: list[RetrievedContext]) -> bool:
    direct_block = answer.lower().split("2. rationale", 1)[0]
    if any(phrase in direct_block for phrase in _GENERIC_NON_ANSWER_PHRASES):
        return False

    focus_terms = _question_focus_terms(question, retrieved_context) or extract_terms(question)
    focus_terms = {term for term in focus_terms if len(term) > 3}
    if sum(1 for term in focus_terms if term in direct_block) >= min(2, max(1, len(focus_terms))):
        return True

    sentence_matches = select_relevant_context_sentences(question, retrieved_context, limit=3)
    if any(len(extract_terms(sentence) & extract_terms(direct_block)) >= 3 for sentence in sentence_matches):
        return True

    enumerated_items = extract_numbered_items(" ".join(item.snippet for item in retrieved_context))
    if enumerated_items:
        anchor_terms: set[str] = set()
        for item in enumerated_items[:4]:
            anchor_terms.update(term for term in extract_terms(item) if len(term) > 4)
        if anchor_terms and any(term in direct_block for term in list(anchor_terms)[:6]):
            return True

    return False


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
