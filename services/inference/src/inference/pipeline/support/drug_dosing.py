from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from inference.clinical.config_repository import load_drug_dosing_catalog_payload
from shared.contracts.inference import RetrievedContext


MAX_VISIBLE_RECOMMENDATIONS = 3
ACTION_PRIORITY = {
    "switch": 0,
    "start": 1,
    "increase": 2,
    "maintain": 3,
}


def _drug_dosing_catalog() -> dict[str, Any]:
    return load_drug_dosing_catalog_payload()



def _family_priority() -> dict[str, int]:
    raw = _drug_dosing_catalog().get("family_priority") or {}
    return {str(key): int(value) for key, value in raw.items()}



def _default_agent(family: str) -> str:
    raw = _drug_dosing_catalog().get("default_agents") or {}
    value = raw.get(family)
    return str(value or family)



def _family_query_order() -> tuple[str, ...]:
    raw = _drug_dosing_catalog().get("family_query_order") or []
    return tuple(str(item) for item in raw)



def _family_keywords() -> dict[str, set[str]]:
    families = _drug_dosing_catalog().get("families") or {}
    return {
        str(family): {str(keyword).lower() for keyword in (spec.get("keywords") or [])}
        for family, spec in families.items()
    }



def _family_query_template(family: str) -> str:
    families = _drug_dosing_catalog().get("families") or {}
    spec = families.get(family) or {}
    return str(spec.get("query_template") or "")


@dataclass(slots=True)
class DrugEvidence:
    family: str
    drug: str | None
    starting_dose: str | None = None
    target_dose: str | None = None
    reduced_start_dose: str | None = None
    usual_range: str | None = None
    start_low_value: float | None = None
    start_high_value: float | None = None
    target_value: float | None = None
    reduced_start_value: float | None = None
    target_frequency: str | None = None
    caution_potassium_gt: float | None = None
    stop_potassium_gt: float | None = None
    caution_creatinine_gt: float | None = None
    stop_creatinine_gt: float | None = None
    caution_egfr_lt: float | None = None
    stop_egfr_lt: float | None = None
    caution_sbp_lt: float | None = None
    caution_hr_lt: float | None = None
    halve_on_excess: bool = False
    double_uptitration: bool = False
    requires_euvolemia_before_start: bool = False
    washout_note: str | None = None
    contraindication_note: str | None = None
    source_chunk_ids: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DrugRecommendation:
    family: str
    drug: str | None
    action: str
    recommended_dose: str | None
    status: str
    note: str | None = None
    tradeoff: str | None = None
    assumed_agent: bool = False
    grounded: bool = False
    evidence_chunk_ids: list[str] = field(default_factory=list)
    evidence_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------- public builders ----------

def build_snapshot(patient_variables: dict[str, Any]) -> dict[str, Any]:
    return {
        "potassium": _first_float(patient_variables, "potassium", "k", "kplus"),
        "egfr": _first_float(patient_variables, "egfr", "e_gfr"),
        "creatinine": _first_float(patient_variables, "creatinine", "scr", "serum_creatinine"),
        "sbp": _first_float(patient_variables, "bpsyst", "sbp", "blood_pressure_systolic"),
        "heart_rate": _first_float(patient_variables, "heartrate", "heart_rate", "pulse", "hr"),
        "ef": _first_float(patient_variables, "ef", "lvef"),
        "nyha": _first_str(patient_variables, "nyha"),
        "weight": _first_float(patient_variables, "weight"),
        "weight_gain_kg": _first_float(patient_variables, "weight_gain_kg"),
        "congestion_present": _first_bool(patient_variables, "congestion_present"),
        "volume_depleted": _first_bool(patient_variables, "volume_depleted"),
        "switch_to_arni": _first_bool(patient_variables, "switch_to_arni"),
        "dose_mra_prev": _first_raw(patient_variables, "dosespiro_prev", "spiro", "mra_dose_prev", "DoseSpiro_prev"),
        "dose_bb_prev": _first_raw(patient_variables, "dosebb_prev", "bb_dose_prev", "beta_blocker_dose_prev", "DoseBB_prev"),
        "dose_ras_prev": _first_raw(patient_variables, "rasdose_prev", "ras_dose_prev", "acei_dose_prev", "arb_dose_prev", "RASDose_prev"),
        "dose_arni_prev": _first_raw(patient_variables, "arnidose_prev", "arni_dose_prev", "ARNIDose_prev"),
        "dose_sglt2_prev": _first_raw(patient_variables, "sglt2dose_prev", "sglt2_dose_prev", "SGLT2Dose_prev"),
        "dose_loop_prev": _first_raw(patient_variables, "loop_dose_prev", "loop_dose", "loop_diuretic_dose_prev", "Loop_dose_prev"),
        "mra_agent": _normalize_agent(_first_str(patient_variables, "mra_agent", "spiro_agent", "mineralocorticoid_agent"), _default_agent("mra")),
        "beta_blocker_agent": _normalize_agent(_first_str(patient_variables, "beta_blocker_agent", "bb_agent"), _default_agent("beta_blocker")),
        "ras_agent": _normalize_agent(_first_str(patient_variables, "ras_agent", "acei_agent", "arb_agent"), _default_agent("ras")),
        "sglt2_agent": _normalize_agent(_first_str(patient_variables, "sglt2_agent"), _default_agent("sglt2")),
        "loop_agent": _normalize_agent(_first_str(patient_variables, "loop_agent", "diuretic_agent"), _default_agent("loop_diuretic")),
    }


def build_drug_retrieval_queries(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    agent_keys = {
        "mra": "mra_agent",
        "beta_blocker": "beta_blocker_agent",
        "ras": "ras_agent",
        "sglt2": "sglt2_agent",
        "loop_diuretic": "loop_agent",
    }
    queries: list[dict[str, str]] = []
    for family in _family_query_order():
        template = _family_query_template(family)
        if not template:
            continue
        agent = snapshot.get(agent_keys.get(family, "")) if family in agent_keys else None
        resolved_agent = agent or _default_agent(family)
        query = template.format(agent=resolved_agent) if "{agent}" in template else template
        queries.append({"family": family, "query": query})
    return queries


def extract_grounded_drug_evidence(
    *,
    retrieved_context: list[RetrievedContext],
    snapshot: dict[str, Any],
    family_contexts: dict[str, list[RetrievedContext]] | None = None,
) -> dict[str, DrugEvidence]:
    grouped = family_contexts or _group_context_by_family(retrieved_context)
    evidence = {
        "mra": _extract_mra_evidence(grouped.get("mra", []), snapshot),
        "beta_blocker": _extract_beta_blocker_evidence(grouped.get("beta_blocker", []), snapshot),
        "ras": _extract_ras_evidence(grouped.get("ras", []), snapshot),
        "arni": _extract_arni_evidence(grouped.get("arni", []), snapshot),
        "sglt2": _extract_sglt2_evidence(grouped.get("sglt2", []), snapshot),
        "loop_diuretic": _extract_loop_evidence(grouped.get("loop_diuretic", []), snapshot),
    }
    return evidence


def build_grounded_drug_dosing_payload(
    *,
    patient_variables: dict[str, Any],
    retrieved_context: list[RetrievedContext],
    retrieval_queries: list[str],
    family_contexts: dict[str, list[RetrievedContext]] | None = None,
) -> dict[str, Any]:
    snapshot = build_snapshot(patient_variables)
    evidence = extract_grounded_drug_evidence(
        retrieved_context=retrieved_context,
        snapshot=snapshot,
        family_contexts=family_contexts,
    )
    recommendations = {
        "mra": _recommend_mra(snapshot, evidence["mra"]),
        "beta_blocker": _recommend_beta_blocker(snapshot, evidence["beta_blocker"]),
        "ras": _recommend_ras(snapshot, evidence["ras"]),
        "arni": _recommend_arni(snapshot, evidence["arni"]),
        "sglt2": _recommend_sglt2(snapshot, evidence["sglt2"]),
        "loop_diuretic": _recommend_loop(snapshot, evidence["loop_diuretic"]),
    }
    selected = _select_visible_recommendations(recommendations, snapshot)
    tradeoffs = _build_tradeoff_notes(recommendations)
    safety_cautions = _build_safety_cautions(recommendations)
    evidence_rows_used = {
        family: item.to_dict()
        for family, item in evidence.items()
        if item.source_chunk_ids
    }
    return {
        "mode": "drug_dosing_grounded",
        "recommendations": {family: item.to_dict() for family, item in recommendations.items()},
        "selected_recommendations": selected,
        "evidence_rows_used": evidence_rows_used,
        "tradeoffs": tradeoffs,
        "safety_cautions": safety_cautions,
        "inputs_used": snapshot,
        "retrieval_queries": retrieval_queries,
    }


def render_drug_dosing_answer(payload: dict[str, Any]) -> str:
    selections = payload.get("selected_recommendations") or []
    if not selections:
        return "No grounded drug dose recommendation could be made from the retrieved guideline context."
    return "\n".join(_render_visible_recommendation(item) for item in selections)


def summarize_drug_dosing_warnings(payload: dict[str, Any]) -> list[str]:
    warnings = [
        "Drug-dosing mode now requires retrieved guideline evidence before a dose recommendation is surfaced.",
        "The answer field is intentionally short; full evidence rows and safety trade-offs are stored in metadata.drug_dosing_payload.",
    ]
    if not payload.get("evidence_rows_used"):
        warnings.append("No drug-specific evidence rows were extracted from retrieved context.")
    if not payload.get("selected_recommendations"):
        warnings.append("No visible recommendation met both grounding and safety checks.")
    assumptions = [
        rec["drug"]
        for rec in payload.get("recommendations", {}).values()
        if rec.get("assumed_agent")
    ]
    if assumptions:
        warnings.append("Some drug families used assumed agents because no explicit agent was supplied: " + ", ".join(sorted(set(assumptions))))
    return warnings


def verify_grounded_payload(payload: dict[str, Any]) -> tuple[str, list[str], str]:
    selected = payload.get("selected_recommendations") or []
    if not selected:
        if payload.get("safety_cautions"):
            return "pass", ["grounded_safety_cautions_without_visible_uptitration"], "high"
        return "pass", ["no_grounded_recommendation"], "medium"
    ungrounded = [item["family"] for item in selected if not item.get("grounded") or not item.get("evidence_chunk_ids")]
    if ungrounded:
        return "fail", [f"ungrounded_recommendation:{family}" for family in ungrounded], "high"
    return "pass", ["grounded_recommendations_with_evidence"], "high"


# ---------- evidence extraction ----------

def _group_context_by_family(retrieved_context: list[RetrievedContext]) -> dict[str, list[RetrievedContext]]:
    grouped = {family: [] for family in _family_query_order()}
    for item in retrieved_context:
        combined = f"{item.title} {item.snippet}".lower()
        for family, keywords in _family_keywords().items():
            if any(keyword in combined for keyword in keywords):
                grouped[family].append(item)
    return grouped


def _extract_mra_evidence(contexts: list[RetrievedContext], snapshot: dict[str, Any]) -> DrugEvidence:
    agent, _ = _agent_or_default(snapshot.get("mra_agent"), _default_agent("mra"))
    evidence = _base_evidence("mra", agent, contexts)
    text = _combined_text(contexts)
    line = _find_agent_line(contexts, agent)
    evidence.starting_dose, evidence.start_low_value, evidence.target_frequency = _parse_labeled_single_dose(line, "starting dose")
    evidence.target_dose, evidence.target_value, _ = _parse_labeled_single_dose(line, "target dose")
    if "k" in text and ">5.0" in text:
        evidence.caution_potassium_gt = 5.0
    if "k" in text and "above 5.5" in text:
        evidence.stop_potassium_gt = 5.5
        evidence.halve_on_excess = True
    if "k" in text and ">6.0" in text:
        evidence.stop_potassium_gt = 6.0
    if "2.5 mg/dl" in text:
        evidence.caution_creatinine_gt = 2.5
    if "3.5 mg/dl" in text:
        evidence.stop_creatinine_gt = 3.5
    if "egfr <30" in text:
        evidence.caution_egfr_lt = 30.0
        evidence.halve_on_excess = True
    if "egfr <20" in text:
        evidence.stop_egfr_lt = 20.0
    return evidence


def _extract_beta_blocker_evidence(contexts: list[RetrievedContext], snapshot: dict[str, Any]) -> DrugEvidence:
    agent, _ = _agent_or_default(snapshot.get("beta_blocker_agent"), _default_agent("beta_blocker"))
    evidence = _base_evidence("beta_blocker", agent, contexts)
    line = _find_agent_line(contexts, agent)
    evidence.starting_dose, evidence.start_low_value, evidence.target_frequency = _parse_labeled_single_dose(line, "starting dose")
    evidence.target_dose, evidence.target_value, _ = _parse_labeled_single_dose(line, "target dose")
    text = _combined_text(contexts)
    evidence.double_uptitration = "double the dose" in text
    if "heart rate <50" in text or "if <50 b.p.m." in text:
        evidence.caution_hr_lt = 50.0
    if "sbp <90" in text:
        evidence.caution_sbp_lt = 90.0
    evidence.requires_euvolemia_before_start = "achieve ‘euvolaemia’ before starting" in text or "achieve 'euvolaemia' before starting" in text or "relieve congestion and achieve" in text
    evidence.halve_on_excess = "halve the dose of beta-blocker" in text or "halve the dose of beta blocker" in text
    return evidence


def _extract_ras_evidence(contexts: list[RetrievedContext], snapshot: dict[str, Any]) -> DrugEvidence:
    agent, _ = _agent_or_default(snapshot.get("ras_agent"), _default_agent("ras"))
    evidence = _base_evidence("ras", agent, contexts)
    line = _find_agent_line(contexts, agent)
    evidence.starting_dose, evidence.start_low_value, evidence.target_frequency = _parse_labeled_single_dose(line, "starting dose")
    evidence.target_dose, evidence.target_value, _ = _parse_labeled_target_dose(line)
    text = _combined_text(contexts)
    evidence.double_uptitration = "double the dose" in text
    if "significant hyperkalaemia (k" in text and ">5.0" in text:
        evidence.caution_potassium_gt = 5.0
    if "k" in text and ">5.5 mmol/l" in text:
        evidence.stop_potassium_gt = 5.5
        evidence.halve_on_excess = True
    if "2.5 mg/dl" in text:
        evidence.caution_creatinine_gt = 2.5
    if "3 mg/dl" in text:
        evidence.stop_creatinine_gt = 3.0
        evidence.halve_on_excess = True
    if "3.5 mg/dl" in text:
        evidence.stop_creatinine_gt = 3.5
    if "egfr <30" in text:
        evidence.caution_egfr_lt = 30.0
    if "egfr <25" in text:
        evidence.stop_egfr_lt = 25.0
        evidence.halve_on_excess = True
    if "egfr <20" in text:
        evidence.stop_egfr_lt = 20.0
    if "sbp <90" in text:
        evidence.caution_sbp_lt = 90.0
    return evidence


def _extract_arni_evidence(contexts: list[RetrievedContext], snapshot: dict[str, Any]) -> DrugEvidence:
    evidence = _base_evidence("arni", "sacubitril/valsartan", contexts)
    text = _combined_text(contexts)
    evidence.starting_dose, evidence.start_low_value, evidence.target_frequency = _parse_arni_main_start(text)
    evidence.target_dose, evidence.target_value, _ = _parse_target_combo(text)
    evidence.reduced_start_dose, evidence.reduced_start_value, _ = _parse_reduced_combo(text)
    if ">5.0" in text:
        evidence.caution_potassium_gt = 5.0
    if ">5.5 mmol/l" in text:
        evidence.stop_potassium_gt = 5.5
        evidence.halve_on_excess = True
    if "egfr <30" in text:
        evidence.stop_egfr_lt = 30.0
    if "sbp <90" in text or "sbp >95" in text:
        evidence.caution_sbp_lt = 90.0
    evidence.double_uptitration = "double the dose" in text
    if "36 h" in text:
        evidence.washout_note = "Requires ACE-I washout of at least 36 hours before switching."
    return evidence


def _extract_sglt2_evidence(contexts: list[RetrievedContext], snapshot: dict[str, Any]) -> DrugEvidence:
    agent, _ = _agent_or_default(snapshot.get("sglt2_agent"), _default_agent("sglt2"))
    evidence = _base_evidence("sglt2", agent, contexts)
    line = _find_agent_line(contexts, agent)
    evidence.starting_dose, evidence.start_low_value, evidence.target_frequency = _parse_labeled_single_dose(line, "starting (and target) dose")
    if evidence.starting_dose:
        evidence.target_dose = evidence.starting_dose
        evidence.target_value = evidence.start_low_value
    text = _combined_text(contexts)
    if "egfr <20" in text:
        evidence.stop_egfr_lt = 20.0
    if "sbp <95" in text:
        evidence.caution_sbp_lt = 95.0
    evidence.contraindication_note = "Monitor fluid balance because SGLT2 inhibitors may intensify diuresis when combined with diuretics."
    return evidence


def _extract_loop_evidence(contexts: list[RetrievedContext], snapshot: dict[str, Any]) -> DrugEvidence:
    agent, _ = _agent_or_default(snapshot.get("loop_agent"), _default_agent("loop_diuretic"))
    evidence = _base_evidence("loop_diuretic", agent, contexts)
    line = _find_agent_line(contexts, agent)
    evidence.starting_dose, evidence.start_low_value, evidence.start_high_value = _parse_loop_start_range(line)
    evidence.usual_range = _parse_loop_usual_range(line)
    text = _combined_text(contexts)
    if "egfr <30" in text:
        evidence.caution_egfr_lt = 30.0
    if "2.5 mg/dl" in text:
        evidence.caution_creatinine_gt = 2.5
    if "sbp <90" in text:
        evidence.caution_sbp_lt = 90.0
    evidence.requires_euvolemia_before_start = "not indicated if the patient has never had symptoms or signs of congestion" in text
    return evidence


def _base_evidence(family: str, drug: str | None, contexts: list[RetrievedContext]) -> DrugEvidence:
    evidence = DrugEvidence(family=family, drug=drug)
    seen_ids: set[str] = set()
    seen_pages: set[int] = set()
    for item in contexts:
        if item.chunk_id and item.chunk_id not in seen_ids:
            evidence.source_chunk_ids.append(item.chunk_id)
            seen_ids.add(item.chunk_id)
        if item.page_number is not None and item.page_number not in seen_pages:
            evidence.source_pages.append(item.page_number)
            seen_pages.add(item.page_number)
        snippet = item.snippet.strip()
        if snippet and snippet not in evidence.snippets:
            evidence.snippets.append(snippet[:500])
    return evidence


# ---------- recommendation engine ----------

def _near_upper_threshold(value: float | None, *, caution: float | None, stop: float | None, margin: float) -> bool:
    if value is None:
        return False
    if caution is not None and value >= caution - margin:
        return True
    if stop is not None and value >= stop - margin:
        return True
    return False


def _near_lower_threshold(value: float | None, *, caution: float | None, stop: float | None, margin: float) -> bool:
    if value is None:
        return False
    if caution is not None and value <= caution + margin:
        return True
    if stop is not None and value <= stop + margin:
        return True
    return False


def _triggers_conservative_raas_hold(
    *,
    potassium: float | None,
    creatinine: float | None,
    egfr: float | None,
    sbp: float | None,
    evidence: DrugEvidence,
) -> bool:
    return (
        _near_upper_threshold(potassium, caution=evidence.caution_potassium_gt, stop=evidence.stop_potassium_gt, margin=0.2)
        or _near_upper_threshold(creatinine, caution=evidence.caution_creatinine_gt, stop=evidence.stop_creatinine_gt, margin=0.3)
        or _near_lower_threshold(egfr, caution=evidence.caution_egfr_lt, stop=evidence.stop_egfr_lt, margin=5.0)
        or _near_lower_threshold(sbp, caution=evidence.caution_sbp_lt, stop=None, margin=0.0)
    )


def _recommend_mra(snapshot: dict[str, Any], evidence: DrugEvidence) -> DrugRecommendation:
    agent, assumed = _agent_or_default(snapshot.get("mra_agent"), _default_agent("mra"))
    if not evidence.source_chunk_ids or not evidence.starting_dose:
        return _ungrounded_recommendation("mra", agent, assumed)
    potassium = snapshot.get("potassium")
    creatinine = snapshot.get("creatinine")
    egfr = snapshot.get("egfr")
    current = _parse_numeric_value(snapshot.get("dose_mra_prev"))

    if _exceeds_stop_thresholds(potassium, creatinine, egfr, evidence):
        if current:
            return _grounded_rec("mra", agent, "stop", None, "contraindicated", "Retrieved MRA safety thresholds were crossed.", assumed, evidence)
        return _grounded_rec("mra", agent, "avoid_start", None, "contraindicated", "Retrieved MRA safety thresholds were crossed.", assumed, evidence)
    if _exceeds_caution_thresholds(potassium, creatinine, egfr, evidence) or _triggers_conservative_raas_hold(
        potassium=potassium,
        creatinine=creatinine,
        egfr=egfr,
        sbp=None,
        evidence=evidence,
    ):
        if current:
            dose = evidence.starting_dose if evidence.starting_dose else _format_numeric_dose(current / 2.0, evidence.target_frequency)
            return _grounded_rec("mra", agent, "reduce", dose, "caution", "Retrieved MRA guidance supports halving or holding when potassium/renal thresholds are exceeded or close to stopping thresholds.", assumed, evidence)
        return _grounded_rec("mra", agent, "avoid_start", None, "caution", "Retrieved MRA starting thresholds are not met or are too close to stopping thresholds.", assumed, evidence)
    if not current:
        return _grounded_rec("mra", agent, "start", evidence.starting_dose, "eligible", None, assumed, evidence)
    if evidence.target_value and current + 0.001 < evidence.target_value:
        return _grounded_rec("mra", agent, "increase", evidence.target_dose or evidence.starting_dose, "eligible", None, assumed, evidence)
    return _grounded_rec("mra", agent, "maintain", evidence.target_dose or evidence.starting_dose, "at_target", None, assumed, evidence)


def _recommend_beta_blocker(snapshot: dict[str, Any], evidence: DrugEvidence) -> DrugRecommendation:
    agent, assumed = _agent_or_default(snapshot.get("beta_blocker_agent"), _default_agent("beta_blocker"))
    if not evidence.source_chunk_ids or not evidence.starting_dose:
        return _ungrounded_recommendation("beta_blocker", agent, assumed)
    current = _parse_numeric_value(snapshot.get("dose_bb_prev"))
    sbp = snapshot.get("sbp")
    heart_rate = snapshot.get("heart_rate")
    congestion = bool(snapshot.get("congestion_present"))

    if evidence.caution_hr_lt is not None and _lt(heart_rate, evidence.caution_hr_lt):
        if current:
            return _grounded_rec(
                "beta_blocker",
                agent,
                "reduce",
                _format_numeric_dose(max(current / 2.0, evidence.start_low_value or current / 2.0), evidence.target_frequency),
                "caution",
                "Retrieved beta-blocker guidance says to halve the dose when heart rate is below 50 b.p.m. with clinical concern.",
                assumed,
                evidence,
            )
        return _grounded_rec("beta_blocker", agent, "avoid_start", None, "caution", "Retrieved beta-blocker guidance cautions against starting when heart rate is below 50 b.p.m.", assumed, evidence)
    if evidence.caution_sbp_lt is not None and _lt(sbp, evidence.caution_sbp_lt):
        if current:
            return _grounded_rec("beta_blocker", agent, "reduce", _format_numeric_dose(max(current / 2.0, evidence.start_low_value or current / 2.0), evidence.target_frequency), "caution", "Retrieved beta-blocker guidance cautions against ongoing titration during hypotension.", assumed, evidence)
        return _grounded_rec("beta_blocker", agent, "avoid_start", None, "caution", "Retrieved beta-blocker guidance cautions against starting while hypotensive.", assumed, evidence)
    if not current and evidence.requires_euvolemia_before_start and congestion:
        return _grounded_rec("beta_blocker", agent, "defer_start", None, "caution", "Retrieved beta-blocker guidance advises relieving congestion and restoring euvolaemia before starting.", assumed, evidence)
    if not current:
        return _grounded_rec("beta_blocker", agent, "start", evidence.starting_dose, "eligible", None, assumed, evidence)
    if evidence.target_value and current + 0.001 >= evidence.target_value:
        return _grounded_rec("beta_blocker", agent, "maintain", _format_numeric_dose(current, evidence.target_frequency) or evidence.target_dose, "at_target", None, assumed, evidence)
    if evidence.double_uptitration:
        next_value = min(current * 2.0, evidence.target_value or current * 2.0)
        return _grounded_rec("beta_blocker", agent, "increase", _format_numeric_dose(next_value, evidence.target_frequency), "eligible", None, assumed, evidence)
    return _grounded_rec("beta_blocker", agent, "maintain", _format_numeric_dose(current, evidence.target_frequency), "caution", "No grounded uptitration interval was extracted for this agent.", assumed, evidence)


def _recommend_ras(snapshot: dict[str, Any], evidence: DrugEvidence) -> DrugRecommendation:
    if snapshot.get("dose_arni_prev") not in {None, "", 0, 0.0} or snapshot.get("switch_to_arni"):
        return DrugRecommendation(family="ras", drug=None, action="do_not_combine", recommended_dose=None, status="not_applicable", note="ARNI pathway takes precedence over ACE-I/ARB co-prescription.")
    agent, assumed = _agent_or_default(snapshot.get("ras_agent"), _default_agent("ras"))
    if not evidence.source_chunk_ids or not evidence.starting_dose:
        return _ungrounded_recommendation("ras", agent, assumed)
    current = _parse_numeric_value(snapshot.get("dose_ras_prev"))
    potassium = snapshot.get("potassium")
    creatinine = snapshot.get("creatinine")
    egfr = snapshot.get("egfr")
    sbp = snapshot.get("sbp")

    if _exceeds_ras_stop_thresholds(potassium, creatinine, egfr, evidence):
        if current:
            return _grounded_rec("ras", agent, "stop", None, "contraindicated", "Retrieved ACE-I/ARB stopping thresholds were crossed.", assumed, evidence)
        return _grounded_rec("ras", agent, "avoid_start", None, "contraindicated", "Retrieved ACE-I/ARB starting thresholds were not met.", assumed, evidence)
    if _exceeds_ras_caution_thresholds(potassium, creatinine, egfr, sbp, evidence) or _triggers_conservative_raas_hold(
        potassium=potassium,
        creatinine=creatinine,
        egfr=egfr,
        sbp=sbp,
        evidence=evidence,
    ):
        if current:
            return _grounded_rec("ras", agent, "reduce", _format_numeric_dose(max(current / 2.0, evidence.start_low_value or current / 2.0), evidence.target_frequency), "caution", "Retrieved ACE-I/ARB guidance supports halving or holding when chemistry rises excessively or is close to stopping thresholds.", assumed, evidence)
        return _grounded_rec("ras", agent, "avoid_start", None, "caution", "Retrieved ACE-I/ARB guidance lists potassium/renal/blood-pressure cautions for initiation.", assumed, evidence)
    if not current:
        return _grounded_rec("ras", agent, "start", evidence.starting_dose, "eligible", None, assumed, evidence)
    if evidence.target_value and current + 0.001 >= evidence.target_value:
        return _grounded_rec("ras", agent, "maintain", _format_numeric_dose(current, evidence.target_frequency) or evidence.target_dose, "at_target", None, assumed, evidence)
    if evidence.double_uptitration:
        next_value = min(current * 2.0, evidence.target_value or current * 2.0)
        return _grounded_rec("ras", agent, "increase", _format_numeric_dose(next_value, evidence.target_frequency), "eligible", None, assumed, evidence)
    return _grounded_rec("ras", agent, "maintain", _format_numeric_dose(current, evidence.target_frequency), "caution", "No grounded uptitration interval was extracted for this agent.", assumed, evidence)


def _recommend_arni(snapshot: dict[str, Any], evidence: DrugEvidence) -> DrugRecommendation:
    if not evidence.source_chunk_ids or not evidence.starting_dose:
        return _ungrounded_recommendation("arni", "sacubitril/valsartan", False)
    potassium = snapshot.get("potassium")
    egfr = snapshot.get("egfr")
    sbp = snapshot.get("sbp")
    current_raw = snapshot.get("dose_arni_prev")
    current = _parse_combo_lead_value(current_raw)
    ras_current = snapshot.get("dose_ras_prev") not in {None, "", 0, 0.0}

    if evidence.stop_potassium_gt is not None and _gt(potassium, evidence.stop_potassium_gt):
        if current is not None:
            return _grounded_rec("arni", "sacubitril/valsartan", "stop", None, "contraindicated", "Retrieved ARNI stopping threshold for potassium was crossed.", False, evidence)
        return _grounded_rec("arni", "sacubitril/valsartan", "avoid_start", None, "contraindicated", "Retrieved ARNI starting threshold for potassium was not met.", False, evidence)
    if evidence.stop_egfr_lt is not None and _lt(egfr, evidence.stop_egfr_lt):
        if current is not None:
            return _grounded_rec("arni", "sacubitril/valsartan", "stop", None, "contraindicated", "Retrieved ARNI stopping threshold for eGFR was crossed.", False, evidence)
        return _grounded_rec("arni", "sacubitril/valsartan", "avoid_start", None, "contraindicated", "Retrieved ARNI starting threshold for eGFR was not met.", False, evidence)
    if evidence.caution_sbp_lt is not None and _lt(sbp, evidence.caution_sbp_lt):
        if current is not None:
            return _grounded_rec("arni", "sacubitril/valsartan", "maintain", _combo_from_lead_value(current, evidence), "caution", "Retrieved ARNI guidance cautions against hypotension.", False, evidence)
        return _grounded_rec("arni", "sacubitril/valsartan", "avoid_start", None, "caution", "Retrieved ARNI guidance cautions against starting while hypotensive.", False, evidence)
    if _triggers_conservative_raas_hold(
        potassium=potassium,
        creatinine=None,
        egfr=egfr,
        sbp=None,
        evidence=evidence,
    ):
        if current is not None:
            reduced = evidence.reduced_start_dose or evidence.starting_dose or _combo_from_lead_value(current, evidence)
            return _grounded_rec("arni", "sacubitril/valsartan", "reduce", reduced, "caution", "Retrieved ARNI guidance lists potassium/renal thresholds that are too close to stopping thresholds for escalation.", False, evidence)
        return _grounded_rec("arni", "sacubitril/valsartan", "avoid_start", None, "caution", "Retrieved ARNI guidance lists potassium/renal thresholds that are too close to stopping thresholds for initiation.", False, evidence)
    if current is not None:
        if evidence.stop_potassium_gt is not None and evidence.caution_potassium_gt is not None and _gt(potassium, evidence.caution_potassium_gt):
            reduced = evidence.reduced_start_dose or evidence.starting_dose
            return _grounded_rec("arni", "sacubitril/valsartan", "reduce", reduced, "caution", "Retrieved ARNI guidance supports halving the dose when chemistry rises excessively.", False, evidence)
        if evidence.target_value and current + 0.001 >= evidence.target_value:
            return _grounded_rec("arni", "sacubitril/valsartan", "maintain", _combo_from_lead_value(current, evidence) or evidence.target_dose, "at_target", None, False, evidence)
        next_dose = evidence.target_dose if current >= (evidence.start_low_value or current) else evidence.starting_dose
        if evidence.reduced_start_value is not None and abs(current - evidence.reduced_start_value) < 0.01:
            next_dose = evidence.starting_dose
        elif evidence.start_low_value is not None and abs(current - evidence.start_low_value) < 0.01:
            next_dose = evidence.target_dose
        return _grounded_rec("arni", "sacubitril/valsartan", "increase", next_dose, "eligible", None, False, evidence)

    use_reduced = _between(sbp, 100.0, 110.0) or _between(egfr, 30.0, 60.0)
    dose = evidence.reduced_start_dose if use_reduced and evidence.reduced_start_dose else evidence.starting_dose
    action = "switch" if ras_current else "start"
    note = evidence.washout_note if ras_current else None
    return _grounded_rec("arni", "sacubitril/valsartan", action, dose, "eligible", note, False, evidence)


def _recommend_sglt2(snapshot: dict[str, Any], evidence: DrugEvidence) -> DrugRecommendation:
    agent, assumed = _agent_or_default(snapshot.get("sglt2_agent"), _default_agent("sglt2"))
    if not evidence.source_chunk_ids or not evidence.starting_dose:
        return _ungrounded_recommendation("sglt2", agent, assumed)
    egfr = snapshot.get("egfr")
    sbp = snapshot.get("sbp")
    current = snapshot.get("dose_sglt2_prev") not in {None, "", 0, 0.0}
    if evidence.stop_egfr_lt is not None and _lt(egfr, evidence.stop_egfr_lt):
        return _grounded_rec("sglt2", agent, "avoid_start", None, "contraindicated", "Retrieved SGLT2 eGFR threshold was not met.", assumed, evidence)
    if evidence.caution_sbp_lt is not None and _lt(sbp, evidence.caution_sbp_lt):
        return _grounded_rec("sglt2", agent, "avoid_start", None, "caution", "Retrieved SGLT2 blood-pressure threshold was not met.", assumed, evidence)
    if current:
        return _grounded_rec("sglt2", agent, "maintain", evidence.starting_dose, "at_target", evidence.contraindication_note, assumed, evidence)
    return _grounded_rec("sglt2", agent, "start", evidence.starting_dose, "eligible", evidence.contraindication_note, assumed, evidence)


def _recommend_loop(snapshot: dict[str, Any], evidence: DrugEvidence) -> DrugRecommendation:
    agent, assumed = _agent_or_default(snapshot.get("loop_agent"), _default_agent("loop_diuretic"))
    if not evidence.source_chunk_ids or evidence.start_low_value is None:
        return _ungrounded_recommendation("loop_diuretic", agent, assumed)
    current = _parse_numeric_value(snapshot.get("dose_loop_prev"))
    congestion = bool(snapshot.get("congestion_present")) or _gt(snapshot.get("weight_gain_kg"), 1.5)
    volume_depleted = bool(snapshot.get("volume_depleted"))
    sbp = snapshot.get("sbp")

    if volume_depleted or ((evidence.caution_sbp_lt is not None and _lt(sbp, evidence.caution_sbp_lt)) and not congestion):
        if current:
            reduced = max(current / 2.0, evidence.start_low_value)
            return _grounded_rec("loop_diuretic", agent, "reduce", _format_numeric_dose(reduced, "daily"), "caution", "Retrieved diuretic guidance supports reducing the dose in hypovolaemia or hypotension without congestion.", assumed, evidence)
        return _grounded_rec("loop_diuretic", agent, "avoid_start", None, "caution", "Retrieved diuretic guidance does not support initiation without congestion or during volume depletion.", assumed, evidence)
    if not congestion and not current:
        return _grounded_rec("loop_diuretic", agent, "no_change", None, "not_indicated", "Retrieved diuretic guidance says loop diuretics are not indicated without symptoms or signs of congestion.", assumed, evidence)
    if not current:
        start_value = evidence.start_high_value if congestion else evidence.start_low_value
        return _grounded_rec("loop_diuretic", agent, "start", _format_numeric_dose(start_value, "daily"), "eligible", None, assumed, evidence)
    return _grounded_rec("loop_diuretic", agent, "maintain", _format_numeric_dose(current, "daily"), "maintain", "Retrieved diuretic guidance emphasizes titrating to congestion and volume status rather than a fixed target.", assumed, evidence)


def _ungrounded_recommendation(family: str, drug: str | None, assumed_agent: bool) -> DrugRecommendation:
    return DrugRecommendation(
        family=family,
        drug=drug,
        action="unavailable",
        recommended_dose=None,
        status="ungrounded",
        note="No retrieved guideline evidence was available for this drug family.",
        assumed_agent=assumed_agent,
        grounded=False,
    )


def _grounded_rec(
    family: str,
    drug: str | None,
    action: str,
    dose: str | None,
    status: str,
    note: str | None,
    assumed: bool,
    evidence: DrugEvidence,
) -> DrugRecommendation:
    tradeoff = None
    if status in {"caution", "contraindicated"}:
        tradeoff = note
    return DrugRecommendation(
        family=family,
        drug=drug,
        action=action,
        recommended_dose=dose,
        status=status,
        note=note,
        tradeoff=tradeoff,
        assumed_agent=assumed,
        grounded=bool(evidence.source_chunk_ids),
        evidence_chunk_ids=list(evidence.source_chunk_ids),
        evidence_pages=list(evidence.source_pages),
    )


def _build_tradeoff_notes(recommendations: dict[str, DrugRecommendation]) -> list[str]:
    notes: list[str] = []
    for item in recommendations.values():
        if item.tradeoff:
            notes.append(f"{item.family}: {item.tradeoff}")
    return notes


def _build_safety_cautions(recommendations: dict[str, DrugRecommendation]) -> list[dict[str, Any]]:
    cautions: list[dict[str, Any]] = []
    for item in recommendations.values():
        if not item.grounded:
            continue
        if item.status not in {"caution", "contraindicated"}:
            continue
        cautions.append(
            {
                "family": item.family,
                "drug": item.drug,
                "action": item.action,
                "status": item.status,
                "note": item.note,
                "evidence_chunk_ids": list(item.evidence_chunk_ids),
            }
        )
    return cautions


def _select_visible_recommendations(
    recommendations: dict[str, DrugRecommendation],
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    current_arni = snapshot.get("dose_arni_prev") not in {None, "", 0, 0.0}
    arni_requested = bool(snapshot.get("switch_to_arni")) or current_arni
    visible: list[DrugRecommendation] = []
    for family, recommendation in recommendations.items():
        if family == "ras" and arni_requested:
            continue
        if family == "arni" and not (arni_requested or recommendation.action == "start"):
            # allow ARNI only when already requested/current or when it is surfaced as a replacement strategy
            continue
        if recommendation.action not in {"switch", "start", "increase"}:
            continue
        if recommendation.status != "eligible":
            continue
        if not recommendation.grounded or not recommendation.evidence_chunk_ids:
            continue
        if family == "loop_diuretic" and recommendation.action == "maintain":
            # keep loop maintenance out of the shortlist unless nothing else is actionable
            continue
        visible.append(recommendation)

    visible.sort(key=_recommendation_sort_key)
    if not visible:
        fallback = recommendations.get("loop_diuretic")
        if fallback and fallback.grounded and fallback.status == "eligible" and fallback.recommended_dose and fallback.action in {"start", "increase"}:
            visible = [fallback]
    visible = visible[:MAX_VISIBLE_RECOMMENDATIONS]
    return [
        {
            "family": recommendation.family,
            "drug": recommendation.drug,
            "dose": recommendation.recommended_dose,
            "action": recommendation.action,
            "status": recommendation.status,
            "grounded": recommendation.grounded,
            "evidence_chunk_ids": recommendation.evidence_chunk_ids,
            "evidence_pages": recommendation.evidence_pages,
        }
        for recommendation in visible
    ]


def _recommendation_sort_key(recommendation: DrugRecommendation) -> tuple[int, int, str]:
    return (
        ACTION_PRIORITY.get(recommendation.action, 99),
        _family_priority().get(recommendation.family, 99),
        recommendation.drug or recommendation.family,
    )


def _render_visible_recommendation(item: dict[str, Any]) -> str:
    drug = item.get("drug") or item.get("family") or "drug"
    dose = item.get("dose")
    return f"{drug}: {dose}" if dose else drug


def select_grounded_rag_context(
    retrieved_context: list[RetrievedContext],
    payload: dict[str, Any],
    *,
    max_items: int = 12,
) -> list[RetrievedContext]:
    relevant_ids: list[str] = []
    for recommendation in (payload.get("recommendations") or {}).values():
        if recommendation.get("grounded") and recommendation.get("evidence_chunk_ids"):
            relevant_ids.extend(recommendation.get("evidence_chunk_ids", []))
    seen_needed: set[str] = set()
    ordered_ids: list[str] = []
    for chunk_id in relevant_ids:
        if chunk_id and chunk_id not in seen_needed:
            seen_needed.add(chunk_id)
            ordered_ids.append(chunk_id)
    selected: list[RetrievedContext] = []
    selected_ids: set[str] = set()
    for chunk_id in ordered_ids:
        for item in retrieved_context:
            if item.chunk_id == chunk_id and chunk_id not in selected_ids:
                selected.append(item)
                selected_ids.add(chunk_id)
                break
        if len(selected) >= max_items:
            break
    if selected:
        return selected
    return retrieved_context[:max_items]


# ---------- parsing helpers ----------

def _combined_text(contexts: Iterable[RetrievedContext]) -> str:
    return "\n".join(f"{item.title} {item.snippet}".lower() for item in contexts)


def _find_agent_line(contexts: list[RetrievedContext], agent: str) -> str:
    normalized_agent = agent.lower().replace("_", " ")
    for item in contexts:
        for line in item.snippet.splitlines():
            lowered = line.lower()
            if normalized_agent in lowered:
                return line.strip()
    return "\n".join(item.snippet for item in contexts)


def _parse_labeled_single_dose(line: str, label: str) -> tuple[str | None, float | None, str | None]:
    if not line:
        return None, None, None
    pattern = re.compile(rf"{re.escape(label)}\s+(\d+(?:\.\d+)?)\s*mg\s*([a-z.]+)", re.IGNORECASE)
    match = pattern.search(line)
    if not match:
        return None, None, None
    value = float(match.group(1))
    frequency = match.group(2).strip().lower()
    return _format_numeric_dose(value, frequency), value, frequency


def _parse_labeled_target_dose(line: str) -> tuple[str | None, float | None, str | None]:
    if not line:
        return None, None, None
    pattern = re.compile(r"target dose\s+(\d+(?:\.\d+)?)(?:\s*[–-]\s*(\d+(?:\.\d+)?))?\s*mg\s*([a-z.]+)", re.IGNORECASE)
    match = pattern.search(line)
    if not match:
        return None, None, None
    first = float(match.group(1))
    second = float(match.group(2)) if match.group(2) else first
    frequency = match.group(3).strip().lower()
    target = max(first, second)
    label = f"{_trim_numeric(first)}-{_trim_numeric(second)} mg {frequency}" if second != first else f"{_trim_numeric(target)} mg {frequency}"
    return label, target, frequency


def _parse_arni_main_start(text: str) -> tuple[str | None, float | None, str | None]:
    match = re.search(r"starting dose\s+(\d+)/(\d+)\s*mg\s*([a-z.]+)", text, re.IGNORECASE)
    if not match:
        return None, None, None
    return f"{match.group(1)}/{match.group(2)} mg {match.group(3).lower()}", float(match.group(1)), match.group(3).lower()


def _parse_target_combo(text: str) -> tuple[str | None, float | None, str | None]:
    match = re.search(r"target dose\s+(\d+)/(\d+)\s*mg\s*([a-z.]+)", text, re.IGNORECASE)
    if not match:
        return None, None, None
    return f"{match.group(1)}/{match.group(2)} mg {match.group(3).lower()}", float(match.group(1)), match.group(3).lower()


def _parse_reduced_combo(text: str) -> tuple[str | None, float | None, str | None]:
    match = re.search(r"(24/26\s*mg\s*[a-z.]+)\s+in selected patients", text, re.IGNORECASE)
    if match:
        dose_text = match.group(1).lower().replace("  ", " ")
        return dose_text, 24.0, dose_text.rsplit(" ", 1)[-1]
    match = re.search(r"reduced starting dose\s*\(?24/26\s*mg\s*([a-z.]+)\)?", text, re.IGNORECASE)
    if match:
        return f"24/26 mg {match.group(1).lower()}", 24.0, match.group(1).lower()
    return None, None, None


def _parse_loop_start_range(line: str) -> tuple[str | None, float | None, float | None]:
    if not line:
        return None, None, None
    match = re.search(r"starting dose\s*(\d+(?:\.\d+)?)\s*[–-]?\s*(\d+(?:\.\d+)?)?\s*mg", line, re.IGNORECASE)
    if not match:
        return None, None, None
    low = float(match.group(1))
    high = float(match.group(2)) if match.group(2) else low
    label = f"{_trim_numeric(low)}-{_trim_numeric(high)} mg daily" if high != low else f"{_trim_numeric(low)} mg daily"
    return label, low, high


def _parse_loop_usual_range(line: str) -> str | None:
    if not line:
        return None
    match = re.search(r"usual dose\s*(\d+(?:\.\d+)?)\s*[–-]?\s*(\d+(?:\.\d+)?)?\s*mg", line, re.IGNORECASE)
    if not match:
        return None
    low = float(match.group(1))
    high = float(match.group(2)) if match.group(2) else low
    return f"{_trim_numeric(low)}-{_trim_numeric(high)} mg daily" if high != low else f"{_trim_numeric(low)} mg daily"


def _parse_combo_lead_value(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("mg", "")
    match = re.search(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return _parse_numeric_value(value)


def _combo_from_lead_value(value: float | None, evidence: DrugEvidence) -> str | None:
    if value is None:
        return None
    mapping = {
        evidence.reduced_start_value: evidence.reduced_start_dose,
        evidence.start_low_value: evidence.starting_dose,
        evidence.target_value: evidence.target_dose,
    }
    for key, dose in mapping.items():
        if key is not None and dose and abs(value - key) < 0.01:
            return dose
    return None


# ---------- numeric helpers ----------

def _first_raw(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
        for existing_key, value in payload.items():
            if existing_key.lower() == key.lower():
                return value
    return None


def _first_float(payload: dict[str, Any], *keys: str) -> float | None:
    return _parse_numeric_value(_first_raw(payload, *keys))


def _first_str(payload: dict[str, Any], *keys: str) -> str | None:
    value = _first_raw(payload, *keys)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_bool(payload: dict[str, Any], *keys: str) -> bool | None:
    value = _first_raw(payload, *keys)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "present", "positive"}:
        return True
    if text in {"0", "false", "no", "n", "absent", "negative"}:
        return False
    return None


def _normalize_agent(value: str | None, default: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "metoprolol": "metoprolol_succinate",
        "metoprolol_cr_xl": "metoprolol_succinate",
        "metoprolol_succinate_(cr/xl)": "metoprolol_succinate",
        "sacubitril_valsartan": "sacubitril/valsartan",
        "sac_val": "sacubitril/valsartan",
        "spironolacton": "spironolactone",
        "mra": default,
        "bb": default,
        "loop": default,
    }
    return aliases.get(normalized, normalized)


def _agent_or_default(value: str | None, default: str) -> tuple[str, bool]:
    if value:
        if value in {"sacubitril/valsartan", "sacubitril_valsartan"}:
            return "sacubitril/valsartan", False
        return value, False
    return default, True


def _parse_numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return numeric
    text = str(value).strip().lower().replace("mg", "")
    if not text:
        return None
    if "/" in text:
        left = text.split("/", 1)[0].strip()
        try:
            return float(left)
        except ValueError:
            return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _format_numeric_dose(value: float | None, frequency: str | None) -> str | None:
    if value is None:
        return None
    freq = (frequency or "").strip()
    if not freq:
        return f"{_trim_numeric(value)} mg"
    return f"{_trim_numeric(value)} mg {freq}"


def _trim_numeric(value: float) -> str:
    if abs(value - round(value)) < 0.001:
        return str(int(round(value)))
    return (f"{value:.3f}").rstrip("0").rstrip(".")


def _gt(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def _lt(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold


def _between(value: float | None, lower: float, upper: float) -> bool:
    return value is not None and lower <= value <= upper


def _exceeds_stop_thresholds(potassium: float | None, creatinine: float | None, egfr: float | None, evidence: DrugEvidence) -> bool:
    return (
        (evidence.stop_potassium_gt is not None and _gt(potassium, evidence.stop_potassium_gt))
        or (evidence.stop_creatinine_gt is not None and _gt(creatinine, evidence.stop_creatinine_gt))
        or (evidence.stop_egfr_lt is not None and _lt(egfr, evidence.stop_egfr_lt))
    )


def _exceeds_caution_thresholds(potassium: float | None, creatinine: float | None, egfr: float | None, evidence: DrugEvidence) -> bool:
    return (
        (evidence.caution_potassium_gt is not None and _gt(potassium, evidence.caution_potassium_gt))
        or (evidence.caution_creatinine_gt is not None and _gt(creatinine, evidence.caution_creatinine_gt))
        or (evidence.caution_egfr_lt is not None and _lt(egfr, evidence.caution_egfr_lt))
    )


def _exceeds_ras_stop_thresholds(potassium: float | None, creatinine: float | None, egfr: float | None, evidence: DrugEvidence) -> bool:
    return (
        (evidence.stop_potassium_gt is not None and _gt(potassium, evidence.stop_potassium_gt))
        or (evidence.stop_creatinine_gt is not None and _gt(creatinine, evidence.stop_creatinine_gt))
        or (evidence.stop_egfr_lt is not None and _lt(egfr, evidence.stop_egfr_lt))
    )


def _exceeds_ras_caution_thresholds(
    potassium: float | None,
    creatinine: float | None,
    egfr: float | None,
    sbp: float | None,
    evidence: DrugEvidence,
) -> bool:
    return (
        (evidence.caution_potassium_gt is not None and _gt(potassium, evidence.caution_potassium_gt))
        or (evidence.caution_creatinine_gt is not None and _gt(creatinine, evidence.caution_creatinine_gt))
        or (evidence.caution_egfr_lt is not None and _lt(egfr, evidence.caution_egfr_lt))
        or (evidence.caution_sbp_lt is not None and _lt(sbp, evidence.caution_sbp_lt))
    )
