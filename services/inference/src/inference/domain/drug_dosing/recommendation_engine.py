from __future__ import annotations

from typing import Any, Callable

from .models import DrugEvidence, DrugRecommendation
from .parsing import between, combo_from_lead_value, format_numeric_dose, gt, lt, parse_combo_lead_value, parse_numeric_value

MAX_VISIBLE_RECOMMENDATIONS = 3
ACTION_PRIORITY = {
    "switch": 0,
    "start": 1,
    "increase": 2,
    "maintain": 3,
}

DefaultAgentResolver = Callable[[str], str]
FamilyPriorityResolver = Callable[[], dict[str, int]]


def build_recommendations(
    snapshot: dict[str, Any],
    evidence: dict[str, DrugEvidence],
    *,
    default_agent: DefaultAgentResolver,
) -> dict[str, DrugRecommendation]:
    return {
        "mra": recommend_mra(snapshot, evidence["mra"], default_agent=default_agent),
        "beta_blocker": recommend_beta_blocker(snapshot, evidence["beta_blocker"], default_agent=default_agent),
        "ras": recommend_ras(snapshot, evidence["ras"], default_agent=default_agent),
        "arni": recommend_arni(snapshot, evidence["arni"]),
        "sglt2": recommend_sglt2(snapshot, evidence["sglt2"], default_agent=default_agent),
        "loop_diuretic": recommend_loop(snapshot, evidence["loop_diuretic"], default_agent=default_agent),
    }


def near_upper_threshold(value: float | None, *, caution: float | None, stop: float | None, margin: float) -> bool:
    if value is None:
        return False
    if caution is not None and value >= caution - margin:
        return True
    if stop is not None and value >= stop - margin:
        return True
    return False


def near_lower_threshold(value: float | None, *, caution: float | None, stop: float | None, margin: float) -> bool:
    if value is None:
        return False
    if caution is not None and value <= caution + margin:
        return True
    if stop is not None and value <= stop + margin:
        return True
    return False


def triggers_conservative_raas_hold(
    *,
    potassium: float | None,
    creatinine: float | None,
    egfr: float | None,
    sbp: float | None,
    evidence: DrugEvidence,
) -> bool:
    return (
        near_upper_threshold(potassium, caution=evidence.caution_potassium_gt, stop=evidence.stop_potassium_gt, margin=0.2)
        or near_upper_threshold(creatinine, caution=evidence.caution_creatinine_gt, stop=evidence.stop_creatinine_gt, margin=0.3)
        or near_lower_threshold(egfr, caution=evidence.caution_egfr_lt, stop=evidence.stop_egfr_lt, margin=5.0)
        or near_lower_threshold(sbp, caution=evidence.caution_sbp_lt, stop=None, margin=0.0)
    )


def recommend_mra(snapshot: dict[str, Any], evidence: DrugEvidence, *, default_agent: DefaultAgentResolver) -> DrugRecommendation:
    agent, assumed = agent_or_default(snapshot.get("mra_agent"), default_agent("mra"))
    if not evidence.source_chunk_ids or not evidence.starting_dose:
        return ungrounded_recommendation("mra", agent, assumed)
    potassium = snapshot.get("potassium")
    creatinine = snapshot.get("creatinine")
    egfr = snapshot.get("egfr")
    current = parse_numeric_value(snapshot.get("dose_mra_prev"))

    if exceeds_stop_thresholds(potassium, creatinine, egfr, evidence):
        if current:
            return grounded_rec("mra", agent, "stop", None, "contraindicated", "Retrieved MRA safety thresholds were crossed.", assumed, evidence)
        return grounded_rec("mra", agent, "avoid_start", None, "contraindicated", "Retrieved MRA safety thresholds were crossed.", assumed, evidence)
    if exceeds_caution_thresholds(potassium, creatinine, egfr, evidence) or triggers_conservative_raas_hold(
        potassium=potassium,
        creatinine=creatinine,
        egfr=egfr,
        sbp=None,
        evidence=evidence,
    ):
        if current:
            dose = evidence.starting_dose if evidence.starting_dose else format_numeric_dose(current / 2.0, evidence.target_frequency)
            return grounded_rec(
                "mra",
                agent,
                "reduce",
                dose,
                "caution",
                "Retrieved MRA guidance supports halving or holding when potassium/renal thresholds are exceeded or close to stopping thresholds.",
                assumed,
                evidence,
            )
        return grounded_rec("mra", agent, "avoid_start", None, "caution", "Retrieved MRA starting thresholds are not met or are too close to stopping thresholds.", assumed, evidence)
    if not current:
        return grounded_rec("mra", agent, "start", evidence.starting_dose, "eligible", None, assumed, evidence)
    if evidence.target_value and current + 0.001 < evidence.target_value:
        return grounded_rec("mra", agent, "increase", evidence.target_dose or evidence.starting_dose, "eligible", None, assumed, evidence)
    return grounded_rec("mra", agent, "maintain", evidence.target_dose or evidence.starting_dose, "at_target", None, assumed, evidence)


def recommend_beta_blocker(snapshot: dict[str, Any], evidence: DrugEvidence, *, default_agent: DefaultAgentResolver) -> DrugRecommendation:
    agent, assumed = agent_or_default(snapshot.get("beta_blocker_agent"), default_agent("beta_blocker"))
    if not evidence.source_chunk_ids or not evidence.starting_dose:
        return ungrounded_recommendation("beta_blocker", agent, assumed)
    current = parse_numeric_value(snapshot.get("dose_bb_prev"))
    sbp = snapshot.get("sbp")
    heart_rate = snapshot.get("heart_rate")
    congestion = bool(snapshot.get("congestion_present"))

    if evidence.caution_hr_lt is not None and lt(heart_rate, evidence.caution_hr_lt):
        if current:
            return grounded_rec(
                "beta_blocker",
                agent,
                "reduce",
                format_numeric_dose(max(current / 2.0, evidence.start_low_value or current / 2.0), evidence.target_frequency),
                "caution",
                "Retrieved beta-blocker guidance says to halve the dose when heart rate is below 50 b.p.m. with clinical concern.",
                assumed,
                evidence,
            )
        return grounded_rec("beta_blocker", agent, "avoid_start", None, "caution", "Retrieved beta-blocker guidance cautions against starting when heart rate is below 50 b.p.m.", assumed, evidence)
    if evidence.caution_sbp_lt is not None and lt(sbp, evidence.caution_sbp_lt):
        if current:
            return grounded_rec(
                "beta_blocker",
                agent,
                "reduce",
                format_numeric_dose(max(current / 2.0, evidence.start_low_value or current / 2.0), evidence.target_frequency),
                "caution",
                "Retrieved beta-blocker guidance cautions against ongoing titration during hypotension.",
                assumed,
                evidence,
            )
        return grounded_rec("beta_blocker", agent, "avoid_start", None, "caution", "Retrieved beta-blocker guidance cautions against starting while hypotensive.", assumed, evidence)
    if not current and evidence.requires_euvolemia_before_start and congestion:
        return grounded_rec("beta_blocker", agent, "defer_start", None, "caution", "Retrieved beta-blocker guidance advises relieving congestion and restoring euvolaemia before starting.", assumed, evidence)
    if not current:
        return grounded_rec("beta_blocker", agent, "start", evidence.starting_dose, "eligible", None, assumed, evidence)
    if evidence.target_value and current + 0.001 >= evidence.target_value:
        return grounded_rec("beta_blocker", agent, "maintain", format_numeric_dose(current, evidence.target_frequency) or evidence.target_dose, "at_target", None, assumed, evidence)
    if evidence.double_uptitration:
        next_value = min(current * 2.0, evidence.target_value or current * 2.0)
        return grounded_rec("beta_blocker", agent, "increase", format_numeric_dose(next_value, evidence.target_frequency), "eligible", None, assumed, evidence)
    return grounded_rec("beta_blocker", agent, "maintain", format_numeric_dose(current, evidence.target_frequency), "caution", "No grounded uptitration interval was extracted for this agent.", assumed, evidence)


def recommend_ras(snapshot: dict[str, Any], evidence: DrugEvidence, *, default_agent: DefaultAgentResolver) -> DrugRecommendation:
    if snapshot.get("dose_arni_prev") not in {None, "", 0, 0.0} or snapshot.get("switch_to_arni"):
        return DrugRecommendation(family="ras", drug=None, action="do_not_combine", recommended_dose=None, status="not_applicable", note="ARNI pathway takes precedence over ACE-I/ARB co-prescription.")
    agent, assumed = agent_or_default(snapshot.get("ras_agent"), default_agent("ras"))
    if not evidence.source_chunk_ids or not evidence.starting_dose:
        return ungrounded_recommendation("ras", agent, assumed)
    current = parse_numeric_value(snapshot.get("dose_ras_prev"))
    potassium = snapshot.get("potassium")
    creatinine = snapshot.get("creatinine")
    egfr = snapshot.get("egfr")
    sbp = snapshot.get("sbp")

    if exceeds_ras_stop_thresholds(potassium, creatinine, egfr, evidence):
        if current:
            return grounded_rec("ras", agent, "stop", None, "contraindicated", "Retrieved ACE-I/ARB stopping thresholds were crossed.", assumed, evidence)
        return grounded_rec("ras", agent, "avoid_start", None, "contraindicated", "Retrieved ACE-I/ARB starting thresholds were not met.", assumed, evidence)
    if exceeds_ras_caution_thresholds(potassium, creatinine, egfr, sbp, evidence) or triggers_conservative_raas_hold(
        potassium=potassium,
        creatinine=creatinine,
        egfr=egfr,
        sbp=sbp,
        evidence=evidence,
    ):
        if current:
            return grounded_rec(
                "ras",
                agent,
                "reduce",
                format_numeric_dose(max(current / 2.0, evidence.start_low_value or current / 2.0), evidence.target_frequency),
                "caution",
                "Retrieved ACE-I/ARB guidance supports halving or holding when chemistry rises excessively or is close to stopping thresholds.",
                assumed,
                evidence,
            )
        return grounded_rec("ras", agent, "avoid_start", None, "caution", "Retrieved ACE-I/ARB guidance lists potassium/renal/blood-pressure cautions for initiation.", assumed, evidence)
    if not current:
        return grounded_rec("ras", agent, "start", evidence.starting_dose, "eligible", None, assumed, evidence)
    if evidence.target_value and current + 0.001 >= evidence.target_value:
        return grounded_rec("ras", agent, "maintain", format_numeric_dose(current, evidence.target_frequency) or evidence.target_dose, "at_target", None, assumed, evidence)
    if evidence.double_uptitration:
        next_value = min(current * 2.0, evidence.target_value or current * 2.0)
        return grounded_rec("ras", agent, "increase", format_numeric_dose(next_value, evidence.target_frequency), "eligible", None, assumed, evidence)
    return grounded_rec("ras", agent, "maintain", format_numeric_dose(current, evidence.target_frequency), "caution", "No grounded uptitration interval was extracted for this agent.", assumed, evidence)


def recommend_arni(snapshot: dict[str, Any], evidence: DrugEvidence) -> DrugRecommendation:
    if not evidence.source_chunk_ids or not evidence.starting_dose:
        return ungrounded_recommendation("arni", "sacubitril/valsartan", False)
    potassium = snapshot.get("potassium")
    egfr = snapshot.get("egfr")
    sbp = snapshot.get("sbp")
    current_raw = snapshot.get("dose_arni_prev")
    current = parse_combo_lead_value(current_raw)
    ras_current = snapshot.get("dose_ras_prev") not in {None, "", 0, 0.0}

    if evidence.stop_potassium_gt is not None and gt(potassium, evidence.stop_potassium_gt):
        if current is not None:
            return grounded_rec("arni", "sacubitril/valsartan", "stop", None, "contraindicated", "Retrieved ARNI stopping threshold for potassium was crossed.", False, evidence)
        return grounded_rec("arni", "sacubitril/valsartan", "avoid_start", None, "contraindicated", "Retrieved ARNI starting threshold for potassium was not met.", False, evidence)
    if evidence.stop_egfr_lt is not None and lt(egfr, evidence.stop_egfr_lt):
        if current is not None:
            return grounded_rec("arni", "sacubitril/valsartan", "stop", None, "contraindicated", "Retrieved ARNI stopping threshold for eGFR was crossed.", False, evidence)
        return grounded_rec("arni", "sacubitril/valsartan", "avoid_start", None, "contraindicated", "Retrieved ARNI starting threshold for eGFR was not met.", False, evidence)
    if evidence.caution_sbp_lt is not None and lt(sbp, evidence.caution_sbp_lt):
        if current is not None:
            return grounded_rec("arni", "sacubitril/valsartan", "maintain", combo_from_lead_value(current, evidence), "caution", "Retrieved ARNI guidance cautions against hypotension.", False, evidence)
        return grounded_rec("arni", "sacubitril/valsartan", "avoid_start", None, "caution", "Retrieved ARNI guidance cautions against starting while hypotensive.", False, evidence)
    if triggers_conservative_raas_hold(
        potassium=potassium,
        creatinine=None,
        egfr=egfr,
        sbp=None,
        evidence=evidence,
    ):
        if current is not None:
            reduced = evidence.reduced_start_dose or evidence.starting_dose or combo_from_lead_value(current, evidence)
            return grounded_rec(
                "arni",
                "sacubitril/valsartan",
                "reduce",
                reduced,
                "caution",
                "Retrieved ARNI guidance lists potassium/renal thresholds that are too close to stopping thresholds for escalation.",
                False,
                evidence,
            )
        return grounded_rec("arni", "sacubitril/valsartan", "avoid_start", None, "caution", "Retrieved ARNI guidance lists potassium/renal thresholds that are too close to stopping thresholds for initiation.", False, evidence)
    if current is not None:
        if evidence.stop_potassium_gt is not None and evidence.caution_potassium_gt is not None and gt(potassium, evidence.caution_potassium_gt):
            reduced = evidence.reduced_start_dose or evidence.starting_dose
            return grounded_rec("arni", "sacubitril/valsartan", "reduce", reduced, "caution", "Retrieved ARNI guidance supports halving the dose when chemistry rises excessively.", False, evidence)
        if evidence.target_value and current + 0.001 >= evidence.target_value:
            return grounded_rec("arni", "sacubitril/valsartan", "maintain", combo_from_lead_value(current, evidence) or evidence.target_dose, "at_target", None, False, evidence)
        next_dose = evidence.target_dose if current >= (evidence.start_low_value or current) else evidence.starting_dose
        if evidence.reduced_start_value is not None and abs(current - evidence.reduced_start_value) < 0.01:
            next_dose = evidence.starting_dose
        elif evidence.start_low_value is not None and abs(current - evidence.start_low_value) < 0.01:
            next_dose = evidence.target_dose
        return grounded_rec("arni", "sacubitril/valsartan", "increase", next_dose, "eligible", None, False, evidence)

    use_reduced = between(sbp, 100.0, 110.0) or between(egfr, 30.0, 60.0)
    dose = evidence.reduced_start_dose if use_reduced and evidence.reduced_start_dose else evidence.starting_dose
    action = "switch" if ras_current else "start"
    note = evidence.washout_note if ras_current else None
    return grounded_rec("arni", "sacubitril/valsartan", action, dose, "eligible", note, False, evidence)


def recommend_sglt2(snapshot: dict[str, Any], evidence: DrugEvidence, *, default_agent: DefaultAgentResolver) -> DrugRecommendation:
    agent, assumed = agent_or_default(snapshot.get("sglt2_agent"), default_agent("sglt2"))
    if not evidence.source_chunk_ids or not evidence.starting_dose:
        return ungrounded_recommendation("sglt2", agent, assumed)
    egfr = snapshot.get("egfr")
    sbp = snapshot.get("sbp")
    current = snapshot.get("dose_sglt2_prev") not in {None, "", 0, 0.0}
    if evidence.stop_egfr_lt is not None and lt(egfr, evidence.stop_egfr_lt):
        return grounded_rec("sglt2", agent, "avoid_start", None, "contraindicated", "Retrieved SGLT2 eGFR threshold was not met.", assumed, evidence)
    if evidence.caution_sbp_lt is not None and lt(sbp, evidence.caution_sbp_lt):
        return grounded_rec("sglt2", agent, "avoid_start", None, "caution", "Retrieved SGLT2 blood-pressure threshold was not met.", assumed, evidence)
    if current:
        return grounded_rec("sglt2", agent, "maintain", evidence.starting_dose, "at_target", evidence.contraindication_note, assumed, evidence)
    return grounded_rec("sglt2", agent, "start", evidence.starting_dose, "eligible", evidence.contraindication_note, assumed, evidence)


def recommend_loop(snapshot: dict[str, Any], evidence: DrugEvidence, *, default_agent: DefaultAgentResolver) -> DrugRecommendation:
    agent, assumed = agent_or_default(snapshot.get("loop_agent"), default_agent("loop_diuretic"))
    if not evidence.source_chunk_ids or evidence.start_low_value is None:
        return ungrounded_recommendation("loop_diuretic", agent, assumed)
    current = parse_numeric_value(snapshot.get("dose_loop_prev"))
    congestion = bool(snapshot.get("congestion_present")) or gt(snapshot.get("weight_gain_kg"), 1.5)
    volume_depleted = bool(snapshot.get("volume_depleted"))
    sbp = snapshot.get("sbp")

    if volume_depleted or ((evidence.caution_sbp_lt is not None and lt(sbp, evidence.caution_sbp_lt)) and not congestion):
        if current:
            reduced = max(current / 2.0, evidence.start_low_value)
            return grounded_rec(
                "loop_diuretic",
                agent,
                "reduce",
                format_numeric_dose(reduced, "daily"),
                "caution",
                "Retrieved diuretic guidance supports reducing the dose in hypovolaemia or hypotension without congestion.",
                assumed,
                evidence,
            )
        return grounded_rec("loop_diuretic", agent, "avoid_start", None, "caution", "Retrieved diuretic guidance does not support initiation without congestion or during volume depletion.", assumed, evidence)
    if not congestion and not current:
        return grounded_rec("loop_diuretic", agent, "no_change", None, "not_indicated", "Retrieved diuretic guidance says loop diuretics are not indicated without symptoms or signs of congestion.", assumed, evidence)
    if not current:
        start_value = evidence.start_high_value if congestion else evidence.start_low_value
        return grounded_rec("loop_diuretic", agent, "start", format_numeric_dose(start_value, "daily"), "eligible", None, assumed, evidence)
    return grounded_rec("loop_diuretic", agent, "maintain", format_numeric_dose(current, "daily"), "maintain", "Retrieved diuretic guidance emphasizes titrating to congestion and volume status rather than a fixed target.", assumed, evidence)


def ungrounded_recommendation(family: str, drug: str | None, assumed_agent: bool) -> DrugRecommendation:
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


def grounded_rec(
    family: str,
    drug: str | None,
    action: str,
    dose: str | None,
    status: str,
    note: str | None,
    assumed: bool,
    evidence: DrugEvidence,
) -> DrugRecommendation:
    tradeoff = note if status in {"caution", "contraindicated"} else None
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


def build_tradeoff_notes(recommendations: dict[str, DrugRecommendation]) -> list[str]:
    return [f"{item.family}: {item.tradeoff}" for item in recommendations.values() if item.tradeoff]


def build_safety_cautions(recommendations: dict[str, DrugRecommendation]) -> list[dict[str, Any]]:
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


def select_visible_recommendations(
    recommendations: dict[str, DrugRecommendation],
    snapshot: dict[str, Any],
    *,
    family_priority: FamilyPriorityResolver,
    max_visible_recommendations: int = MAX_VISIBLE_RECOMMENDATIONS,
) -> list[dict[str, Any]]:
    current_arni = snapshot.get("dose_arni_prev") not in {None, "", 0, 0.0}
    arni_requested = bool(snapshot.get("switch_to_arni")) or current_arni
    visible: list[DrugRecommendation] = []
    for family, recommendation in recommendations.items():
        if family == "ras" and arni_requested:
            continue
        if family == "arni" and not (arni_requested or recommendation.action == "start"):
            continue
        if recommendation.action not in {"switch", "start", "increase"}:
            continue
        if recommendation.status != "eligible":
            continue
        if not recommendation.grounded or not recommendation.evidence_chunk_ids:
            continue
        if family == "loop_diuretic" and recommendation.action == "maintain":
            continue
        visible.append(recommendation)

    visible.sort(key=lambda recommendation: recommendation_sort_key(recommendation, family_priority=family_priority))
    if not visible:
        fallback = recommendations.get("loop_diuretic")
        if fallback and fallback.grounded and fallback.status == "eligible" and fallback.recommended_dose and fallback.action in {"start", "increase"}:
            visible = [fallback]
    visible = visible[:max_visible_recommendations]
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


def recommendation_sort_key(recommendation: DrugRecommendation, *, family_priority: FamilyPriorityResolver) -> tuple[int, int, str]:
    return (
        ACTION_PRIORITY.get(recommendation.action, 99),
        family_priority().get(recommendation.family, 99),
        recommendation.drug or recommendation.family,
    )


def exceeds_stop_thresholds(potassium: float | None, creatinine: float | None, egfr: float | None, evidence: DrugEvidence) -> bool:
    return (
        (evidence.stop_potassium_gt is not None and gt(potassium, evidence.stop_potassium_gt))
        or (evidence.stop_creatinine_gt is not None and gt(creatinine, evidence.stop_creatinine_gt))
        or (evidence.stop_egfr_lt is not None and lt(egfr, evidence.stop_egfr_lt))
    )


def exceeds_caution_thresholds(potassium: float | None, creatinine: float | None, egfr: float | None, evidence: DrugEvidence) -> bool:
    return (
        (evidence.caution_potassium_gt is not None and gt(potassium, evidence.caution_potassium_gt))
        or (evidence.caution_creatinine_gt is not None and gt(creatinine, evidence.caution_creatinine_gt))
        or (evidence.caution_egfr_lt is not None and lt(egfr, evidence.caution_egfr_lt))
    )


def exceeds_ras_stop_thresholds(potassium: float | None, creatinine: float | None, egfr: float | None, evidence: DrugEvidence) -> bool:
    return (
        (evidence.stop_potassium_gt is not None and gt(potassium, evidence.stop_potassium_gt))
        or (evidence.stop_creatinine_gt is not None and gt(creatinine, evidence.stop_creatinine_gt))
        or (evidence.stop_egfr_lt is not None and lt(egfr, evidence.stop_egfr_lt))
    )


def exceeds_ras_caution_thresholds(
    potassium: float | None,
    creatinine: float | None,
    egfr: float | None,
    sbp: float | None,
    evidence: DrugEvidence,
) -> bool:
    return (
        (evidence.caution_potassium_gt is not None and gt(potassium, evidence.caution_potassium_gt))
        or (evidence.caution_creatinine_gt is not None and gt(creatinine, evidence.caution_creatinine_gt))
        or (evidence.caution_egfr_lt is not None and lt(egfr, evidence.caution_egfr_lt))
        or (evidence.caution_sbp_lt is not None and lt(sbp, evidence.caution_sbp_lt))
    )


def agent_or_default(value: str | None, default: str) -> tuple[str, bool]:
    if value:
        if value in {"sacubitril/valsartan", "sacubitril_valsartan"}:
            return "sacubitril/valsartan", False
        return value, False
    return default, True
