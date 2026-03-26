from __future__ import annotations

from typing import Any, Callable

from shared.contracts.inference import RetrievedContext

from .models import DrugEvidence
from .parsing import (
    agent_or_default,
    combined_text,
    find_agent_line,
    parse_arni_main_start,
    parse_labeled_single_dose,
    parse_labeled_target_dose,
    parse_loop_start_range,
    parse_loop_usual_range,
    parse_reduced_combo,
    parse_target_combo,
)

FamilyQueryOrder = Callable[[], tuple[str, ...]]
FamilyKeywordsResolver = Callable[[], dict[str, set[str]]]
DefaultAgentResolver = Callable[[str], str]


def extract_grounded_drug_evidence(
    *,
    retrieved_context: list[RetrievedContext],
    snapshot: dict[str, Any],
    family_contexts: dict[str, list[RetrievedContext]] | None = None,
    family_query_order: FamilyQueryOrder,
    family_keywords: FamilyKeywordsResolver,
    default_agent: DefaultAgentResolver,
) -> dict[str, DrugEvidence]:
    grouped = family_contexts or group_context_by_family(
        retrieved_context,
        family_query_order=family_query_order,
        family_keywords=family_keywords,
    )
    return {
        "mra": extract_mra_evidence(grouped.get("mra", []), snapshot, default_agent=default_agent),
        "beta_blocker": extract_beta_blocker_evidence(grouped.get("beta_blocker", []), snapshot, default_agent=default_agent),
        "ras": extract_ras_evidence(grouped.get("ras", []), snapshot, default_agent=default_agent),
        "arni": extract_arni_evidence(grouped.get("arni", []), snapshot),
        "sglt2": extract_sglt2_evidence(grouped.get("sglt2", []), snapshot, default_agent=default_agent),
        "loop_diuretic": extract_loop_evidence(grouped.get("loop_diuretic", []), snapshot, default_agent=default_agent),
    }


def group_context_by_family(
    retrieved_context: list[RetrievedContext],
    *,
    family_query_order: FamilyQueryOrder,
    family_keywords: FamilyKeywordsResolver,
) -> dict[str, list[RetrievedContext]]:
    grouped = {family: [] for family in family_query_order()}
    for item in retrieved_context:
        text = f"{item.title} {item.snippet}".lower()
        for family, keywords in family_keywords().items():
            if any(keyword in text for keyword in keywords):
                grouped[family].append(item)
    return grouped


def extract_mra_evidence(contexts: list[RetrievedContext], snapshot: dict[str, Any], *, default_agent: DefaultAgentResolver) -> DrugEvidence:
    agent, _ = agent_or_default(snapshot.get("mra_agent"), default_agent("mra"))
    evidence = base_evidence("mra", agent, contexts)
    text = combined_text(contexts)
    line = find_agent_line(contexts, agent)
    evidence.starting_dose, evidence.start_low_value, evidence.target_frequency = parse_labeled_single_dose(line, "starting dose")
    evidence.target_dose, evidence.target_value, _ = parse_labeled_single_dose(line, "target dose")
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


def extract_beta_blocker_evidence(
    contexts: list[RetrievedContext],
    snapshot: dict[str, Any],
    *,
    default_agent: DefaultAgentResolver,
) -> DrugEvidence:
    agent, _ = agent_or_default(snapshot.get("beta_blocker_agent"), default_agent("beta_blocker"))
    evidence = base_evidence("beta_blocker", agent, contexts)
    line = find_agent_line(contexts, agent)
    evidence.starting_dose, evidence.start_low_value, evidence.target_frequency = parse_labeled_single_dose(line, "starting dose")
    evidence.target_dose, evidence.target_value, _ = parse_labeled_single_dose(line, "target dose")
    text = combined_text(contexts)
    evidence.double_uptitration = "double the dose" in text
    if "heart rate <50" in text or "if <50 b.p.m." in text:
        evidence.caution_hr_lt = 50.0
    if "sbp <90" in text:
        evidence.caution_sbp_lt = 90.0
    evidence.requires_euvolemia_before_start = (
        "achieve ‘euvolaemia’ before starting" in text
        or "achieve 'euvolaemia' before starting" in text
        or "relieve congestion and achieve" in text
    )
    evidence.halve_on_excess = "halve the dose of beta-blocker" in text or "halve the dose of beta blocker" in text
    return evidence


def extract_ras_evidence(contexts: list[RetrievedContext], snapshot: dict[str, Any], *, default_agent: DefaultAgentResolver) -> DrugEvidence:
    agent, _ = agent_or_default(snapshot.get("ras_agent"), default_agent("ras"))
    evidence = base_evidence("ras", agent, contexts)
    line = find_agent_line(contexts, agent)
    evidence.starting_dose, evidence.start_low_value, evidence.target_frequency = parse_labeled_single_dose(line, "starting dose")
    evidence.target_dose, evidence.target_value, _ = parse_labeled_target_dose(line)
    text = combined_text(contexts)
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


def extract_arni_evidence(contexts: list[RetrievedContext], snapshot: dict[str, Any]) -> DrugEvidence:
    del snapshot
    evidence = base_evidence("arni", "sacubitril/valsartan", contexts)
    text = combined_text(contexts)
    evidence.starting_dose, evidence.start_low_value, evidence.target_frequency = parse_arni_main_start(text)
    evidence.target_dose, evidence.target_value, _ = parse_target_combo(text)
    evidence.reduced_start_dose, evidence.reduced_start_value, _ = parse_reduced_combo(text)
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


def extract_sglt2_evidence(contexts: list[RetrievedContext], snapshot: dict[str, Any], *, default_agent: DefaultAgentResolver) -> DrugEvidence:
    agent, _ = agent_or_default(snapshot.get("sglt2_agent"), default_agent("sglt2"))
    evidence = base_evidence("sglt2", agent, contexts)
    line = find_agent_line(contexts, agent)
    evidence.starting_dose, evidence.start_low_value, evidence.target_frequency = parse_labeled_single_dose(line, "starting (and target) dose")
    if evidence.starting_dose:
        evidence.target_dose = evidence.starting_dose
        evidence.target_value = evidence.start_low_value
    text = combined_text(contexts)
    if "egfr <20" in text:
        evidence.stop_egfr_lt = 20.0
    if "sbp <95" in text:
        evidence.caution_sbp_lt = 95.0
    evidence.contraindication_note = "Monitor fluid balance because SGLT2 inhibitors may intensify diuresis when combined with diuretics."
    return evidence


def extract_loop_evidence(contexts: list[RetrievedContext], snapshot: dict[str, Any], *, default_agent: DefaultAgentResolver) -> DrugEvidence:
    agent, _ = agent_or_default(snapshot.get("loop_agent"), default_agent("loop_diuretic"))
    evidence = base_evidence("loop_diuretic", agent, contexts)
    line = find_agent_line(contexts, agent)
    evidence.starting_dose, evidence.start_low_value, evidence.start_high_value = parse_loop_start_range(line)
    evidence.usual_range = parse_loop_usual_range(line)
    text = combined_text(contexts)
    if "egfr <30" in text:
        evidence.caution_egfr_lt = 30.0
    if "2.5 mg/dl" in text:
        evidence.caution_creatinine_gt = 2.5
    if "sbp <90" in text:
        evidence.caution_sbp_lt = 90.0
    evidence.requires_euvolemia_before_start = "not indicated if the patient has never had symptoms or signs of congestion" in text
    return evidence


def base_evidence(family: str, drug: str | None, contexts: list[RetrievedContext]) -> DrugEvidence:
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
