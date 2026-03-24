from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow direct script execution from finetuning-experiments/ while importing service modules.
REPO_ROOT = Path(__file__).resolve().parents[2]
for extra_path in (
    REPO_ROOT / "services" / "inference" / "src",
    REPO_ROOT / "services" / "shared" / "src",
    REPO_ROOT / "finetuning-experiments",
):
    extra = str(extra_path)
    if extra not in sys.path:
        sys.path.insert(0, extra)

from datasets.observation import is_observation_only_case
from datasets.schema import BenchmarkCase
from inference.pipeline.support.drug_dosing import build_drug_retrieval_queries, build_grounded_drug_dosing_payload
from shared.contracts.inference import RetrievedContext


DEFAULT_QUESTION_MIX: dict[str, float] = {
    "clinical-scenario": 0.55,
    "slightly-indirect": 0.25,
    "factual": 0.20,
}


MRA_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 4",
    snippet=(
        "WHICH MRA AND WHAT DOSE? Eplerenone: starting dose 25 mg o.d., target dose 50 mg o.d. "
        "Spironolactone: starting dose 25 mg o.d., target dose 50 mg o.d. "
        "If K rises above 5.5 mmol/L or creatinine rises to 2.5 mg/dL/eGFR <30, halve a dose. "
        "If K rises to >6.0 mmol/L or creatinine to 3.5 mg/dL/eGFR <20, stop MRA immediately."
    ),
    chunk_id="chunk-mra",
    page_number=12,
)

BETA_BLOCKER_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 3",
    snippet=(
        "WHICH BETA-BLOCKER AND WHAT DOSE? Bisoprolol: starting dose 1.25 mg o.d., target dose 10 mg o.d. "
        "Double the dose at not less than 2-week intervals. "
        "If persisting signs of congestion, hypotension (SBP <90 mmHg), relieve congestion and achieve euvolaemia before starting a beta-blocker. "
        "If <50 b.p.m. and worsening symptoms, halve the dose of beta-blocker."
    ),
    chunk_id="chunk-bb",
    page_number=11,
)

RAS_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 2",
    snippet=(
        "WHICH ACE-I AND WHAT DOSE? Enalapril: starting dose 2.5 mg b.i.d., target dose 10-20 mg b.i.d. "
        "Double the dose at not less than 2-week intervals. "
        "Significant hyperkalaemia (K >5.0 mmol/L), significant renal dysfunction [creatinine >2.5 mg/dL or eGFR <30], "
        "and SBP <90 mmHg are cautions. An increase in K to <= 5.5 mmol/L is acceptable. "
        "If K rises to >5.5 mmol/L or creatinine to >3.5 mg/dL/eGFR <20, the ACE-I should be stopped."
    ),
    chunk_id="chunk-ras",
    page_number=9,
)

ARNI_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 5",
    snippet=(
        "WHAT DOSE? Sac/Val: starting dose 49/51 mg b.i.d., target dose 97/103 mg b.i.d. "
        "24/26 mg b.i.d. in selected patients. A washout period of at least 36 h after ACE-I therapy is required. "
        "In some patients, one may consider a reduced starting dose (24/26 mg b.i.d.), namely in those with SBP 100-110 mmHg, eGFR 30-60 mL/min/1.73 m2. "
        "Significant hyperkalaemia (K >5.0 mmol/L). eGFR <30 and SBP <90 mmHg are cautions/contraindications. "
        "If K rises to >5.5 mmol/L or eGFR lowers to <30, the ARNI should be stopped."
    ),
    chunk_id="chunk-arni",
    page_number=14,
)

SGLT2_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 6",
    snippet=(
        "WHAT DOSE? Dapagliflozin: starting (and target) dose 10 mg o.d. Empagliflozin: starting (and target) dose 10 mg o.d. "
        "eGFR <20 mL/min/1.73 m2 and SBP <95 mmHg are contraindications/cautions. "
        "Fluid balance needs to be monitored because SGLT2 inhibitors may intensify diuresis."
    ),
    chunk_id="chunk-sglt2",
    page_number=15,
)

LOOP_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 7",
    snippet=(
        "WHICH DIURETIC AND WHAT DAILY DOSE? Loop diuretics: Furosemide: starting dose 20-40 mg, usual dose 40-240 mg. "
        "Bumetanide: starting dose 0.5-1 mg, usual dose 1-5 mg. Torasemide: starting dose 5-10 mg, usual dose 10-20 mg. "
        "Not indicated if the patient has never had symptoms or signs of congestion. "
        "SBP <90 mmHg and renal dysfunction are cautions. Hypovolaemia/dehydration: consider a diuretic dosage reduction."
    ),
    chunk_id="chunk-loop",
    page_number=17,
)

CANONICAL_CONTEXT = [
    MRA_CONTEXT,
    BETA_BLOCKER_CONTEXT,
    RAS_CONTEXT,
    ARNI_CONTEXT,
    SGLT2_CONTEXT,
    LOOP_CONTEXT,
]


@dataclass(slots=True)
class ScenarioSpec:
    name: str
    reasoning_type: str
    difficulty: str
    tags: list[str]


SCENARIO_SPECS: list[ScenarioSpec] = [
    ScenarioSpec("foundational_start", "prioritization", "easy", ["foundational-therapy", "initiation"]),
    ScenarioSpec("arni_switch", "scenario-application", "medium", ["arni", "switch"]),
    ScenarioSpec("arni_reduced_start", "safety-check", "medium", ["arni", "borderline-hemodynamics"]),
    ScenarioSpec("mra_caution", "safety-check", "medium", ["mra", "hyperkalaemia"]),
    ScenarioSpec("multidrug_stop", "safety-check", "hard", ["hyperkalaemia", "renal-dysfunction", "contraindication"]),
    ScenarioSpec("beta_blocker_bradycardia", "safety-check", "medium", ["beta-blocker", "bradycardia"]),
    ScenarioSpec("beta_blocker_congestion", "prioritization", "medium", ["beta-blocker", "congestion"]),
    ScenarioSpec("sglt2_start", "prioritization", "easy", ["sglt2", "initiation"]),
    ScenarioSpec("sglt2_hold", "safety-check", "medium", ["sglt2", "contraindication"]),
    ScenarioSpec("loop_intensify", "monitoring", "easy", ["loop-diuretic", "congestion"]),
    ScenarioSpec("loop_reduce", "monitoring", "medium", ["loop-diuretic", "volume-depletion"]),
    ScenarioSpec("stable_maintenance", "interpretation", "easy", ["maintenance", "stability"]),
]


QUESTION_TEMPLATES = {
    "clinical-scenario": [
        "What dose actions should the drug-dosing pipeline prioritize for this patient profile?",
        "Given this medication history and safety profile, which grounded drug-dose steps appear most appropriate?",
        "Which heart-failure drug families look most actionable for this patient right now?",
    ],
    "slightly-indirect": [
        "Which medication changes would you expect the drug-dosing pipeline to surface first for this profile?",
        "What would a grounded dosing-oriented answer likely focus on for this patient?",
        "Which therapy adjustments are most likely to be prioritized when this profile is sent to the dosing pipeline?",
    ],
    "factual": [
        "Based on this profile, which drug families look reasonable to start, increase, reduce, or avoid?",
        "Which dosing recommendations are most defensible from the guideline-style drug tables for this patient?",
        "What grounded medication actions are supported for this patient profile?",
    ],
}


ACTION_FACT_TEMPLATES = {
    "start": "{drug}",
    "increase": "{drug}",
    "switch": "{drug}",
    "maintain": "{drug}",
    "reduce": "reduce {drug}",
    "stop": "stop {drug}",
    "avoid_start": "avoid starting {drug}",
}


_DRUG_NAME_CLEANUPS = {
    "sac/val": "sacubitril/valsartan",
}


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_clean_text(v) for v in values if _clean_text(v)]


def _slug_like(value: Any) -> str:
    text = _clean_text(value).lower().replace("_", "-")
    text = re.sub(r"[^a-z0-9-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


def _sample_value(rng: random.Random, low: float, high: float, decimals: int = 1) -> float:
    return round(rng.uniform(low, high), decimals)


def _mix_counts(total: int, mix: dict[str, float]) -> list[str]:
    if total <= 0:
        return []
    mix_total = sum(mix.values())
    if mix_total <= 0:
        raise ValueError("question_mix must sum to more than zero")
    normalized = {key: value / mix_total for key, value in mix.items()}
    counts = {key: int(value * total) for key, value in normalized.items()}
    while sum(counts.values()) < total:
        for key in sorted(normalized, key=lambda item: normalized[item], reverse=True):
            counts[key] += 1
            if sum(counts.values()) == total:
                break
    return [key for key, count in counts.items() for _ in range(count)]


def _choose_question(question_type: str, rng: random.Random) -> str:
    options = QUESTION_TEMPLATES.get(question_type) or QUESTION_TEMPLATES["clinical-scenario"]
    return rng.choice(options)


def _normalize_drug_name(value: str | None) -> str:
    if not value:
        return "therapy"
    lowered = value.strip().lower()
    return _DRUG_NAME_CLEANUPS.get(lowered, lowered)


def _as_bool(value: bool | None) -> bool | None:
    return bool(value) if value is not None else None


def _build_patient_profile(spec: ScenarioSpec, rng: random.Random) -> dict[str, Any]:
    patient: dict[str, Any] = {
        "age": int(rng.randint(52, 84)),
        "gender": rng.choice(["male", "female"]),
        "ef": int(rng.randint(22, 38)),
        "nyha": rng.choice([2, 3]),
        "weight": int(rng.randint(64, 98)),
        "blood_pressure_diastolic": int(rng.randint(58, 76)),
    }

    if spec.name == "foundational_start":
        patient.update({
            "potassium": _sample_value(rng, 4.0, 4.8),
            "egfr": _sample_value(rng, 48, 72, 0),
            "creatinine": _sample_value(rng, 1.0, 1.5),
            "blood_pressure_systolic": int(rng.randint(106, 122)),
            "heart_rate": int(rng.randint(68, 88)),
            "DoseSpiro_prev": 0,
            "DoseBB_prev": 1.25,
            "RASDose_prev": 2.5,
            "ARNIDose_prev": 0,
            "SGLT2Dose_prev": 0,
            "Loop_dose_prev": 20,
            "congestion_present": False,
        })
    elif spec.name == "arni_switch":
        patient.update({
            "potassium": _sample_value(rng, 4.1, 4.9),
            "egfr": _sample_value(rng, 42, 65, 0),
            "creatinine": _sample_value(rng, 1.1, 1.7),
            "blood_pressure_systolic": int(rng.randint(108, 126)),
            "heart_rate": int(rng.randint(66, 84)),
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 2.5,
            "RASDose_prev": 10,
            "ARNIDose_prev": 0,
            "SGLT2Dose_prev": 10,
            "Loop_dose_prev": 40,
            "switch_to_arni": True,
            "congestion_present": False,
        })
    elif spec.name == "arni_reduced_start":
        patient.update({
            "potassium": _sample_value(rng, 4.3, 4.9),
            "egfr": _sample_value(rng, 31, 45, 0),
            "creatinine": _sample_value(rng, 1.6, 2.1),
            "blood_pressure_systolic": int(rng.randint(100, 110)),
            "heart_rate": int(rng.randint(70, 90)),
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 2.5,
            "RASDose_prev": 10,
            "ARNIDose_prev": 0,
            "SGLT2Dose_prev": 10,
            "Loop_dose_prev": 40,
            "switch_to_arni": True,
        })
    elif spec.name == "mra_caution":
        patient.update({
            "potassium": _sample_value(rng, 5.2, 5.5),
            "egfr": _sample_value(rng, 30, 36, 0),
            "creatinine": _sample_value(rng, 1.8, 2.4),
            "blood_pressure_systolic": int(rng.randint(94, 108)),
            "heart_rate": int(rng.randint(76, 96)),
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 2.5,
            "RASDose_prev": 10,
            "ARNIDose_prev": 0,
            "SGLT2Dose_prev": 0,
            "Loop_dose_prev": 40,
        })
    elif spec.name == "multidrug_stop":
        patient.update({
            "potassium": _sample_value(rng, 5.7, 6.1),
            "egfr": _sample_value(rng, 16, 24, 0),
            "creatinine": _sample_value(rng, 2.8, 3.4),
            "blood_pressure_systolic": int(rng.randint(82, 90)),
            "heart_rate": int(rng.randint(44, 54)),
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 2.5,
            "RASDose_prev": 10,
            "ARNIDose_prev": "49/51 mg b.i.d.",
            "SGLT2Dose_prev": 10,
            "Loop_dose_prev": 40,
        })
    elif spec.name == "beta_blocker_bradycardia":
        patient.update({
            "potassium": _sample_value(rng, 4.0, 4.8),
            "egfr": _sample_value(rng, 42, 70, 0),
            "creatinine": _sample_value(rng, 1.0, 1.7),
            "blood_pressure_systolic": int(rng.randint(102, 120)),
            "heart_rate": int(rng.randint(42, 49)),
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 5,
            "RASDose_prev": 10,
            "ARNIDose_prev": 0,
            "SGLT2Dose_prev": 10,
            "Loop_dose_prev": 20,
        })
    elif spec.name == "beta_blocker_congestion":
        patient.update({
            "potassium": _sample_value(rng, 4.2, 4.9),
            "egfr": _sample_value(rng, 34, 58, 0),
            "creatinine": _sample_value(rng, 1.2, 1.9),
            "blood_pressure_systolic": int(rng.randint(92, 102)),
            "heart_rate": int(rng.randint(78, 104)),
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 0,
            "RASDose_prev": 10,
            "ARNIDose_prev": 0,
            "SGLT2Dose_prev": 0,
            "Loop_dose_prev": 40,
            "congestion_present": True,
            "weight_gain_kg": _sample_value(rng, 1.8, 3.5),
        })
    elif spec.name == "sglt2_start":
        patient.update({
            "potassium": _sample_value(rng, 4.0, 4.8),
            "egfr": _sample_value(rng, 32, 62, 0),
            "creatinine": _sample_value(rng, 1.1, 1.9),
            "blood_pressure_systolic": int(rng.randint(100, 118)),
            "heart_rate": int(rng.randint(70, 92)),
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 2.5,
            "RASDose_prev": 10,
            "ARNIDose_prev": 0,
            "SGLT2Dose_prev": 0,
            "Loop_dose_prev": 40,
        })
    elif spec.name == "sglt2_hold":
        patient.update({
            "potassium": _sample_value(rng, 4.3, 5.0),
            "egfr": _sample_value(rng, 16, 24, 0),
            "creatinine": _sample_value(rng, 2.2, 3.0),
            "blood_pressure_systolic": int(rng.randint(88, 96)),
            "heart_rate": int(rng.randint(72, 96)),
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 2.5,
            "RASDose_prev": 10,
            "ARNIDose_prev": 0,
            "SGLT2Dose_prev": 0,
            "Loop_dose_prev": 40,
            "volume_depleted": rng.choice([True, False]),
        })
    elif spec.name == "loop_intensify":
        patient.update({
            "potassium": _sample_value(rng, 4.0, 4.8),
            "egfr": _sample_value(rng, 30, 55, 0),
            "creatinine": _sample_value(rng, 1.3, 2.0),
            "blood_pressure_systolic": int(rng.randint(96, 116)),
            "heart_rate": int(rng.randint(80, 104)),
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 2.5,
            "RASDose_prev": 10,
            "ARNIDose_prev": 0,
            "SGLT2Dose_prev": 10,
            "Loop_dose_prev": 40,
            "congestion_present": True,
            "weight_gain_kg": _sample_value(rng, 2.0, 4.0),
        })
    elif spec.name == "loop_reduce":
        patient.update({
            "potassium": _sample_value(rng, 4.0, 4.9),
            "egfr": _sample_value(rng, 32, 60, 0),
            "creatinine": _sample_value(rng, 1.2, 2.0),
            "blood_pressure_systolic": int(rng.randint(94, 108)),
            "heart_rate": int(rng.randint(72, 96)),
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 2.5,
            "RASDose_prev": 10,
            "ARNIDose_prev": 0,
            "SGLT2Dose_prev": 10,
            "Loop_dose_prev": 80,
            "congestion_present": False,
            "volume_depleted": True,
        })
    elif spec.name == "stable_maintenance":
        patient.update({
            "potassium": _sample_value(rng, 4.1, 4.8),
            "egfr": _sample_value(rng, 46, 70, 0),
            "creatinine": _sample_value(rng, 1.0, 1.5),
            "blood_pressure_systolic": int(rng.randint(108, 126)),
            "heart_rate": int(rng.randint(62, 82)),
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 5,
            "RASDose_prev": 10,
            "ARNIDose_prev": "49/51 mg b.i.d.",
            "SGLT2Dose_prev": 10,
            "Loop_dose_prev": 20,
            "congestion_present": False,
        })
    else:
        raise ValueError(f"Unsupported scenario: {spec.name}")

    return patient


def _selected_recommendation_facts(payload: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    for item in payload.get("selected_recommendations") or []:
        drug = _normalize_drug_name(item.get("drug"))
        action = str(item.get("action") or "").strip().lower()
        dose = _clean_text(item.get("dose"))
        fact = ACTION_FACT_TEMPLATES.get(action, drug).format(drug=drug)
        facts.append(fact)
        if dose:
            facts.append(f"{drug} {dose}")
    return facts


def _safety_facts(payload: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    for item in payload.get("safety_cautions") or []:
        drug = _normalize_drug_name(item.get("drug"))
        action = str(item.get("action") or "").strip().lower()
        status = str(item.get("status") or "").strip().lower()
        note = _clean_text(item.get("note"))
        if action == "stop":
            facts.append(f"stop {drug}")
        elif action == "reduce":
            facts.append(f"reduce {drug}")
        elif action == "avoid_start":
            facts.append(f"avoid starting {drug}")
        if status == "contraindicated":
            facts.append(f"{drug} contraindicated")
        if note:
            facts.append(note)
    return facts


def _required_facts(payload: dict[str, Any]) -> list[str]:
    facts = _selected_recommendation_facts(payload)
    if not facts:
        facts.extend(["no grounded drug dose recommendation", "safety constraints dominate"])
    facts.extend(_safety_facts(payload))
    deduped: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        cleaned = _clean_text(fact)
        lowered = cleaned.lower()
        if cleaned and lowered not in seen:
            seen.add(lowered)
            deduped.append(cleaned)
    return deduped[:6]


def _forbidden_facts(payload: dict[str, Any]) -> list[str]:
    forbidden: list[str] = []
    for family, item in (payload.get("recommendations") or {}).items():
        action = str(item.get("action") or "").strip().lower()
        drug = _normalize_drug_name(item.get("drug") or family)
        if action in {"stop", "reduce", "avoid_start"}:
            forbidden.append(f"start {drug}")
            forbidden.append(f"increase {drug}")
    deduped: list[str] = []
    seen: set[str] = set()
    for fact in forbidden:
        lowered = fact.lower()
        if lowered not in seen:
            seen.add(lowered)
            deduped.append(fact)
    return deduped[:6]


def _reference_answer(payload: dict[str, Any], patient_variables: dict[str, Any]) -> str:
    visible = payload.get("selected_recommendations") or []
    cautions = payload.get("safety_cautions") or []
    if not visible:
        why = []
        if patient_variables.get("potassium") is not None and float(patient_variables["potassium"]) > 5.5:
            why.append("marked hyperkalaemia")
        if patient_variables.get("egfr") is not None and float(patient_variables["egfr"]) < 30:
            why.append("renal dysfunction")
        if patient_variables.get("blood_pressure_systolic") is not None and float(patient_variables["blood_pressure_systolic"]) < 95:
            why.append("low systolic blood pressure")
        driver = ", ".join(why) if why else "multiple safety constraints"
        return (
            "A grounded dosing answer should acknowledge that no visible dose recommendation can be safely surfaced "
            f"because {driver} limits escalation. It should stay cautious rather than inventing an up-titration plan."
        )

    action_lines = []
    for item in visible[:3]:
        drug = _normalize_drug_name(item.get("drug"))
        dose = _clean_text(item.get("dose"))
        action = str(item.get("action") or "").replace("_", " ")
        action_lines.append(f"{action} {drug} at {dose}" if dose else f"{action} {drug}")

    answer = "A grounded dosing answer should prioritize " + "; ".join(action_lines) + "."
    if cautions:
        caution_bits = []
        for item in cautions[:2]:
            drug = _normalize_drug_name(item.get("drug"))
            action = str(item.get("action") or "").replace("_", " ")
            caution_bits.append(f"{action} or restraint for {drug}")
        answer += " It should also surface safety caution around " + ", ".join(caution_bits) + "."
    return answer


def _retrieval_hints(payload: dict[str, Any], patient_variables: dict[str, Any]) -> dict[str, Any]:
    key_terms = []
    for item in payload.get("selected_recommendations") or []:
        drug = _clean_text(item.get("drug"))
        action = _clean_text(item.get("action")).replace("_", " ")
        if drug:
            key_terms.append(drug)
        if action:
            key_terms.append(action)
    for item in payload.get("safety_cautions") or []:
        note = _clean_text(item.get("note"))
        if note:
            key_terms.append(note)
    if patient_variables.get("potassium") is not None and float(patient_variables["potassium"]) > 5.0:
        key_terms.append("hyperkalaemia")
    if patient_variables.get("egfr") is not None and float(patient_variables["egfr"]) < 30:
        key_terms.append("renal dysfunction")
    if patient_variables.get("blood_pressure_systolic") is not None and float(patient_variables["blood_pressure_systolic"]) < 95:
        key_terms.append("hypotension")
    if _as_bool(patient_variables.get("congestion_present")):
        key_terms.append("congestion")
    return {
        "key_terms": [term for term in dict.fromkeys(_clean_list(key_terms))][:8],
        "expected_section": "supplementary dosing tables",
        "document_scope": "behavioral-observation",
    }


def _hallucination_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    unsupported_targets = [
        "inventing non-grounded dose escalations",
        "recommending contraindicated escalation despite safety thresholds",
        "naming drug substitutions not present in the retrieved dosing tables",
    ]
    if not payload.get("selected_recommendations"):
        likely_failure_modes = [
            "fabricating a visible dose recommendation despite safety constraints",
            "ignoring the need to abstain when no grounded recommendation survives filtering",
        ]
        risk_level = "high"
    else:
        likely_failure_modes = [
            "overstating certainty beyond the grounded table rows",
            "missing a key safety caution while recommending an otherwise grounded dose",
        ]
        risk_level = "medium"
    return {
        "risk_level": risk_level,
        "likely_failure_modes": likely_failure_modes,
        "unsupported_targets": unsupported_targets,
        "is_hallucination_test": False,
        "case_kind": "drug_dosing_only",
        "expected_behavior": "answer_from_drug_dosing_pipeline_without_explicit_question",
    }


def _load_existing_cases(base_dataset_path: str | Path | None) -> tuple[dict[str, Any] | None, list[BenchmarkCase]]:
    if not base_dataset_path:
        return None, []
    base_payload = json.loads(Path(base_dataset_path).read_text(encoding="utf-8"))
    existing_cases = [BenchmarkCase(**case) for case in base_payload.get("cases") or []]
    return base_payload, existing_cases


def _assign_unique_ids(new_cases: list[BenchmarkCase], existing_cases: list[BenchmarkCase]) -> None:
    seen_ids = {case.id for case in existing_cases}
    next_index = 1
    for case in new_cases:
        original_id = case.id
        while case.id in seen_ids:
            case.id = f"drug-case-{next_index:03d}"
            next_index += 1
        seen_ids.add(case.id)
        if case.id != original_id:
            case.generation_metadata["original_case_id"] = original_id


def _mode_counts(cases: list[BenchmarkCase]) -> dict[str, int]:
    normal_count = 0
    biomarker_count = 0
    drug_count = 0
    observation_count = 0
    for case in cases:
        raw = case.to_dict()
        generation_metadata = raw.get("generation_metadata") or {}
        request_mode = str(generation_metadata.get("request_mode") or "").strip().lower()
        tags = {str(tag).strip().lower() for tag in (raw.get("tags") or [])}
        if is_observation_only_case(raw):
            observation_count += 1
        else:
            normal_count += 1
        if request_mode == "biomarker_only" or "biomarker-only" in tags:
            biomarker_count += 1
        if request_mode == "drug_dosing_only" or "drug-dosing-only" in tags:
            drug_count += 1
    return {
        "normal_case_count": normal_count,
        "biomarker_observation_case_count": biomarker_count,
        "drug_dosing_observation_case_count": drug_count,
        "observation_only_case_count": observation_count,
    }


def _summary(cases: list[BenchmarkCase]) -> dict[str, Any]:
    return {
        "question_type_counts": dict(sorted(Counter(case.question_type for case in cases).items())),
        "answerability_counts": dict(sorted(Counter(case.answerability for case in cases).items())),
        "difficulty_counts": dict(sorted(Counter(case.difficulty for case in cases).items())),
        "reasoning_type_counts": dict(sorted(Counter(case.reasoning_type for case in cases).items())),
        "average_patient_variable_count": round(sum(len(case.patient_variables) for case in cases) / max(1, len(cases)), 1),
        **_mode_counts(cases),
    }


def _build_case(index: int, spec: ScenarioSpec, question_type: str, rng: random.Random, dataset_version: str, prompt_version: str) -> BenchmarkCase:
    patient_variables = _build_patient_profile(spec, rng)
    retrieval_queries = [item["query"] for item in build_drug_retrieval_queries(patient_variables)]
    payload = build_grounded_drug_dosing_payload(
        patient_variables=patient_variables,
        retrieved_context=CANONICAL_CONTEXT,
        retrieval_queries=retrieval_queries,
    )
    tags = sorted(set([_slug_like(tag) for tag in spec.tags] + ["drug-dosing-only", "observation-case"]))
    return BenchmarkCase(
        id=f"drug-case-{index:03d}",
        dataset_version=dataset_version,
        question=_choose_question(question_type, rng),
        question_type=question_type,
        reasoning_type=spec.reasoning_type,
        difficulty=spec.difficulty,
        answerability="answerable",
        expected_behavior="answer_from_drug_dosing_pipeline_without_explicit_question",
        expected_abstention_style="stay_cautious_when_no_grounded_recommendation_survives" if not payload.get("selected_recommendations") else None,
        case_weight=0.75,
        review_status="auto_generated_unreviewed",
        patient_variables=patient_variables,
        gold_passage_id=None,
        gold_passage_text=None,
        gold_passage_normalized=None,
        gold_passage_hash=None,
        anchor_start_text=None,
        anchor_end_text=None,
        source_document_id=None,
        source_document_name=None,
        source_page=None,
        source_block_index=None,
        reference_answer=_reference_answer(payload, patient_variables),
        required_facts=_required_facts(payload),
        forbidden_facts=_forbidden_facts(payload),
        query_variants=[],
        tags=tags,
        retrieval_hints=_retrieval_hints(payload, patient_variables),
        unanswerable_reason=None,
        generation_metadata={
            "provider": "rule_based",
            "prompt_version": prompt_version,
            "case_source": "drug_dosing_generator_v1",
            "request_mode": "drug_dosing_only",
            "omit_question_from_request": True,
            "evaluation_intent": "behavior_observation",
            "evaluation_profile": "observation_only",
            "pipeline_variant": "drug_dosing",
            "request_options": {"pipeline_variant": "drug_dosing"},
            "canonical_evidence_source": "embedded_supplementary_dosing_rows_v1",
            "retrieval_query_count": len(retrieval_queries),
            "selected_recommendation_count": len(payload.get("selected_recommendations") or []),
        },
        passage_metadata={
            "word_count": 0,
            "char_count": 0,
            "section_title": None,
            "recommended_action_count": len(payload.get("selected_recommendations") or []),
            "safety_caution_count": len(payload.get("safety_cautions") or []),
        },
        hallucination_metadata=_hallucination_metadata(payload),
    )


def build_drug_dosing_dataset(
    *,
    output_path: str | Path,
    output_jsonl_path: str | Path | None,
    dataset_id: str,
    dataset_version: str,
    dataset_size: int,
    base_dataset_path: str | Path | None = None,
    seed: int,
    prompt_version: str,
    question_mix: dict[str, float] | None = None,
) -> list[BenchmarkCase]:
    rng = random.Random(seed)
    question_types = _mix_counts(dataset_size, question_mix or DEFAULT_QUESTION_MIX)
    rng.shuffle(question_types)

    new_cases: list[BenchmarkCase] = []
    for index in range(dataset_size):
        spec = SCENARIO_SPECS[index % len(SCENARIO_SPECS)]
        new_cases.append(
            _build_case(
                index=index + 1,
                spec=spec,
                question_type=question_types[index],
                rng=rng,
                dataset_version=dataset_version,
                prompt_version=prompt_version,
            )
        )

    base_payload, existing_cases = _load_existing_cases(base_dataset_path)
    _assign_unique_ids(new_cases, existing_cases)
    combined_cases = existing_cases + new_cases

    payload = {
        "schema_version": str((base_payload or {}).get("schema_version") or "1.4"),
        "dataset_type": "benchmark_dataset",
        "dataset_id": str((base_payload or {}).get("dataset_id") or dataset_id),
        "dataset_version": str((base_payload or {}).get("dataset_version") or dataset_version),
        "dataset_size": len(combined_cases),
        "question_mix": (base_payload or {}).get("question_mix") or question_mix or DEFAULT_QUESTION_MIX,
        "case_kind_mix": (base_payload or {}).get("case_kind_mix") or {"answerable": 1.0},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_metadata": {
            **dict((base_payload or {}).get("generation_metadata") or {}),
            "drug_dosing_augmentation": {
                "provider": "rule_based",
                "prompt_version": prompt_version,
                "generator_mode": "drug_dosing_only_append",
                "request_question_policy": "omit_question_from_request",
                "pipeline_variant": "drug_dosing",
                "canonical_evidence_source": "embedded_supplementary_dosing_rows_v1",
                "added_case_count": len(new_cases),
                "base_dataset_path": str(base_dataset_path) if base_dataset_path else None,
            },
        },
        "source_documents": list((base_payload or {}).get("source_documents") or []),
        "summary": _summary(combined_cases),
        "cases": [case.to_dict() for case in combined_cases],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    jsonl = Path(output_jsonl_path) if output_jsonl_path else output.with_suffix(".jsonl")
    jsonl.write_text(
        "\n".join(json.dumps(case.to_dict(), ensure_ascii=False) for case in combined_cases) + "\n",
        encoding="utf-8",
    )
    return combined_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--base-dataset", default=None)
    parser.add_argument("--dataset-id", default="benchmark_drug_dosing_v1")
    parser.add_argument("--dataset-version", default="benchmark_drug_dosing_v1")
    parser.add_argument("--dataset-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-version", default="drug_dosing_dataset_generation_v1")
    args = parser.parse_args()

    cases = build_drug_dosing_dataset(
        output_path=args.output,
        output_jsonl_path=args.output_jsonl,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        dataset_size=args.dataset_size,
        base_dataset_path=args.base_dataset,
        seed=args.seed,
        prompt_version=args.prompt_version,
    )
    print(f"Wrote {len(cases)} total cases")
    print(f"JSON: {args.output}")
    print(f"JSONL: {args.output_jsonl or str(Path(args.output).with_suffix('.jsonl'))}")


if __name__ == "__main__":
    main()
