from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NumericDoseStep:
    value: float
    label: str
    frequency: str


@dataclass(frozen=True, slots=True)
class CombinationDoseStep:
    numerator: int
    denominator: int
    label: str
    frequency: str


@dataclass(frozen=True, slots=True)
class DrugRecommendation:
    family: str
    drug: str | None
    action: str
    recommended_dose: str | None
    status: str
    note: str | None = None
    assumed_agent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MRA_REGIMENS: dict[str, list[NumericDoseStep]] = {
    "spironolactone": [
        NumericDoseStep(25.0, "25 mg", "o.d."),
        NumericDoseStep(50.0, "50 mg", "o.d."),
    ],
    "eplerenone": [
        NumericDoseStep(25.0, "25 mg", "o.d."),
        NumericDoseStep(50.0, "50 mg", "o.d."),
    ],
}

BETA_BLOCKER_REGIMENS: dict[str, list[NumericDoseStep]] = {
    "bisoprolol": [
        NumericDoseStep(1.25, "1.25 mg", "o.d."),
        NumericDoseStep(2.5, "2.5 mg", "o.d."),
        NumericDoseStep(3.75, "3.75 mg", "o.d."),
        NumericDoseStep(5.0, "5 mg", "o.d."),
        NumericDoseStep(7.5, "7.5 mg", "o.d."),
        NumericDoseStep(10.0, "10 mg", "o.d."),
    ],
    "carvedilol": [
        NumericDoseStep(3.125, "3.125 mg", "b.i.d."),
        NumericDoseStep(6.25, "6.25 mg", "b.i.d."),
        NumericDoseStep(12.5, "12.5 mg", "b.i.d."),
        NumericDoseStep(25.0, "25 mg", "b.i.d."),
        NumericDoseStep(50.0, "50 mg", "b.i.d."),
    ],
    "metoprolol_succinate": [
        NumericDoseStep(12.5, "12.5 mg", "o.d."),
        NumericDoseStep(25.0, "25 mg", "o.d."),
        NumericDoseStep(50.0, "50 mg", "o.d."),
        NumericDoseStep(100.0, "100 mg", "o.d."),
        NumericDoseStep(200.0, "200 mg", "o.d."),
    ],
    "nebivolol": [
        NumericDoseStep(1.25, "1.25 mg", "o.d."),
        NumericDoseStep(2.5, "2.5 mg", "o.d."),
        NumericDoseStep(5.0, "5 mg", "o.d."),
        NumericDoseStep(10.0, "10 mg", "o.d."),
    ],
}

RAS_REGIMENS: dict[str, list[NumericDoseStep]] = {
    "enalapril": [
        NumericDoseStep(2.5, "2.5 mg", "b.i.d."),
        NumericDoseStep(5.0, "5 mg", "b.i.d."),
        NumericDoseStep(10.0, "10 mg", "b.i.d."),
        NumericDoseStep(20.0, "20 mg", "b.i.d."),
    ],
    "lisinopril": [
        NumericDoseStep(2.5, "2.5 mg", "o.d."),
        NumericDoseStep(5.0, "5 mg", "o.d."),
        NumericDoseStep(10.0, "10 mg", "o.d."),
        NumericDoseStep(20.0, "20 mg", "o.d."),
        NumericDoseStep(35.0, "35 mg", "o.d."),
    ],
    "ramipril": [
        NumericDoseStep(2.5, "2.5 mg", "o.d."),
        NumericDoseStep(5.0, "5 mg", "o.d."),
        NumericDoseStep(10.0, "10 mg", "o.d."),
    ],
    "trandolapril": [
        NumericDoseStep(0.5, "0.5 mg", "o.d."),
        NumericDoseStep(1.0, "1 mg", "o.d."),
        NumericDoseStep(2.0, "2 mg", "o.d."),
        NumericDoseStep(4.0, "4 mg", "o.d."),
    ],
    "captopril": [
        NumericDoseStep(6.25, "6.25 mg", "t.i.d."),
        NumericDoseStep(12.5, "12.5 mg", "t.i.d."),
        NumericDoseStep(25.0, "25 mg", "t.i.d."),
        NumericDoseStep(50.0, "50 mg", "t.i.d."),
    ],
}

ARNI_REGIMEN = [
    CombinationDoseStep(24, 26, "24/26 mg", "b.i.d."),
    CombinationDoseStep(49, 51, "49/51 mg", "b.i.d."),
    CombinationDoseStep(97, 103, "97/103 mg", "b.i.d."),
]

SGLT2_REGIMENS: dict[str, list[NumericDoseStep]] = {
    "dapagliflozin": [NumericDoseStep(10.0, "10 mg", "o.d.")],
    "empagliflozin": [NumericDoseStep(10.0, "10 mg", "o.d.")],
}

LOOP_REGIMENS: dict[str, list[NumericDoseStep]] = {
    "furosemide": [
        NumericDoseStep(20.0, "20 mg", "daily"),
        NumericDoseStep(40.0, "40 mg", "daily"),
        NumericDoseStep(80.0, "80 mg", "daily"),
        NumericDoseStep(120.0, "120 mg", "daily"),
        NumericDoseStep(160.0, "160 mg", "daily"),
        NumericDoseStep(240.0, "240 mg", "daily"),
    ],
    "bumetanide": [
        NumericDoseStep(0.5, "0.5 mg", "daily"),
        NumericDoseStep(1.0, "1 mg", "daily"),
        NumericDoseStep(2.0, "2 mg", "daily"),
        NumericDoseStep(5.0, "5 mg", "daily"),
    ],
    "torasemide": [
        NumericDoseStep(5.0, "5 mg", "daily"),
        NumericDoseStep(10.0, "10 mg", "daily"),
        NumericDoseStep(20.0, "20 mg", "daily"),
    ],
}

DEFAULT_AGENTS = {
    "mra": "spironolactone",
    "beta_blocker": "bisoprolol",
    "ras": "enalapril",
    "arni": "sacubitril_valsartan",
    "sglt2": "dapagliflozin",
    "loop_diuretic": "furosemide",
}


def build_drug_dosing_payload(patient_variables: dict[str, Any]) -> dict[str, Any]:
    snapshot = build_snapshot(patient_variables)
    recommendations = {
        "mra": _recommend_mra(snapshot),
        "beta_blocker": _recommend_beta_blocker(snapshot),
        "ras": _recommend_ras(snapshot),
        "arni": _recommend_arni(snapshot),
        "sglt2": _recommend_sglt2(snapshot),
        "loop_diuretic": _recommend_loop(snapshot),
    }
    output = {
        "mode": "drug_dosing",
        "recommendations": {key: value.to_dict() for key, value in recommendations.items()},
        "inputs_used": snapshot,
    }
    return output



def render_drug_dosing_answer(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)



def summarize_drug_dosing_warnings(patient_variables: dict[str, Any]) -> list[str]:
    snapshot = build_snapshot(patient_variables)
    warnings = [
        "Drug-dosing mode uses deterministic guideline-style rules and does not call the generative answer pipeline.",
        "Drug-dosing output is intentionally compact and should be reviewed clinically before use.",
    ]
    missing = [
        key
        for key in ("potassium", "egfr", "creatinine", "sbp", "heart_rate")
        if snapshot.get(key) is None
    ]
    if missing:
        warnings.append("Some dosing gates are missing: " + ", ".join(missing))
    if snapshot.get("ef") is None:
        warnings.append("HFrEF confirmation is missing; the dosing prototype assumes a heart-failure context.")
    return warnings



def build_snapshot(patient_variables: dict[str, Any]) -> dict[str, Any]:
    return {
        "potassium": _first_float(patient_variables, "potassium", "k", "kplus"),
        "egfr": _first_float(patient_variables, "egfr", "e_gfr"),
        "creatinine": _first_float(patient_variables, "creatinine", "scr", "serum_creatinine"),
        "sbp": _first_float(patient_variables, "bpsyst", "sbp", "blood_pressure_systolic"),
        "heart_rate": _first_float(patient_variables, "heartrate", "heart_rate", "pulse", "hr"),
        "ef": _first_float(patient_variables, "ef", "lvef"),
        "nyha": _first_str(patient_variables, "nyha"),
        "dose_mra_prev": _first_raw(patient_variables, "dosespiro_prev", "spiro", "mra_dose_prev"),
        "dose_bb_prev": _first_raw(patient_variables, "dosebb_prev", "bb_dose_prev", "beta_blocker_dose_prev"),
        "dose_ras_prev": _first_raw(patient_variables, "rasdose_prev", "ras_dose_prev", "acei_dose_prev", "arb_dose_prev"),
        "dose_arni_prev": _first_raw(patient_variables, "arnidose_prev", "arni_dose_prev"),
        "dose_sglt2_prev": _first_raw(patient_variables, "sglt2dose_prev", "sglt2_dose_prev"),
        "dose_loop_prev": _first_raw(patient_variables, "loop_dose_prev", "loop_dose", "loop_diuretic_dose_prev"),
        "mra_agent": _normalize_agent(_first_str(patient_variables, "mra_agent", "spiro_agent", "mineralocorticoid_agent"), DEFAULT_AGENTS["mra"]),
        "beta_blocker_agent": _normalize_agent(_first_str(patient_variables, "beta_blocker_agent", "bb_agent"), DEFAULT_AGENTS["beta_blocker"]),
        "ras_agent": _normalize_agent(_first_str(patient_variables, "ras_agent", "acei_agent", "arb_agent"), DEFAULT_AGENTS["ras"]),
        "sglt2_agent": _normalize_agent(_first_str(patient_variables, "sglt2_agent"), DEFAULT_AGENTS["sglt2"]),
        "loop_agent": _normalize_agent(_first_str(patient_variables, "loop_agent", "loop_diuretic_agent"), DEFAULT_AGENTS["loop_diuretic"]),
        "congestion_present": _first_bool(patient_variables, "congestion_present", "congestion", "edema", "orthopnea", "rales"),
        "volume_depleted": _first_bool(patient_variables, "volume_depleted", "dehydrated", "hypovolemia", "hypovolaemia"),
        "weight_gain_kg": _first_float(patient_variables, "weight_gain_kg", "weight_gain"),
        "switch_to_arni": _first_bool(patient_variables, "switch_to_arni", "prefer_arni"),
    }



def _recommend_mra(snapshot: dict[str, Any]) -> DrugRecommendation:
    agent, assumed = _agent_or_default(snapshot.get("mra_agent"), DEFAULT_AGENTS["mra"])
    current_index = _current_numeric_step_index(snapshot.get("dose_mra_prev"), MRA_REGIMENS[agent])
    potassium = snapshot.get("potassium")
    egfr = snapshot.get("egfr")
    creatinine = snapshot.get("creatinine")

    if _gt(potassium, 6.0) or _lt(egfr, 20.0) or _gt(creatinine, 3.5):
        return DrugRecommendation("mra", agent, "stop", None, "contraindicated", "Severe renal or potassium safety threshold crossed.", assumed)
    if current_index is not None and (_gt(potassium, 5.5) or _lt(egfr, 30.0) or _gt(creatinine, 2.5)):
        reduced = MRA_REGIMENS[agent][max(0, current_index - 1)]
        return DrugRecommendation("mra", agent, "reduce", _dose_text(reduced), "caution", "Reduce one step and monitor potassium/renal function closely.", assumed)
    if current_index is None and (_gt(potassium, 5.0) or _lt(egfr, 30.0) or _gt(creatinine, 2.5)):
        return DrugRecommendation("mra", agent, "avoid_start", None, "caution", "Do not start while potassium or renal function is outside the starting thresholds.", assumed)
    if current_index is None:
        return DrugRecommendation("mra", agent, "start", _dose_text(MRA_REGIMENS[agent][0]), "eligible", None, assumed)
    if current_index >= len(MRA_REGIMENS[agent]) - 1:
        return DrugRecommendation("mra", agent, "maintain", _dose_text(MRA_REGIMENS[agent][current_index]), "at_target", None, assumed)
    return DrugRecommendation("mra", agent, "increase", _dose_text(MRA_REGIMENS[agent][current_index + 1]), "eligible", None, assumed)



def _recommend_beta_blocker(snapshot: dict[str, Any]) -> DrugRecommendation:
    agent, assumed = _agent_or_default(snapshot.get("beta_blocker_agent"), DEFAULT_AGENTS["beta_blocker"])
    steps = BETA_BLOCKER_REGIMENS[agent]
    current_index = _current_numeric_step_index(snapshot.get("dose_bb_prev"), steps)
    sbp = snapshot.get("sbp")
    heart_rate = snapshot.get("heart_rate")
    congestion = bool(snapshot.get("congestion_present"))

    if _lt(heart_rate, 50.0) or _lt(sbp, 90.0):
        if current_index is None:
            return DrugRecommendation("beta_blocker", agent, "avoid_start", None, "caution", "Do not start while bradycardic or hypotensive.", assumed)
        reduced = steps[max(0, current_index - 1)]
        action = "reduce" if current_index > 0 else "hold"
        dose = _dose_text(reduced) if current_index > 0 else None
        return DrugRecommendation("beta_blocker", agent, action, dose, "caution", "Reduce or hold for bradycardia/hypotension.", assumed)
    if congestion and current_index is None:
        return DrugRecommendation("beta_blocker", agent, "defer_start", None, "caution", "Relieve congestion and achieve euvolaemia before starting.", assumed)
    if current_index is None:
        return DrugRecommendation("beta_blocker", agent, "start", _dose_text(steps[0]), "eligible", None, assumed)
    if current_index >= len(steps) - 1:
        return DrugRecommendation("beta_blocker", agent, "maintain", _dose_text(steps[current_index]), "at_target", None, assumed)
    return DrugRecommendation("beta_blocker", agent, "increase", _dose_text(steps[current_index + 1]), "eligible", None, assumed)



def _recommend_ras(snapshot: dict[str, Any]) -> DrugRecommendation:
    if snapshot.get("dose_arni_prev") not in {None, "", 0, 0.0}:
        return DrugRecommendation("ras", None, "do_not_combine", None, "not_applicable", "ARNI is already present; do not co-prescribe ACE-I/ARB with ARNI.")
    if snapshot.get("switch_to_arni"):
        return DrugRecommendation("ras", None, "replace_with_arni", None, "not_applicable", "Switch pathway requested; ARNI recommendation takes precedence over RAS titration.")

    agent, assumed = _agent_or_default(snapshot.get("ras_agent"), DEFAULT_AGENTS["ras"])
    steps = RAS_REGIMENS[agent]
    current_index = _current_numeric_step_index(snapshot.get("dose_ras_prev"), steps)
    potassium = snapshot.get("potassium")
    egfr = snapshot.get("egfr")
    creatinine = snapshot.get("creatinine")
    sbp = snapshot.get("sbp")

    if _gt(potassium, 5.5) or _lt(egfr, 20.0) or _gt(creatinine, 3.5) or _lt(sbp, 90.0):
        if current_index is None:
            return DrugRecommendation("ras", agent, "avoid_start", None, "contraindicated", "Safety threshold crossed for starting or continuing ACE-I/ARB titration.", assumed)
        return DrugRecommendation("ras", agent, "stop", None, "contraindicated", "Stop and seek specialist review for severe potassium, renal, or blood-pressure safety issues.", assumed)
    if _gt(potassium, 5.0) or _lt(egfr, 30.0) or _gt(creatinine, 2.5):
        if current_index is None:
            return DrugRecommendation("ras", agent, "avoid_start", None, "caution", "Do not start while potassium or renal function is outside the starting thresholds.", assumed)
        reduced = steps[max(0, current_index - 1)]
        return DrugRecommendation("ras", agent, "reduce", _dose_text(reduced), "caution", "Reduce one step and re-check chemistry within 1-2 weeks.", assumed)
    if current_index is None:
        return DrugRecommendation("ras", agent, "start", _dose_text(steps[0]), "eligible", None, assumed)
    if current_index >= len(steps) - 1:
        return DrugRecommendation("ras", agent, "maintain", _dose_text(steps[current_index]), "at_target", None, assumed)
    return DrugRecommendation("ras", agent, "increase", _dose_text(steps[current_index + 1]), "eligible", None, assumed)



def _recommend_arni(snapshot: dict[str, Any]) -> DrugRecommendation:
    potassium = snapshot.get("potassium")
    egfr = snapshot.get("egfr")
    sbp = snapshot.get("sbp")
    current_index = _current_combo_step_index(snapshot.get("dose_arni_prev"), ARNI_REGIMEN)
    ras_current = snapshot.get("dose_ras_prev") not in {None, "", 0, 0.0}

    if _lt(egfr, 30.0) or _lt(sbp, 90.0) or _gt(potassium, 5.5):
        if current_index is None:
            return DrugRecommendation("arni", "sacubitril/valsartan", "avoid_start", None, "contraindicated", "Do not start when eGFR <30, SBP <90, or potassium >5.5.")
        return DrugRecommendation("arni", "sacubitril/valsartan", "stop", None, "contraindicated", "Stop for severe renal, potassium, or blood-pressure safety issue.")
    if current_index is not None:
        if current_index >= len(ARNI_REGIMEN) - 1:
            return DrugRecommendation("arni", "sacubitril/valsartan", "maintain", _dose_text(ARNI_REGIMEN[current_index]), "at_target", None)
        if _gt(potassium, 5.0):
            return DrugRecommendation("arni", "sacubitril/valsartan", "maintain", _dose_text(ARNI_REGIMEN[current_index]), "caution", "Do not up-titrate while potassium is above the routine titration threshold.")
        return DrugRecommendation("arni", "sacubitril/valsartan", "increase", _dose_text(ARNI_REGIMEN[current_index + 1]), "eligible", None)

    reduced_start = _between(sbp, 100.0, 110.0) or _between(egfr, 30.0, 60.0)
    step = ARNI_REGIMEN[0] if reduced_start else ARNI_REGIMEN[1]
    note = None
    action = "start"
    if ras_current:
        action = "switch"
        note = "Requires ACE-I washout of at least 36 hours before sacubitril/valsartan if switching from ACE-I."
    return DrugRecommendation("arni", "sacubitril/valsartan", action, _dose_text(step), "eligible", note)



def _recommend_sglt2(snapshot: dict[str, Any]) -> DrugRecommendation:
    agent, assumed = _agent_or_default(snapshot.get("sglt2_agent"), DEFAULT_AGENTS["sglt2"])
    egfr = snapshot.get("egfr")
    sbp = snapshot.get("sbp")

    if _lt(egfr, 20.0) or _lt(sbp, 95.0):
        return DrugRecommendation("sglt2", agent, "avoid_start", None, "contraindicated", "Do not start when eGFR <20 or SBP <95.", assumed)
    step = SGLT2_REGIMENS[agent][0]
    if snapshot.get("dose_sglt2_prev") not in {None, "", 0, 0.0}:
        return DrugRecommendation("sglt2", agent, "maintain", _dose_text(step), "at_target", None, assumed)
    return DrugRecommendation("sglt2", agent, "start", _dose_text(step), "eligible", None, assumed)



def _recommend_loop(snapshot: dict[str, Any]) -> DrugRecommendation:
    agent, assumed = _agent_or_default(snapshot.get("loop_agent"), DEFAULT_AGENTS["loop_diuretic"])
    steps = LOOP_REGIMENS[agent]
    current_index = _current_numeric_step_index(snapshot.get("dose_loop_prev"), steps)
    congestion = bool(snapshot.get("congestion_present")) or _gt(snapshot.get("weight_gain_kg"), 1.5)
    volume_depleted = bool(snapshot.get("volume_depleted"))
    sbp = snapshot.get("sbp")

    if volume_depleted or (_lt(sbp, 90.0) and not congestion):
        if current_index is None:
            return DrugRecommendation("loop_diuretic", agent, "avoid_start", None, "caution", "Do not start while volume depleted or hypotensive without congestion.", assumed)
        reduced = steps[max(0, current_index - 1)]
        action = "reduce" if current_index > 0 else "hold"
        dose = _dose_text(reduced) if current_index > 0 else None
        return DrugRecommendation("loop_diuretic", agent, action, dose, "caution", "Reduce or hold if hypovolaemic/hypotensive and not congested.", assumed)
    if not congestion and current_index is None:
        return DrugRecommendation("loop_diuretic", agent, "no_change", None, "not_indicated", "No clear congestion signal was provided.", assumed)
    if current_index is None:
        starting = steps[1] if agent == "furosemide" else steps[0]
        return DrugRecommendation("loop_diuretic", agent, "start", _dose_text(starting), "eligible", None, assumed)
    if congestion and current_index < len(steps) - 1:
        return DrugRecommendation("loop_diuretic", agent, "increase", _dose_text(steps[current_index + 1]), "eligible", None, assumed)
    return DrugRecommendation("loop_diuretic", agent, "maintain", _dose_text(steps[current_index]), "maintain", None, assumed)



def _dose_text(step: NumericDoseStep | CombinationDoseStep) -> str:
    return f"{step.label} {step.frequency}".strip()



def _current_numeric_step_index(raw_value: Any, steps: list[NumericDoseStep]) -> int | None:
    numeric = _parse_numeric_value(raw_value)
    if numeric is None or numeric <= 0:
        return None
    if 0 < numeric <= 1:
        numeric = numeric * 100
    if numeric <= 100 and not any(_close(numeric, step.value) for step in steps):
        target = steps[-1].value
        scaled = (numeric / 100.0) * target
        return _nearest_step_index(scaled, steps)
    return _nearest_step_index(numeric, steps)



def _current_combo_step_index(raw_value: Any, steps: list[CombinationDoseStep]) -> int | None:
    if raw_value is None or raw_value == "":
        return None
    text = str(raw_value).strip().lower().replace("mg", "")
    for index, step in enumerate(steps):
        if f"{step.numerator}/{step.denominator}" in text:
            return index
    numeric = _parse_numeric_value(raw_value)
    if numeric is None or numeric <= 0:
        return None
    first_components = [step.numerator for step in steps]
    return min(range(len(steps)), key=lambda idx: abs(first_components[idx] - numeric))



def _nearest_step_index(value: float, steps: list[NumericDoseStep]) -> int:
    return min(range(len(steps)), key=lambda idx: abs(steps[idx].value - value))



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



def _parse_numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        return float(value)
    text = str(value).strip().lower().replace("mg", "")
    if not text:
        return None
    if "/" in text:
        left = text.split("/", 1)[0].strip()
        try:
            return float(left)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None



def _normalize_agent(value: str | None, default: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "metoprolol": "metoprolol_succinate",
        "metoprolol_cr_xl": "metoprolol_succinate",
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



def _close(left: float, right: float) -> bool:
    return abs(left - right) < 0.001



def _gt(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold



def _lt(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold



def _between(value: float | None, lower: float, upper: float) -> bool:
    return value is not None and lower <= value <= upper
