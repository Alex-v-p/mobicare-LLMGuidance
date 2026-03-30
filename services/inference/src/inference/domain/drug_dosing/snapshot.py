from __future__ import annotations

from typing import Any, Callable

from .parsing import first_bool, first_float, first_raw, first_str, normalize_agent


DefaultAgentResolver = Callable[[str], str]


def build_snapshot(patient_variables: dict[str, Any], *, default_agent: DefaultAgentResolver) -> dict[str, Any]:
    return {
        "potassium": first_float(patient_variables, "potassium", "k", "kplus"),
        "egfr": first_float(patient_variables, "egfr", "e_gfr"),
        "creatinine": first_float(patient_variables, "creatinine", "scr", "serum_creatinine"),
        "sbp": first_float(patient_variables, "bpsyst", "sbp", "blood_pressure_systolic"),
        "heart_rate": first_float(patient_variables, "heartrate", "heart_rate", "pulse", "hr"),
        "ef": first_float(patient_variables, "ef", "lvef"),
        "nyha": first_str(patient_variables, "nyha"),
        "weight": first_float(patient_variables, "weight"),
        "weight_gain_kg": first_float(patient_variables, "weight_gain_kg"),
        "congestion_present": first_bool(patient_variables, "congestion_present"),
        "volume_depleted": first_bool(patient_variables, "volume_depleted"),
        "switch_to_arni": first_bool(patient_variables, "switch_to_arni"),
        "dose_mra_prev": first_raw(patient_variables, "dosespiro_prev", "spiro", "mra_dose_prev", "DoseSpiro_prev"),
        "dose_bb_prev": first_raw(patient_variables, "dosebb_prev", "bb_dose_prev", "beta_blocker_dose_prev", "DoseBB_prev"),
        "dose_ras_prev": first_raw(patient_variables, "rasdose_prev", "ras_dose_prev", "acei_dose_prev", "arb_dose_prev", "RASDose_prev"),
        "dose_arni_prev": first_raw(patient_variables, "arnidose_prev", "arni_dose_prev", "ARNIDose_prev"),
        "dose_sglt2_prev": first_raw(patient_variables, "sglt2dose_prev", "sglt2_dose_prev", "SGLT2Dose_prev"),
        "dose_loop_prev": first_raw(patient_variables, "loop_dose_prev", "loop_dose", "loop_diuretic_dose_prev", "Loop_dose_prev"),
        "mra_agent": normalize_agent(first_str(patient_variables, "mra_agent", "spiro_agent", "mineralocorticoid_agent"), default_agent("mra")),
        "beta_blocker_agent": normalize_agent(first_str(patient_variables, "beta_blocker_agent", "bb_agent"), default_agent("beta_blocker")),
        "ras_agent": normalize_agent(first_str(patient_variables, "ras_agent", "acei_agent", "arb_agent"), default_agent("ras")),
        "sglt2_agent": normalize_agent(first_str(patient_variables, "sglt2_agent"), default_agent("sglt2")),
        "loop_agent": normalize_agent(first_str(patient_variables, "loop_agent", "diuretic_agent"), default_agent("loop_diuretic")),
    }
