from __future__ import annotations

from inference.clinical import ClinicalProfile
from inference.pipeline.support.specialty import SpecialtyFocus, detected_clusters


def missing_details(
    patient_variables: dict[str, object],
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
