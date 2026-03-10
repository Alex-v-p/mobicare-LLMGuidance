from __future__ import annotations

from typing import Any, Dict, List

from shared.contracts.inference import RetrievedContext


FAKE_CONTEXTS = [
    RetrievedContext(
        source_id="fake-guideline-001",
        title="Heart failure beta-blocker titration",
        snippet=(
            "When heart rate and blood pressure allow, treatment advice can mention gradual "
            "up-titration of beta-blocker therapy with monitoring for dizziness, hypotension, and bradycardia."
        ),
    ),
    RetrievedContext(
        source_id="fake-guideline-002",
        title="Renal function and potassium monitoring",
        snippet=(
            "Renal function markers such as creatinine, urea, cystatin C, sodium, and potassium should be checked "
            "before intensifying RAAS-oriented therapy or mineralocorticoid receptor antagonists."
        ),
    ),
    RetrievedContext(
        source_id="fake-guideline-003",
        title="Congestion symptom review",
        snippet=(
            "Clinical congestion indicators such as edema, orthopnea, rales, jugular venous distension, and BNP can be "
            "used to frame whether fluid status may require closer review."
        ),
    ),
]


IMPORTANT_VARIABLES = {
    "age",
    "gender",
    "Weight",
    "BMI",
    "HeartRate",
    "BPsyst",
    "BPdiast",
    "EF",
    "NYHA",
    "Crea",
    "Urea",
    "CysC",
    "Sodium",
    "Potassium",
    "BNP",
    "Rales",
    "Orthopnea",
    "Edema",
    "Jugularvein_01",
    "DoseBB_prev",
    "RASDose_prev",
    "DoseSpiro_prev",
    "Loop_dose_prev",
    "SGLT2Dose_prev",
    "ARNIDose_prev",
}


class FakeRetriever:
    def retrieve(self, patient_variables: Dict[str, Any], use_fake_rag: bool = True) -> List[RetrievedContext]:
        if not use_fake_rag:
            return []

        matches = [key for key in patient_variables if key in IMPORTANT_VARIABLES and patient_variables[key] is not None]
        if not matches:
            return FAKE_CONTEXTS[:1]

        if any(key in matches for key in {"Crea", "Urea", "CysC", "Sodium", "Potassium"}):
            return [FAKE_CONTEXTS[1], FAKE_CONTEXTS[0]]
        if any(key in matches for key in {"Rales", "Orthopnea", "Edema", "Jugularvein_01", "BNP"}):
            return [FAKE_CONTEXTS[2], FAKE_CONTEXTS[0]]
        return [FAKE_CONTEXTS[0], FAKE_CONTEXTS[2]]
