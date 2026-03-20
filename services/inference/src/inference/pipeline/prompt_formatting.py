from __future__ import annotations

from typing import Any, Iterable

from inference.clinical import ClinicalProfile
from shared.contracts.inference import RetrievedContext


def patient_lines(patient_variables: dict[str, Any]) -> list[str]:
    lines = [f"- {key}: {value}" for key, value in sorted(patient_variables.items()) if value is not None]
    return lines or ["- No patient variables were provided."]



def profile_lines(clinical_profile: ClinicalProfile) -> list[str]:
    lines: list[str] = []
    if clinical_profile.abnormal_variables:
        lines.append("Abnormal / clinically relevant findings:")
        lines.extend(f"- {finding.summary}" for finding in clinical_profile.abnormal_variables)
    elif clinical_profile.recognized_variables:
        lines.append("Recognized variables:")
        lines.extend(f"- {finding.summary}" for finding in clinical_profile.recognized_variables[:6])
    if clinical_profile.informational_variables:
        lines.append("Context variables:")
        lines.extend(f"- {finding.label}={finding.value}" for finding in clinical_profile.informational_variables[:8])
    if clinical_profile.unknown_variables:
        lines.append("Variables without configured interpretation:")
        lines.extend(f"- {key}" for key in clinical_profile.unknown_variables[:10])
    return lines or ["- No interpreted clinical findings were derived from the patient variables."]



def context_lines(retrieved_context: list[RetrievedContext]) -> list[str]:
    lines = [f"Excerpt {index}: {item.snippet}" for index, item in enumerate(retrieved_context, start=1)]
    return lines or ["No retrieval context was provided."]



def bullet_block(lines: Iterable[str]) -> str:
    return "\n".join(lines)
