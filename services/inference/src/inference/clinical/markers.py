from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any


@dataclass(frozen=True, slots=True)
class MarkerRange:
    key: str
    label: str
    low: float | None = None
    high: float | None = None
    unit: str | None = None
    direction: str = "outside_range"


@dataclass(frozen=True, slots=True)
class ClinicalFinding:
    key: str
    label: str
    value: Any
    status: str
    summary: str
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class ClinicalProfile:
    recognized_variables: list[ClinicalFinding]
    abnormal_variables: list[ClinicalFinding]
    unknown_variables: list[str]

    @property
    def has_abnormal_variables(self) -> bool:
        return bool(self.abnormal_variables)

    def relevant_terms(self) -> list[str]:
        terms = [finding.label for finding in self.abnormal_variables]
        if not terms:
            terms = [finding.label for finding in self.recognized_variables[:3]]
        return terms


@lru_cache(maxsize=1)
def load_marker_ranges() -> dict[str, MarkerRange]:
    with resources.files("inference.clinical").joinpath("marker_ranges.json").open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    ranges: dict[str, MarkerRange] = {}
    for key, payload in raw.items():
        ranges[key.lower()] = MarkerRange(
            key=key.lower(),
            label=str(payload.get("label") or key.replace("_", " ").title()),
            low=_to_float(payload.get("low")),
            high=_to_float(payload.get("high")),
            unit=payload.get("unit"),
            direction=str(payload.get("direction") or "outside_range"),
        )
    return ranges


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def _classify(value: float, marker: MarkerRange) -> str:
    if marker.direction == "low_is_bad":
        if marker.low is not None and value < marker.low:
            return "low"
        if marker.high is not None and value > marker.high:
            return "high"
        return "normal"
    if marker.direction == "high_is_bad":
        if marker.high is not None and value > marker.high:
            return "high"
        if marker.low is not None and value < marker.low:
            return "low"
        return "normal"
    if marker.low is not None and value < marker.low:
        return "low"
    if marker.high is not None and value > marker.high:
        return "high"
    return "normal"


def _summary(value: Any, marker: MarkerRange, status: str) -> str:
    rendered = f"{marker.label}={value}"
    if marker.unit:
        rendered += f" {marker.unit}"
    if status == "normal":
        return f"{rendered} is within the configured range."
    if status == "low":
        return f"{rendered} is below the configured range."
    if status == "high":
        return f"{rendered} is above the configured range."
    return f"{rendered} could not be interpreted against the configured range."


def build_clinical_profile(patient_variables: dict[str, Any]) -> ClinicalProfile:
    marker_ranges = load_marker_ranges()
    recognized: list[ClinicalFinding] = []
    abnormal: list[ClinicalFinding] = []
    unknown: list[str] = []

    for raw_key, raw_value in sorted(patient_variables.items()):
        key = _normalize_key(raw_key)
        marker = marker_ranges.get(key)
        if marker is None:
            unknown.append(raw_key)
            continue
        numeric_value = _to_float(raw_value)
        if numeric_value is None:
            unknown.append(raw_key)
            continue
        status = _classify(numeric_value, marker)
        finding = ClinicalFinding(
            key=key,
            label=marker.label,
            value=raw_value,
            status=status,
            summary=_summary(raw_value, marker, status),
            unit=marker.unit,
        )
        recognized.append(finding)
        if status != "normal":
            abnormal.append(finding)

    return ClinicalProfile(
        recognized_variables=recognized,
        abnormal_variables=abnormal,
        unknown_variables=unknown,
    )


def build_question_from_patient_data(patient_variables: dict[str, Any], profile: ClinicalProfile | None = None) -> str:
    clinical_profile = profile or build_clinical_profile(patient_variables)
    relevant_terms = clinical_profile.relevant_terms()
    if relevant_terms:
        focus = ", ".join(relevant_terms[:3])
        return (
            "Based on the available patient variables, what document-grounded treatment, "
            f"management, or follow-up guidance is most relevant, especially regarding {focus}?"
        )
    return (
        "Based on the available patient variables, what document-grounded treatment, "
        "management, or follow-up guidance is most relevant?"
    )
