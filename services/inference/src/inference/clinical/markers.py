from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any


@dataclass(frozen=True, slots=True)
class MarkerBand:
    low: float | None = None
    high: float | None = None
    direction: str = "outside_range"
    gender: tuple[str, ...] = ()
    age_min: float | None = None
    age_max: float | None = None
    note: str | None = None

    def matches(self, context: dict[str, Any]) -> bool:
        gender_value = _normalize_gender(context.get("gender"))
        if self.gender and gender_value not in self.gender:
            return False

        age_value = _to_float(context.get("age"))
        if self.age_min is not None and (age_value is None or age_value < self.age_min):
            return False
        if self.age_max is not None and (age_value is None or age_value > self.age_max):
            return False
        return True

    def description(self) -> str | None:
        parts: list[str] = []
        if self.gender:
            if len(self.gender) == 1:
                parts.append(f"gender={self.gender[0]}")
            else:
                parts.append(f"gender in {', '.join(self.gender)}")
        if self.age_min is not None and self.age_max is not None:
            parts.append(f"age {int(self.age_min)}-{int(self.age_max)}")
        elif self.age_min is not None:
            parts.append(f"age >= {int(self.age_min)}")
        elif self.age_max is not None:
            parts.append(f"age <= {int(self.age_max)}")
        if self.note:
            parts.append(self.note)
        return "; ".join(parts) if parts else None


@dataclass(frozen=True, slots=True)
class MarkerDefinition:
    key: str
    label: str
    unit: str | None = None
    kind: str = "numeric"
    direction: str = "outside_range"
    bands: tuple[MarkerBand, ...] = ()
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AppliedReference:
    low: float | None = None
    high: float | None = None
    unit: str | None = None
    direction: str = "outside_range"
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ClinicalFinding:
    key: str
    label: str
    value: Any
    status: str
    summary: str
    unit: str | None = None
    reference_description: str | None = None


@dataclass(frozen=True, slots=True)
class ClinicalProfile:
    recognized_variables: list[ClinicalFinding]
    abnormal_variables: list[ClinicalFinding]
    informational_variables: list[ClinicalFinding]
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
def load_marker_ranges() -> dict[str, MarkerDefinition]:
    with resources.files("inference.clinical").joinpath("marker_ranges.json").open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    definitions: dict[str, MarkerDefinition] = {}
    for key, payload in raw.items():
        normalized_key = _normalize_key(key)
        definition = _parse_marker_definition(normalized_key, payload)
        for alias in (normalized_key, *definition.aliases):
            definitions[_normalize_key(alias)] = definition
    return definitions



def _parse_marker_definition(key: str, payload: dict[str, Any]) -> MarkerDefinition:
    aliases = tuple(_normalize_key(alias) for alias in payload.get("aliases", []))
    kind = str(payload.get("kind") or "numeric")
    direction = str(payload.get("direction") or "outside_range")
    label = str(payload.get("label") or key.replace("_", " ").title())
    unit = payload.get("unit")

    raw_bands = payload.get("bands")
    if raw_bands:
        bands = tuple(_parse_band(item, default_direction=direction) for item in raw_bands)
    else:
        bands = (
            MarkerBand(
                low=_to_float(payload.get("low")),
                high=_to_float(payload.get("high")),
                direction=direction,
            ),
        )

    return MarkerDefinition(
        key=key,
        label=label,
        unit=unit,
        kind=kind,
        direction=direction,
        bands=bands,
        aliases=aliases,
    )



def _parse_band(payload: dict[str, Any], *, default_direction: str) -> MarkerBand:
    genders = payload.get("gender") or payload.get("genders") or []
    if isinstance(genders, str):
        gender_values = (_normalize_gender(genders),)
    else:
        gender_values = tuple(
            normalized for normalized in (_normalize_gender(item) for item in genders) if normalized
        )
    return MarkerBand(
        low=_to_float(payload.get("low")),
        high=_to_float(payload.get("high")),
        direction=str(payload.get("direction") or default_direction),
        gender=gender_values,
        age_min=_to_float(payload.get("age_min")),
        age_max=_to_float(payload.get("age_max")),
        note=str(payload.get("note")) if payload.get("note") is not None else None,
    )



def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None



def _normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")



def _normalize_gender(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "m": "male",
        "man": "male",
        "male": "male",
        "f": "female",
        "woman": "female",
        "female": "female",
    }
    return aliases.get(normalized, normalized)



def _resolve_reference(definition: MarkerDefinition, context: dict[str, Any]) -> AppliedReference | None:
    if definition.kind != "numeric":
        return AppliedReference(unit=definition.unit, direction=definition.direction)

    for band in definition.bands:
        if band.matches(context):
            return AppliedReference(
                low=band.low,
                high=band.high,
                unit=definition.unit,
                direction=band.direction,
                description=band.description(),
            )

    fallback = definition.bands[0] if definition.bands else None
    if fallback is None:
        return None
    return AppliedReference(
        low=fallback.low,
        high=fallback.high,
        unit=definition.unit,
        direction=fallback.direction,
        description=fallback.description(),
    )



def _classify(value: float, reference: AppliedReference | None) -> str:
    if reference is None:
        return "uninterpretable"
    if reference.direction == "context":
        return "informational"
    if reference.direction == "low_is_bad":
        if reference.low is not None and value < reference.low:
            return "low"
        if reference.high is not None and value > reference.high:
            return "high"
        return "normal"
    if reference.direction == "high_is_bad":
        if reference.high is not None and value > reference.high:
            return "high"
        if reference.low is not None and value < reference.low:
            return "low"
        return "normal"
    if reference.low is not None and value < reference.low:
        return "low"
    if reference.high is not None and value > reference.high:
        return "high"
    if reference.low is None and reference.high is None:
        return "informational"
    return "normal"



def _format_range(reference: AppliedReference) -> str:
    bounds: list[str] = []
    if reference.low is not None:
        bounds.append(str(reference.low))
    if reference.high is not None:
        bounds.append(str(reference.high))
    range_text = " to ".join(bounds)
    if reference.unit and range_text:
        range_text += f" {reference.unit}"
    if reference.description:
        if range_text:
            return f"{range_text} ({reference.description})"
        return reference.description
    return range_text



def _summary(value: Any, definition: MarkerDefinition, status: str, reference: AppliedReference | None) -> str:
    rendered = f"{definition.label}={value}"
    if definition.unit:
        rendered += f" {definition.unit}"
    if status == "informational":
        return f"{rendered} was recorded for context."
    if reference is None:
        return f"{rendered} could not be interpreted against the configured reference." 

    reference_text = _format_range(reference)
    if status == "normal":
        suffix = f" within the configured reference" + (f" ({reference_text})" if reference_text else "")
        return f"{rendered} is{suffix}."
    if status == "low":
        suffix = f" below the configured reference" + (f" ({reference_text})" if reference_text else "")
        return f"{rendered} is{suffix}."
    if status == "high":
        suffix = f" above the configured reference" + (f" ({reference_text})" if reference_text else "")
        return f"{rendered} is{suffix}."
    return f"{rendered} could not be interpreted against the configured reference."



def build_clinical_profile(patient_variables: dict[str, Any]) -> ClinicalProfile:
    definitions = load_marker_ranges()
    normalized_context = {_normalize_key(key): value for key, value in patient_variables.items()}
    if "sex" in normalized_context and "gender" not in normalized_context:
        normalized_context["gender"] = normalized_context["sex"]

    recognized: list[ClinicalFinding] = []
    abnormal: list[ClinicalFinding] = []
    informational: list[ClinicalFinding] = []
    unknown: list[str] = []

    for raw_key, raw_value in sorted(patient_variables.items()):
        key = _normalize_key(raw_key)
        definition = definitions.get(key)
        if definition is None:
            unknown.append(raw_key)
            continue

        if definition.kind == "context":
            finding = ClinicalFinding(
                key=key,
                label=definition.label,
                value=raw_value,
                status="informational",
                summary=_summary(raw_value, definition, "informational", AppliedReference(unit=definition.unit, direction="context")),
                unit=definition.unit,
            )
            recognized.append(finding)
            informational.append(finding)
            continue

        numeric_value = _to_float(raw_value)
        if numeric_value is None:
            unknown.append(raw_key)
            continue

        reference = _resolve_reference(definition, normalized_context)
        status = _classify(numeric_value, reference)
        finding = ClinicalFinding(
            key=key,
            label=definition.label,
            value=raw_value,
            status=status,
            summary=_summary(raw_value, definition, status, reference),
            unit=definition.unit,
            reference_description=reference.description if reference is not None else None,
        )
        recognized.append(finding)
        if status in {"low", "high"}:
            abnormal.append(finding)
        elif status == "informational":
            informational.append(finding)

    return ClinicalProfile(
        recognized_variables=recognized,
        abnormal_variables=abnormal,
        informational_variables=informational,
        unknown_variables=unknown,
    )



def build_question_from_patient_data(patient_variables: dict[str, Any], profile: ClinicalProfile | None = None) -> str:
    clinical_profile = profile or build_clinical_profile(patient_variables)
    relevant_terms = clinical_profile.relevant_terms()
    if relevant_terms:
        focus = ", ".join(relevant_terms[:3])
        return (
            "What are the most relevant next-step priorities, safety checks, and interpretation points "
            f"for this patient, especially regarding {focus}?"
        )
    return "What are the most relevant next-step priorities and safety checks for this patient based on the available variables?"
