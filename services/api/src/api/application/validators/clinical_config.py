from __future__ import annotations

from typing import Any, cast

from api.infrastructure.repositories.clinical_config import InvalidClinicalConfigError, UnknownClinicalConfigError
from shared.contracts.clinical_config import ClinicalConfigName


_ALLOWED_CONFIG_NAMES = {"marker_ranges", "drug_dosing_catalog"}


def normalize_clinical_config_name(config_name: ClinicalConfigName | str) -> ClinicalConfigName:
    if config_name not in _ALLOWED_CONFIG_NAMES:
        raise UnknownClinicalConfigError(f"Clinical config '{config_name}' is not managed")
    return cast(ClinicalConfigName, config_name)


def validate_clinical_config_payload(
    config_name: ClinicalConfigName | str,
    payload: dict[str, Any],
) -> tuple[ClinicalConfigName, dict[str, Any]]:
    normalized_name = normalize_clinical_config_name(config_name)

    if not isinstance(payload, dict):
        raise InvalidClinicalConfigError("Clinical config payload must be a JSON object")
    if not payload:
        raise InvalidClinicalConfigError("Clinical config payload must not be empty")

    if normalized_name == "marker_ranges":
        return normalized_name, _validate_marker_ranges_payload(payload)
    if normalized_name == "drug_dosing_catalog":
        return normalized_name, _validate_drug_dosing_catalog_payload(payload)

    raise UnknownClinicalConfigError(f"Clinical config '{config_name}' is not managed")


def _validate_marker_ranges_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for marker_name, definition in payload.items():
        if not isinstance(marker_name, str) or not marker_name.strip():
            raise InvalidClinicalConfigError("Marker range keys must be non-empty strings")
        if not isinstance(definition, dict):
            raise InvalidClinicalConfigError(f"Marker '{marker_name}' must map to an object")
        if "label" in definition and not isinstance(definition["label"], str):
            raise InvalidClinicalConfigError(f"Marker '{marker_name}' label must be a string")
        if "bands" in definition and not isinstance(definition["bands"], list):
            raise InvalidClinicalConfigError(f"Marker '{marker_name}' bands must be a list when provided")
    return payload


def _validate_drug_dosing_catalog_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required_keys = ("default_agents", "family_query_order", "family_priority", "families")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise InvalidClinicalConfigError("Drug dosing catalog is missing required keys: " + ", ".join(missing))

    default_agents = payload["default_agents"]
    family_query_order = payload["family_query_order"]
    family_priority = payload["family_priority"]
    families = payload["families"]

    if not isinstance(default_agents, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in default_agents.items()):
        raise InvalidClinicalConfigError("Drug dosing catalog default_agents must be an object of strings")
    if not isinstance(family_query_order, list) or not all(isinstance(item, str) for item in family_query_order):
        raise InvalidClinicalConfigError("Drug dosing catalog family_query_order must be a list of strings")
    if not isinstance(family_priority, dict) or not all(isinstance(k, str) and isinstance(v, int) for k, v in family_priority.items()):
        raise InvalidClinicalConfigError("Drug dosing catalog family_priority must be an object of integers")
    if not isinstance(families, dict) or not families:
        raise InvalidClinicalConfigError("Drug dosing catalog families must be a non-empty object")

    for family_name, definition in families.items():
        if not isinstance(family_name, str) or not family_name.strip():
            raise InvalidClinicalConfigError("Drug dosing family names must be non-empty strings")
        if not isinstance(definition, dict):
            raise InvalidClinicalConfigError(f"Drug dosing family '{family_name}' must map to an object")
        keywords = definition.get("keywords")
        query_template = definition.get("query_template")
        if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
            raise InvalidClinicalConfigError(
                f"Drug dosing family '{family_name}' keywords must be a list of strings"
            )
        if not isinstance(query_template, str) or not query_template.strip():
            raise InvalidClinicalConfigError(
                f"Drug dosing family '{family_name}' query_template must be a non-empty string"
            )

    return payload


__all__ = ["normalize_clinical_config_name", "validate_clinical_config_payload"]
