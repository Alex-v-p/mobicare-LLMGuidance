from __future__ import annotations

import json
from importlib import resources
from typing import Any

from shared.config import Settings

_DEFAULTS_PACKAGE = "shared.resources.clinical_configs"
MANAGED_CLINICAL_CONFIG_FILENAMES: dict[str, str] = {
    "drug_dosing_catalog": "drug_dosing_catalog.json",
    "marker_ranges": "marker_ranges.json",
}


def _resource_name_for_config(config_name: str) -> str:
    try:
        return MANAGED_CLINICAL_CONFIG_FILENAMES[config_name]
    except KeyError as exc:
        raise ValueError(f"Unknown managed clinical config '{config_name}'") from exc


def load_clinical_config_default_bytes(config_name: str) -> bytes:
    resource_name = _resource_name_for_config(config_name)
    return resources.files(_DEFAULTS_PACKAGE).joinpath(resource_name).read_bytes()



def load_clinical_config_default_payload(config_name: str) -> dict[str, Any]:
    return json.loads(load_clinical_config_default_bytes(config_name).decode("utf-8"))



def build_managed_clinical_object_names(settings: Settings) -> dict[str, str]:
    prefix = settings.clinical_config_prefix.strip().strip("/")

    def _join(object_name: str) -> str:
        normalized_name = object_name.strip().lstrip("/")
        if not prefix:
            return normalized_name
        return f"{prefix}/{normalized_name}"

    return {
        "drug_dosing_catalog": _join(settings.clinical_drug_dosing_catalog_object_name),
        "marker_ranges": _join(settings.clinical_marker_ranges_object_name),
    }
