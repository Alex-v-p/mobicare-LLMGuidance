from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.contracts.clinical_config import ClinicalConfigMetadata, ClinicalConfigName


@dataclass(frozen=True)
class ManagedClinicalConfig:
    config_name: ClinicalConfigName
    bucket: str
    object_name: str


@dataclass(frozen=True)
class CurrentConfigState:
    metadata: ClinicalConfigMetadata
    payload: dict[str, Any]
    raw_bytes: bytes


__all__ = ["CurrentConfigState", "ManagedClinicalConfig"]
