from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DrugEvidence:
    family: str
    drug: str | None
    starting_dose: str | None = None
    target_dose: str | None = None
    reduced_start_dose: str | None = None
    usual_range: str | None = None
    start_low_value: float | None = None
    start_high_value: float | None = None
    target_value: float | None = None
    reduced_start_value: float | None = None
    target_frequency: str | None = None
    caution_potassium_gt: float | None = None
    stop_potassium_gt: float | None = None
    caution_creatinine_gt: float | None = None
    stop_creatinine_gt: float | None = None
    caution_egfr_lt: float | None = None
    stop_egfr_lt: float | None = None
    caution_sbp_lt: float | None = None
    caution_hr_lt: float | None = None
    halve_on_excess: bool = False
    double_uptitration: bool = False
    requires_euvolemia_before_start: bool = False
    washout_note: str | None = None
    contraindication_note: str | None = None
    source_chunk_ids: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DrugRecommendation:
    family: str
    drug: str | None
    action: str
    recommended_dose: str | None
    status: str
    note: str | None = None
    tradeoff: str | None = None
    assumed_agent: bool = False
    grounded: bool = False
    evidence_chunk_ids: list[str] = field(default_factory=list)
    evidence_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
