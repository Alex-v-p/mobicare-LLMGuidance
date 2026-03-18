from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TelemetryStage:
    name: str
    status: str = "unknown"
    duration_ms: float | None = None
    started_at: str | None = None
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TelemetryEnvelope:
    stages: list[TelemetryStage] = field(default_factory=list)
    derived: dict[str, float | None] = field(default_factory=dict)
    source: str = "derived"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "derived": dict(self.derived),
            "source": self.source,
        }
