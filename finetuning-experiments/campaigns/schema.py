from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CampaignExecutionConfig:
    fail_fast: bool = False
    max_runs: int | None = None
    retry_failed: bool = False
    continue_on_error: bool = True
    write_expanded_configs: bool = True


@dataclass(slots=True)
class CampaignRunTemplate:
    label_template: str = "{campaign_label}-{index:03d}"
    notes_template: str = ""
    change_note_template: str = ""


@dataclass(slots=True)
class CampaignConfig:
    label: str
    base_config_path: str
    description: str = ""
    group_by: list[str] = field(default_factory=list)
    matrix: dict[str, list[Any]] = field(default_factory=dict)
    include: list[dict[str, Any]] = field(default_factory=list)
    exclude: list[dict[str, Any]] = field(default_factory=list)
    run_template: CampaignRunTemplate = field(default_factory=CampaignRunTemplate)
    execution: CampaignExecutionConfig = field(default_factory=CampaignExecutionConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "base_config_path": self.base_config_path,
            "description": self.description,
            "group_by": list(self.group_by),
            "matrix": dict(self.matrix),
            "include": list(self.include),
            "exclude": list(self.exclude),
            "run_template": {
                "label_template": self.run_template.label_template,
                "notes_template": self.run_template.notes_template,
                "change_note_template": self.run_template.change_note_template,
            },
            "execution": {
                "fail_fast": self.execution.fail_fast,
                "max_runs": self.execution.max_runs,
                "retry_failed": self.execution.retry_failed,
                "continue_on_error": self.execution.continue_on_error,
                "write_expanded_configs": self.execution.write_expanded_configs,
            },
            "metadata": dict(self.metadata),
        }
