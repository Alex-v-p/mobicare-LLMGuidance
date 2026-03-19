from __future__ import annotations

from typing import Any

from campaigns.schema import CampaignConfig


_ALLOWED_TOP_LEVEL_PATHS = {"ingestion", "source_mapping", "inference", "execution", "label", "notes", "change_note", "documents_version", "dataset_path", "dataset_version"}


def _is_match(candidate: dict[str, Any], rule: dict[str, Any]) -> bool:
    return all(candidate.get(key) == value for key, value in rule.items())



def validate_campaign_config(config: CampaignConfig) -> None:
    errors: list[str] = []
    if not config.label.strip():
        errors.append("campaign.label must not be empty.")
    if not config.base_config_path:
        errors.append("campaign.base_config_path is required.")
    elif not __import__("pathlib").Path(config.base_config_path).exists():
        errors.append(f"campaign.base_config_path does not exist: {config.base_config_path}")
    if not config.matrix and not config.include:
        errors.append("campaign must define at least one matrix entry or explicit include override.")
    for container_name, rules in (("matrix", config.matrix),):
        for path, values in rules.items():
            if not path or "." not in path:
                errors.append(f"{container_name} key '{path}' must be a dotted path like 'inference.top_k'.")
                continue
            if path.split(".", 1)[0] not in _ALLOWED_TOP_LEVEL_PATHS:
                errors.append(f"{container_name} key '{path}' starts with unsupported section '{path.split('.', 1)[0]}'.")
            if not isinstance(values, list) or not values:
                errors.append(f"{container_name} entry '{path}' must be a non-empty list.")
    for collection_name, collection in (("include", config.include), ("exclude", config.exclude)):
        if not isinstance(collection, list):
            errors.append(f"campaign.{collection_name} must be a list.")
            continue
        for index, item in enumerate(collection):
            if not isinstance(item, dict) or not item:
                errors.append(f"campaign.{collection_name}[{index}] must be a non-empty object.")
                continue
            for path in item:
                if "." not in path:
                    errors.append(f"campaign.{collection_name}[{index}] key '{path}' must be a dotted path.")
                elif path.split(".", 1)[0] not in _ALLOWED_TOP_LEVEL_PATHS:
                    errors.append(
                        f"campaign.{collection_name}[{index}] key '{path}' starts with unsupported section '{path.split('.', 1)[0]}'."
                    )
    if config.execution.max_runs is not None and config.execution.max_runs <= 0:
        errors.append("campaign.execution.max_runs must be positive when provided.")
    if not config.run_template.label_template.strip():
        errors.append("campaign.run_template.label_template must not be empty.")
    if errors:
        raise ValueError("Campaign config validation failed:\n- " + "\n- ".join(errors))



def is_excluded(candidate: dict[str, Any], exclusions: list[dict[str, Any]]) -> bool:
    return any(_is_match(candidate, rule) for rule in exclusions)



def deduplicate_overrides(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        key = tuple(sorted(item.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
