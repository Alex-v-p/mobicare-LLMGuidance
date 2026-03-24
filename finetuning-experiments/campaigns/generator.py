from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from campaigns.constraints import deduplicate_overrides, is_excluded, validate_campaign_config
from campaigns.schema import CampaignConfig, CampaignExecutionConfig, CampaignRunTemplate
from configs.loader import build_run_config
from utils.hashing import fingerprint
from utils.json import read_json, write_json



def load_campaign_config(path: str | Path) -> CampaignConfig:
    file_path = Path(path)
    raw = read_json(file_path)
    base_config_path = Path(raw["base_config_path"])
    if not base_config_path.is_absolute():
        base_config_path = (file_path.parent / base_config_path).resolve()
    config = CampaignConfig(
        label=raw["label"],
        base_config_path=str(base_config_path),
        description=raw.get("description", ""),
        group_by=list(raw.get("group_by") or []),
        matrix=dict(raw.get("matrix") or {}),
        include=list(raw.get("include") or []),
        exclude=list(raw.get("exclude") or []),
        run_template=CampaignRunTemplate(**(raw.get("run_template") or {})),
        execution=CampaignExecutionConfig(**(raw.get("execution") or {})),
        metadata=dict(raw.get("metadata") or {}),
    )
    validate_campaign_config(config)
    return config



def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[parts[-1]] = value



def _get_path(target: dict[str, Any], path: str) -> Any:
    current: Any = target
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current



def _build_matrix_overrides(matrix: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    keys = list(matrix.keys())
    values = [matrix[key] for key in keys]
    overrides: list[dict[str, Any]] = []
    for combo in product(*values):
        overrides.append({key: value for key, value in zip(keys, combo, strict=False)})
    return overrides



def _render_template(template: str, *, campaign_label: str, index: int, override_map: dict[str, Any], run_config: dict[str, Any]) -> str:
    if not template:
        return ""
    safe_context = {
        "campaign_label": campaign_label,
        "index": index,
        "config_label": run_config.get("label", ""),
    }
    for path, value in override_map.items():
        safe_context[path.replace(".", "_")] = value
    try:
        return template.format(**safe_context)
    except Exception:
        return template



def generate_campaign_runs(config: CampaignConfig) -> list[dict[str, Any]]:
    validate_campaign_config(config)
    base_payload = read_json(config.base_config_path)

    matrix_overrides = _build_matrix_overrides(config.matrix)
    combined_overrides = deduplicate_overrides(matrix_overrides + list(config.include or []))
    generated: list[dict[str, Any]] = []

    for index, override_map in enumerate(combined_overrides, start=1):
        if is_excluded(override_map, config.exclude):
            continue
        run_payload = deepcopy(base_payload)
        for path, value in override_map.items():
            _set_path(run_payload, path, value)
        run_payload["label"] = _render_template(
            config.run_template.label_template,
            campaign_label=config.label,
            index=index,
            override_map=override_map,
            run_config=run_payload,
        ) or f"{config.label}-{index:03d}"
        if config.run_template.notes_template:
            run_payload["notes"] = _render_template(
                config.run_template.notes_template,
                campaign_label=config.label,
                index=index,
                override_map=override_map,
                run_config=run_payload,
            )
        if config.run_template.change_note_template:
            run_payload["change_note"] = _render_template(
                config.run_template.change_note_template,
                campaign_label=config.label,
                index=index,
                override_map=override_map,
                run_config=run_payload,
            )
        generated.append(
            {
                "index": index,
                "overrides": override_map,
                "group_values": {path: _get_path(run_payload, path) for path in config.group_by},
                "config": build_run_config(run_payload),
                "config_fingerprint": fingerprint(run_payload, prefix="campaign-config"),
            }
        )
    if config.execution.max_runs is not None:
        return generated[: config.execution.max_runs]
    return generated



def write_expanded_campaign_configs(output_dir: str | Path, campaign_label: str, generated_runs: list[dict[str, Any]]) -> list[Path]:
    root = Path(output_dir) / "_campaigns" / campaign_label / "configs"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for item in generated_runs:
        config = item["config"]
        safe_label = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in config.label)
        path = write_json(root / f"{item['index']:03d}_{safe_label}.json", config.to_dict())
        paths.append(path)
    return paths
