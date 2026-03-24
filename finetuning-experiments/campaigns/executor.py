from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean
from typing import Any

from artifacts.loader import load_run_artifact
from campaigns.generator import generate_campaign_runs, write_expanded_campaign_configs
from campaigns.schema import CampaignConfig
from runners.benchmark_runner import run_benchmark
from utils.datetime import utc_now_iso
from utils.hashing import fingerprint
from utils.json import write_json

logger = logging.getLogger(__name__)



def _avg(values: list[float]) -> float:
    return mean(values) if values else 0.0



def _campaign_output_root(config: CampaignConfig, output_dir: str | Path) -> Path:
    return Path(output_dir) / "_campaigns" / config.label



def _build_campaign_summary(executed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    run_artifacts = [item["artifact"] for item in executed_runs if item.get("artifact")]
    metric_keys = sorted({key for artifact in run_artifacts for key in ((artifact.get("normalized_metrics") or {}).keys())})
    averages = {
        key: _avg(
            [
                float((artifact.get("normalized_metrics") or {}).get(key, 0.0))
                for artifact in run_artifacts
                if (artifact.get("normalized_metrics") or {}).get(key) is not None
            ]
        )
        for key in metric_keys
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in executed_runs:
        artifact = item.get("artifact") or {}
        group_values = item.get("group_values") or {}
        group_key = ", ".join(f"{key}={value}" for key, value in group_values.items()) if group_values else "ungrouped"
        groups.setdefault(group_key, []).append(
            {
                "run_id": artifact.get("run_id"),
                "label": artifact.get("label") or item.get("label"),
                "normalized_metrics": artifact.get("normalized_metrics") or {},
                "overrides": item.get("overrides") or {},
            }
        )
    grouped_summary = {}
    for group_key, items in groups.items():
        group_metric_keys = sorted({k for item in items for k in (item.get("normalized_metrics") or {}).keys()})
        grouped_summary[group_key] = {
            "run_count": len(items),
            "average_metrics": {
                key: _avg(
                    [
                        float((item.get("normalized_metrics") or {}).get(key, 0.0))
                        for item in items
                        if (item.get("normalized_metrics") or {}).get(key) is not None
                    ]
                )
                for key in group_metric_keys
            },
            "runs": items,
        }
    return {
        "run_count": len(run_artifacts),
        "average_normalized_metrics": averages,
        "groups": grouped_summary,
    }



def execute_campaign(config: CampaignConfig) -> Path:
    generated_runs = generate_campaign_runs(config)
    if not generated_runs:
        raise ValueError(f"Campaign '{config.label}' produced no runnable configurations.")

    first_output_dir = generated_runs[0]["config"].execution.output_dir
    campaign_root = _campaign_output_root(config, first_output_dir)
    campaign_root.mkdir(parents=True, exist_ok=True)

    if config.execution.write_expanded_configs:
        write_expanded_campaign_configs(first_output_dir, config.label, generated_runs)

    completed_runs: list[dict[str, Any]] = []
    failed_runs: list[dict[str, Any]] = []

    for item in generated_runs:
        run_config = item["config"]
        try:
            artifact_path = run_benchmark(run_config)
            artifact = load_run_artifact(artifact_path)
            completed_runs.append(
                {
                    "index": item["index"],
                    "run_id": artifact.get("run_id"),
                    "label": artifact.get("label"),
                    "artifact_path": str(artifact_path),
                    "status": "completed",
                    "overrides": item.get("overrides") or {},
                    "group_values": item.get("group_values") or {},
                    "config_fingerprint": item.get("config_fingerprint"),
                    "run_fingerprint": (artifact.get("cache") or {}).get("run_fingerprint"),
                    "ingestion_fingerprint": (artifact.get("cache") or {}).get("ingestion_fingerprint"),
                    "normalized_metrics": artifact.get("normalized_metrics") or {},
                    "artifact": artifact,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Campaign run failed label=%s index=%s", run_config.label, item["index"])
            failure = {
                "index": item["index"],
                "label": run_config.label,
                "status": "failed",
                "error": str(exc),
                "overrides": item.get("overrides") or {},
                "group_values": item.get("group_values") or {},
                "config_fingerprint": item.get("config_fingerprint"),
            }
            failed_runs.append(failure)
            if config.execution.fail_fast or not config.execution.continue_on_error:
                break

    payload = {
        "artifact_type": "campaign",
        "artifact_version": "1.0",
        "campaign_id": fingerprint({"label": config.label, "created_at": utc_now_iso()}, prefix="campaign"),
        "label": config.label,
        "datetime": utc_now_iso(),
        "description": config.description,
        "config": config.to_dict(),
        "run_count": len(completed_runs),
        "failed_run_count": len(failed_runs),
        "runs": [{k: v for k, v in item.items() if k != "artifact"} for item in completed_runs],
        "failed_runs": failed_runs,
        "campaign_summary": _build_campaign_summary(completed_runs),
    }
    return write_json(campaign_root / f"{config.label}.campaign.json", payload)
