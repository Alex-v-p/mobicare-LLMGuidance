from __future__ import annotations

from typing import Any


def is_observation_only_case(case: dict[str, Any]) -> bool:
    generation_metadata = case.get("generation_metadata") or {}
    tags = {str(tag).strip().lower() for tag in (case.get("tags") or [])}
    request_mode = str(generation_metadata.get("request_mode") or "").strip().lower()
    pipeline_variant = str(generation_metadata.get("pipeline_variant") or "").strip().lower()
    return bool(
        generation_metadata.get("evaluation_profile") == "observation_only"
        or request_mode in {"biomarker_only", "drug_dosing_only"}
        or request_mode.endswith("_only")
        or generation_metadata.get("omit_question_from_request")
        or "observation-case" in tags
        or "biomarker-only" in tags
        or "drug-dosing-only" in tags
        or (pipeline_variant == "drug_dosing" and generation_metadata.get("evaluation_intent") == "behavior_observation")
    )
