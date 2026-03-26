from __future__ import annotations

from typing import Any


def verify_grounded_payload(payload: dict[str, Any]) -> tuple[str, list[str], str]:
    selected = payload.get("selected_recommendations") or []
    if not selected:
        if payload.get("safety_cautions"):
            return "pass", ["grounded_safety_cautions_without_visible_uptitration"], "high"
        return "pass", ["no_grounded_recommendation"], "medium"
    ungrounded = [item["family"] for item in selected if not item.get("grounded") or not item.get("evidence_chunk_ids")]
    if ungrounded:
        return "fail", [f"ungrounded_recommendation:{family}" for family in ungrounded], "high"
    return "pass", ["grounded_recommendations_with_evidence"], "high"
