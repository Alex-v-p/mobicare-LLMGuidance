from __future__ import annotations

from typing import Any


def render_drug_dosing_answer(payload: dict[str, Any]) -> str:
    selections = payload.get("selected_recommendations") or []
    if not selections:
        return "No grounded drug dose recommendation could be made from the retrieved guideline context."
    return "\n".join(render_visible_recommendation(item) for item in selections)


def summarize_drug_dosing_warnings(payload: dict[str, Any]) -> list[str]:
    warnings = [
        "Drug-dosing mode now requires retrieved guideline evidence before a dose recommendation is surfaced.",
        "The answer field is intentionally short; full evidence rows and safety trade-offs are stored in metadata.drug_dosing_payload.",
    ]
    if not payload.get("evidence_rows_used"):
        warnings.append("No drug-specific evidence rows were extracted from retrieved context.")
    if not payload.get("selected_recommendations"):
        warnings.append("No visible recommendation met both grounding and safety checks.")
    assumptions = [
        rec["drug"]
        for rec in payload.get("recommendations", {}).values()
        if rec.get("assumed_agent")
    ]
    if assumptions:
        warnings.append(
            "Some drug families used assumed agents because no explicit agent was supplied: "
            + ", ".join(sorted(set(assumptions)))
        )
    return warnings


def render_visible_recommendation(item: dict[str, Any]) -> str:
    drug = item.get("drug") or item.get("family") or "drug"
    dose = item.get("dose")
    return f"{drug}: {dose}" if dose else drug
