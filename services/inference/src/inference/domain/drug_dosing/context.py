from __future__ import annotations

from typing import Any

from shared.contracts.inference import RetrievedContext


def select_grounded_rag_context(
    retrieved_context: list[RetrievedContext],
    payload: dict[str, Any],
    *,
    max_items: int = 12,
) -> list[RetrievedContext]:
    relevant_ids: list[str] = []
    for recommendation in (payload.get("recommendations") or {}).values():
        if recommendation.get("grounded") and recommendation.get("evidence_chunk_ids"):
            relevant_ids.extend(recommendation.get("evidence_chunk_ids", []))
    seen_needed: set[str] = set()
    ordered_ids: list[str] = []
    for chunk_id in relevant_ids:
        if chunk_id and chunk_id not in seen_needed:
            seen_needed.add(chunk_id)
            ordered_ids.append(chunk_id)
    selected: list[RetrievedContext] = []
    selected_ids: set[str] = set()
    for chunk_id in ordered_ids:
        for item in retrieved_context:
            if item.chunk_id == chunk_id and chunk_id not in selected_ids:
                selected.append(item)
                selected_ids.add(chunk_id)
                break
        if len(selected) >= max_items:
            break
    if selected:
        return selected
    return retrieved_context[:max_items]
