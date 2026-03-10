from __future__ import annotations

from typing import Any


def normalize_for_offset_matching(text: str) -> str:
    return " ".join(text.split())


def resolve_page_number(metadata: dict[str, Any], *, chunk_text: str, start_offset: int | None = None) -> int | None:
    explicit_page_number = metadata.get("page_number")
    if explicit_page_number is not None:
        try:
            return int(explicit_page_number)
        except (TypeError, ValueError):
            return None

    page_ranges = metadata.get("page_ranges")
    if not isinstance(page_ranges, list) or not page_ranges:
        return None

    if start_offset is None:
        normalized_source_text = str(metadata.get("normalized_source_text") or "")
        normalized_chunk_text = normalize_for_offset_matching(chunk_text)
        if normalized_source_text and normalized_chunk_text:
            start_offset = normalized_source_text.find(normalized_chunk_text)

    if start_offset is None or start_offset < 0:
        return None

    chunk_length = len(normalize_for_offset_matching(chunk_text))
    chunk_midpoint = start_offset + max(0, chunk_length // 2)

    for page_range in page_ranges:
        if not isinstance(page_range, dict):
            continue
        page_start = int(page_range.get("start", 0))
        page_end = int(page_range.get("end", 0))
        if page_start <= chunk_midpoint < page_end:
            return int(page_range.get("page_number"))

    trailing = page_ranges[-1]
    if isinstance(trailing, dict):
        try:
            return int(trailing.get("page_number"))
        except (TypeError, ValueError):
            return None
    return None
