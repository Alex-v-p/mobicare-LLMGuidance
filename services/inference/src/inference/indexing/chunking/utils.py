from __future__ import annotations

from typing import Any, Iterable

from inference.indexing.models import SourceDocument, TextChunk


def normalize_for_offset_matching(text: str) -> str:
    return " ".join(text.split())


def normalize_chunk_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def sliding_window_chunks(text: str, *, chunk_size: int, chunk_overlap: int) -> list[tuple[int, str]]:
    cleaned = normalize_chunk_text(text)
    if not cleaned:
        return []

    size = max(int(chunk_size), 1)
    overlap = max(0, min(int(chunk_overlap), size - 1))

    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        chunk_text = cleaned[start:end].strip()
        if chunk_text:
            chunks.append((start, chunk_text))
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunk(document: SourceDocument, *, index: int, text: str, strategy: str, start_offset: int | None = None) -> TextChunk:
    return TextChunk(
        chunk_id=f"{document.source_id}-chunk-{index}",
        source_id=str(document.metadata.get("source_object_name", document.source_id)),
        title=document.title,
        text=text,
        metadata={
            **document.metadata,
            "chunk_index": index,
            "chunking_strategy": strategy,
            "page_number": resolve_page_number(document.metadata, chunk_text=text, start_offset=start_offset),
        },
    )


def build_page_ranges(page_texts: Iterable[str]) -> tuple[list[dict[str, int]], str]:
    page_ranges: list[dict[str, int]] = []
    normalized_pages: list[str] = []
    cursor = 0

    for index, page_text in enumerate(page_texts, start=1):
        normalized_page = normalize_for_offset_matching(page_text)
        if not normalized_page:
            continue
        start = cursor
        end = start + len(normalized_page)
        page_ranges.append({"page_number": index, "start": start, "end": end})
        normalized_pages.append(normalized_page)
        cursor = end + 1

    return page_ranges, " ".join(normalized_pages)


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
