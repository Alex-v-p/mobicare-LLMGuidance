from __future__ import annotations

import re

from inference.indexing.chunking.base import DocumentChunker
from inference.indexing.chunking.utils import normalize_for_offset_matching, resolve_page_number
from inference.indexing.models import SourceDocument, TextChunk


class LateChunker(DocumentChunker):
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 100) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: SourceDocument) -> list[TextChunk]:
        blocks = [block.strip() for block in re.split(r"\n\n+", document.text) if block.strip()]
        if not blocks:
            return []

        chunks: list[TextChunk] = []
        current: list[str] = []
        current_len = 0
        search_cursor = 0
        normalized_source_text = str(document.metadata.get("normalized_source_text") or "")

        for block in blocks:
            if len(block) > self._chunk_size:
                if current:
                    chunk = self._build_chunk(document, len(chunks), current, start_offset=search_cursor)
                    chunks.append(chunk)
                    search_cursor = self._advance_cursor(normalized_source_text, chunk.text, search_cursor)
                    current = []
                    current_len = 0

                oversized_parts = self._split_large_block(block)
                for part in oversized_parts:
                    start_offset = self._find_chunk_offset(normalized_source_text, part, search_cursor)
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{document.source_id}-chunk-{len(chunks)}",
                            source_id=str(document.metadata.get("source_object_name", document.source_id)),
                            title=document.title,
                            text=part,
                            metadata={
                                **document.metadata,
                                "chunk_index": len(chunks),
                                "chunking_strategy": "late",
                                "page_number": resolve_page_number(
                                    document.metadata,
                                    chunk_text=part,
                                    start_offset=start_offset,
                                ),
                            },
                        )
                    )
                    if start_offset >= 0:
                        search_cursor = start_offset + len(normalize_for_offset_matching(part))
                continue

            separator = 2 if current else 0
            if current and current_len + separator + len(block) > self._chunk_size:
                chunk = self._build_chunk(document, len(chunks), current, start_offset=search_cursor)
                chunks.append(chunk)
                search_cursor = self._advance_cursor(normalized_source_text, chunk.text, search_cursor)
                current = self._tail_overlap(current)
                current_len = len("\n\n".join(current)) if current else 0

            current.append(block)
            current_len = len("\n\n".join(current))

        if current:
            chunk = self._build_chunk(document, len(chunks), current, start_offset=search_cursor)
            chunks.append(chunk)

        return chunks

    def _tail_overlap(self, paragraphs: list[str]) -> list[str]:
        tail: list[str] = []
        running = 0
        for paragraph in reversed(paragraphs):
            extra = len(paragraph) + (2 if tail else 0)
            if running + extra > self._chunk_overlap:
                break
            tail.insert(0, paragraph)
            running += extra
        return tail

    def _split_large_block(self, block: str) -> list[str]:
        text = normalize_for_offset_matching(block)
        if not text:
            return []

        parts: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self._chunk_size)
            part = text[start:end].strip()
            if part:
                parts.append(part)
            if end >= len(text):
                break
            start = max(0, end - self._chunk_overlap)
        return parts

    def _build_chunk(
        self,
        document: SourceDocument,
        index: int,
        paragraphs: list[str],
        start_offset: int | None = None,
    ) -> TextChunk:
        chunk_text = "\n\n".join(paragraphs).strip()
        return TextChunk(
            chunk_id=f"{document.source_id}-chunk-{index}",
            source_id=str(document.metadata.get("source_object_name", document.source_id)),
            title=document.title,
            text=chunk_text,
            metadata={
                **document.metadata,
                "chunk_index": index,
                "chunking_strategy": "late",
                "page_number": resolve_page_number(
                    document.metadata,
                    chunk_text=chunk_text,
                    start_offset=start_offset,
                ),
            },
        )

    def _find_chunk_offset(self, normalized_source_text: str, chunk_text: str, start_cursor: int) -> int:
        if not normalized_source_text:
            return -1
        normalized_chunk = normalize_for_offset_matching(chunk_text)
        if not normalized_chunk:
            return -1
        found = normalized_source_text.find(normalized_chunk, start_cursor)
        if found >= 0:
            return found
        return normalized_source_text.find(normalized_chunk)

    def _advance_cursor(self, normalized_source_text: str, chunk_text: str, start_cursor: int) -> int:
        found = self._find_chunk_offset(normalized_source_text, chunk_text, start_cursor)
        if found < 0:
            return start_cursor
        return found + len(normalize_for_offset_matching(chunk_text))
