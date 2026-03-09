from __future__ import annotations

import re

from inference.indexing.chunking.base import DocumentChunker
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

        for block in blocks:
            # If one block is already too large, flush current first,
            # then split that block safely with sliding windows.
            if len(block) > self._chunk_size:
                if current:
                    chunks.append(self._build_chunk(document, len(chunks), current))
                    current = []
                    current_len = 0

                oversized_parts = self._split_large_block(block)
                for part in oversized_parts:
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{document.source_id}-chunk-{len(chunks)}",
                            source_id=str(document.metadata.get("source_object_name", document.source_id)),
                            title=document.title,
                            text=part,
                            metadata={**document.metadata, "chunk_index": len(chunks), "chunking_strategy": "late"},
                        )
                    )
                continue

            separator = 2 if current else 0
            if current and current_len + separator + len(block) > self._chunk_size:
                chunks.append(self._build_chunk(document, len(chunks), current))
                current = self._tail_overlap(current)
                current_len = len("\n\n".join(current)) if current else 0

            current.append(block)
            current_len = len("\n\n".join(current))

        if current:
            chunks.append(self._build_chunk(document, len(chunks), current))

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
        text = " ".join(block.split())
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

    def _build_chunk(self, document: SourceDocument, index: int, paragraphs: list[str]) -> TextChunk:
        return TextChunk(
            chunk_id=f"{document.source_id}-chunk-{index}",
            source_id=str(document.metadata.get("source_object_name", document.source_id)),
            title=document.title,
            text="\n\n".join(paragraphs).strip(),
            metadata={**document.metadata, "chunk_index": index, "chunking_strategy": "late"},
        )