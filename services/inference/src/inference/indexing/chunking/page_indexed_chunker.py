from __future__ import annotations

from inference.indexing.chunking.base import DocumentChunker
from inference.indexing.models import SourceDocument, TextChunk


class PageIndexedChunker(DocumentChunker):
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: SourceDocument) -> list[TextChunk]:
        text = document.text.strip()
        if not text:
            return []

        page_number = document.metadata.get("page_number")
        suffix = f"-page-{page_number}" if page_number else "-page"

        parts = self._split_text(text)

        return [
            TextChunk(
                chunk_id=f"{document.source_id}{suffix}-chunk-{index}",
                source_id=str(document.metadata.get("source_object_name", document.source_id)),
                title=document.title,
                text=part,
                metadata={
                    **document.metadata,
                    "chunk_index": index,
                    "chunking_strategy": "page_indexed",
                },
            )
            for index, part in enumerate(parts)
        ]

    def _split_text(self, text: str) -> list[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []

        if len(cleaned) <= self._chunk_size:
            return [cleaned]

        parts: list[str] = []
        start = 0

        while start < len(cleaned):
            end = min(len(cleaned), start + self._chunk_size)
            part = cleaned[start:end].strip()
            if part:
                parts.append(part)

            if end >= len(cleaned):
                break

            start = max(0, end - self._chunk_overlap)

        return parts