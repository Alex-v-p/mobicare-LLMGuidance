from __future__ import annotations

from inference.indexing.chunking.base import DocumentChunker
from inference.indexing.chunking.utils import normalize_chunk_text, sliding_window_chunks
from inference.indexing.models import SourceDocument, TextChunk


class PageIndexedChunker(DocumentChunker):
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: SourceDocument) -> list[TextChunk]:
        text = normalize_chunk_text(document.text)
        if not text:
            return []

        page_number = document.metadata.get("page_number")
        suffix = f"-page-{page_number}" if page_number else "-page"
        chunks = sliding_window_chunks(text, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)

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
                    "page_number": page_number,
                },
            )
            for index, (_, part) in enumerate(chunks)
        ]
