from __future__ import annotations

from inference.indexing.chunking.base import DocumentChunker
from inference.indexing.models import SourceDocument, TextChunk


class NaiveChunker(DocumentChunker):
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 100) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: SourceDocument) -> list[TextChunk]:
        text = " ".join(document.text.split())
        if not text:
            return []

        chunks: list[TextChunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(len(text), start + self._chunk_size)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document.source_id}-chunk-{index}",
                        source_id=document.source_id,
                        title=document.title,
                        text=chunk_text,
                        metadata={**document.metadata, "chunk_index": index, "chunking_strategy": "naive"},
                    )
                )
            if end >= len(text):
                break
            start = max(0, end - self._chunk_overlap)
            index += 1
        return chunks
