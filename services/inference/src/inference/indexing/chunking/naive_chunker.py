from __future__ import annotations

from inference.indexing.chunking.base import DocumentChunker
from inference.indexing.chunking.utils import build_chunk, normalize_for_offset_matching, sliding_window_chunks
from inference.indexing.models import SourceDocument, TextChunk


class NaiveChunker(DocumentChunker):
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 100) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: SourceDocument) -> list[TextChunk]:
        text = normalize_for_offset_matching(document.text)
        if not text:
            return []

        return [
            build_chunk(document, index=index, text=chunk_text, strategy="naive", start_offset=start_offset)
            for index, (start_offset, chunk_text) in enumerate(
                sliding_window_chunks(text, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
            )
        ]
