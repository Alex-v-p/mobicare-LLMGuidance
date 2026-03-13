from __future__ import annotations

from inference.indexing.chunking.naive_chunker import NaiveChunker


class BasicChunker(NaiveChunker):
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
