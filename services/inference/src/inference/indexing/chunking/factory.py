from __future__ import annotations

from shared.contracts.ingestion import ChunkingStrategy

from inference.indexing.chunking.base import DocumentChunker
from inference.indexing.chunking.late_chunker import LateChunker
from inference.indexing.chunking.naive_chunker import NaiveChunker
from inference.indexing.chunking.page_indexed_chunker import PageIndexedChunker


class ChunkerFactory:
    @staticmethod
    def create(strategy: ChunkingStrategy, params: dict[str, object]) -> DocumentChunker:
        if strategy == "naive":
            return NaiveChunker(
                chunk_size=int(params.get("chunk_size", 300)),
                chunk_overlap=int(params.get("chunk_overlap", 100)),
            )
        if strategy == "page_indexed":
            return PageIndexedChunker(
                chunk_size=params.get("chunk_size", 1200),
                chunk_overlap=params.get("chunk_overlap", 150),
            )
        if strategy == "late":
            return LateChunker(
                chunk_size=int(params.get("chunk_size", 300)),
                chunk_overlap=int(params.get("chunk_overlap", 100)),
            )
        raise ValueError(f"Unsupported chunking strategy: {strategy}")
