from __future__ import annotations

from inference.indexing.models import SourceDocument, TextChunk


class DocumentChunker:
    def chunk(self, document: SourceDocument) -> list[TextChunk]:
        raise NotImplementedError
