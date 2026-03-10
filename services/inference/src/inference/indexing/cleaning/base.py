from __future__ import annotations

from inference.indexing.models import SourceDocument


class DocumentCleaner:
    def clean(self, document: SourceDocument) -> SourceDocument:
        raise NotImplementedError
