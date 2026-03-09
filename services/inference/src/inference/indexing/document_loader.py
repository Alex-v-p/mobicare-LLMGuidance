from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inference.storage.minio_documents import MinioDocumentStore

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(slots=True)
class LoadedDocument:
    path: str
    title: str
    text: str


class DocumentLoader:
    def __init__(self, document_store: MinioDocumentStore | None = None) -> None:
        self._document_store = document_store or MinioDocumentStore()

    def load_all(self) -> list[LoadedDocument]:
        documents: list[LoadedDocument] = []
        for obj in self._document_store.list_documents():
            suffix = Path(obj.object_name).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            if not obj.text.strip():
                continue
            documents.append(
                LoadedDocument(path=obj.object_name, title=Path(obj.object_name).stem, text=obj.text)
            )
        return documents
