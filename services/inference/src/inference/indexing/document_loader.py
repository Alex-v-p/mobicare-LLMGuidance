from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference.storage.minio_documents import MinioDocumentStore

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(slots=True)
class LoadedDocument:
    path: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    def __init__(self, document_store: MinioDocumentStore | None = None) -> None:
        self._document_store = document_store or MinioDocumentStore()

    def load_all(self, *, split_pdf_pages: bool = False) -> list[LoadedDocument]:
        documents: list[LoadedDocument] = []
        for obj in self._document_store.list_documents(split_pdf_pages=split_pdf_pages):
            base_object_name = obj.object_name.split("#", 1)[0]
            suffix = Path(base_object_name).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            if not obj.text.strip():
                continue
            documents.append(
                LoadedDocument(
                    path=obj.object_name,
                    title=Path(obj.title).stem,
                    text=obj.text,
                    metadata=dict(obj.metadata),
                )
            )
        return documents
