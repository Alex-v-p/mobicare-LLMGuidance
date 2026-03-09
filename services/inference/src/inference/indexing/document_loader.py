from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

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

    def load_all(self, bucket: str | None = None, prefix: str | None = None) -> list[LoadedDocument]:
        documents: list[LoadedDocument] = []
        for obj in self._document_store.list_documents(bucket=bucket, prefix=prefix):
            suffix = Path(obj.object_name).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            text = self._read_text(obj.data, suffix)
            if not text.strip():
                continue
            documents.append(
                LoadedDocument(
                    path=obj.object_name,
                    title=Path(obj.object_name).stem,
                    text=text,
                )
            )
        return documents

    def _read_text(self, data: bytes, suffix: str) -> str:
        if suffix in {".txt", ".md"}:
            return data.decode("utf-8", errors="ignore")
        if suffix == ".pdf":
            reader = PdfReader(BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        raise ValueError(f"Unsupported document type: {suffix}")
