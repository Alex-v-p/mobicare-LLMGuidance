from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from minio import Minio
from pypdf import PdfReader

from shared.bootstrap import create_minio_client_from_settings, ensure_minio_bucket
from shared.config import Settings, get_settings


@dataclass
class MinioDocument:
    object_name: str
    title: str
    text: str
    metadata: dict[str, str | int]


class MinioDocumentStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._documents_bucket = self._settings.minio_documents_bucket
        self._documents_prefix = self._settings.minio_documents_prefix
        self._client = create_minio_client_from_settings(self._settings)

    @property
    def client(self) -> Minio:
        return self._client

    @property
    def documents_bucket(self) -> str:
        return self._documents_bucket

    @property
    def documents_prefix(self) -> str:
        return self._documents_prefix

    def ensure_bucket_exists(self) -> None:
        ensure_minio_bucket(self._client, self._documents_bucket)

    def list_documents(self, *, split_pdf_pages: bool = False) -> list[MinioDocument]:
        self.ensure_bucket_exists()
        documents: list[MinioDocument] = []
        for obj in self._client.list_objects(self._documents_bucket, prefix=self._documents_prefix, recursive=True):
            if obj.is_dir:
                continue
            object_name = obj.object_name
            suffix = object_name.lower().rsplit(".", 1)[-1] if "." in object_name else ""
            if suffix not in {"txt", "md", "pdf"}:
                continue

            response = self._client.get_object(self._documents_bucket, object_name)
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()

            documents.extend(self._decode_document(object_name=object_name, payload=data, split_pdf_pages=split_pdf_pages))
        return documents

    def _decode_document(self, object_name: str, payload: bytes, *, split_pdf_pages: bool) -> Iterable[MinioDocument]:
        lowered = object_name.lower()
        base_metadata: dict[str, str | int] = {
            "bucket": self._documents_bucket,
            "object_name": object_name,
        }

        if lowered.endswith((".txt", ".md")):
            yield MinioDocument(
                object_name=object_name,
                title=Path(object_name).name,
                text=payload.decode("utf-8", errors="ignore"),
                metadata=base_metadata,
            )
            return

        if lowered.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(payload))
            page_count = len(reader.pages)
            if split_pdf_pages:
                for index, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    yield MinioDocument(
                        object_name=f"{object_name}#page-{index + 1}",
                        title=Path(object_name).name,
                        text=text,
                        metadata={
                            **base_metadata,
                            "source_object_name": object_name,
                            "page_number": index + 1,
                            "page_count": page_count,
                        },
                    )
                return

            page_texts = [(page.extract_text() or "") for page in reader.pages]
            combined_text = "\n\n".join(page_texts)
            yield MinioDocument(
                object_name=object_name,
                title=Path(object_name).name,
                text=combined_text,
                metadata={**base_metadata, "page_count": page_count, "raw_page_texts": page_texts},
            )
