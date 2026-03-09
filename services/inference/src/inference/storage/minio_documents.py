from __future__ import annotations

import io
import os
from dataclasses import dataclass

from minio import Minio
from pypdf import PdfReader


@dataclass
class MinioDocument:
    object_name: str
    title: str
    text: str
    metadata: dict[str, str]


class MinioDocumentStore:
    def __init__(self) -> None:
        endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000").replace("http://", "").replace("https://", "")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        self._documents_bucket = os.getenv("MINIO_DOCUMENTS_BUCKET", "guidance-documents")
        self._documents_prefix = os.getenv("MINIO_DOCUMENTS_PREFIX", "")
        self._client = Minio(
            endpoint,
            access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
            secure=secure,
        )

    @property
    def documents_bucket(self) -> str:
        return self._documents_bucket

    @property
    def documents_prefix(self) -> str:
        return self._documents_prefix

    def ensure_bucket_exists(self) -> None:
        if not self._client.bucket_exists(self._documents_bucket):
            self._client.make_bucket(self._documents_bucket)

    def list_documents(self) -> list[MinioDocument]:
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

            text = self._decode_document(object_name=object_name, payload=data)
            if not text.strip():
                continue
            documents.append(
                MinioDocument(
                    object_name=object_name,
                    title=object_name.rsplit("/", 1)[-1],
                    text=text,
                    metadata={"bucket": self._documents_bucket, "object_name": object_name},
                )
            )
        return documents

    def _decode_document(self, object_name: str, payload: bytes) -> str:
        lowered = object_name.lower()
        if lowered.endswith((".txt", ".md")):
            return payload.decode("utf-8", errors="ignore")
        if lowered.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(payload))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        return ""
