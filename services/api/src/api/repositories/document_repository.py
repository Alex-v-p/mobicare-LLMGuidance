from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from shared.contracts.documents import DocumentMetadata


class DocumentRepositoryError(RuntimeError):
    pass


class DocumentNotFoundError(DocumentRepositoryError):
    pass


@dataclass
class DocumentBlob:
    object_name: str
    bucket: str
    content: bytes
    content_type: str | None
    etag: str | None
    last_modified: object | None
    size_bytes: int


class DocumentRepository:
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

    def list_documents(self) -> list[DocumentMetadata]:
        self._ensure_bucket_exists()
        documents: list[DocumentMetadata] = []
        for obj in self._client.list_objects(self._documents_bucket, prefix=self._documents_prefix, recursive=True):
            if obj.is_dir:
                continue

            extension = Path(obj.object_name).suffix.lower().lstrip(".") or None
            content_type = mimetypes.guess_type(obj.object_name)[0] or "application/octet-stream"
            documents.append(
                DocumentMetadata(
                    object_name=obj.object_name,
                    title=Path(obj.object_name).name,
                    bucket=self._documents_bucket,
                    prefix=self._documents_prefix,
                    size_bytes=getattr(obj, "size", 0) or 0,
                    extension=extension,
                    content_type=content_type,
                    etag=getattr(obj, "etag", None),
                    last_modified=getattr(obj, "last_modified", None),
                )
            )

        documents.sort(key=lambda doc: doc.object_name)
        return documents

    def get_document(self, object_name: str) -> DocumentBlob:
        self._ensure_bucket_exists()
        try:
            stat = self._client.stat_object(self._documents_bucket, object_name)
            response = self._client.get_object(self._documents_bucket, object_name)
        except S3Error as exc:
            if getattr(exc, "code", "") in {"NoSuchKey", "NoSuchObject", "NoSuchVersion", "ResourceNotFound"}:
                raise DocumentNotFoundError(f"Document '{object_name}' was not found") from exc
            raise DocumentRepositoryError(str(exc)) from exc

        try:
            content = response.read()
        finally:
            response.close()
            response.release_conn()

        content_type = getattr(stat, "content_type", None) or mimetypes.guess_type(object_name)[0] or "application/octet-stream"
        return DocumentBlob(
            object_name=object_name,
            bucket=self._documents_bucket,
            content=content,
            content_type=content_type,
            etag=getattr(stat, "etag", None),
            last_modified=getattr(stat, "last_modified", None),
            size_bytes=getattr(stat, "size", 0) or len(content),
        )

    def _ensure_bucket_exists(self) -> None:
        try:
            exists = self._client.bucket_exists(self._documents_bucket)
        except S3Error as exc:
            raise DocumentRepositoryError(str(exc)) from exc

        if not exists:
            raise DocumentRepositoryError(f"Documents bucket '{self._documents_bucket}' does not exist")
