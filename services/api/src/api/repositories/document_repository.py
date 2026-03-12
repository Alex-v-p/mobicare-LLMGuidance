from __future__ import annotations

import io
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from shared.contracts.documents import DocumentDeleteResponse, DocumentMetadata


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
            documents.append(self._build_metadata_from_object(obj.object_name, obj))

        documents.sort(key=lambda doc: doc.object_name)
        return documents

    def get_document(self, object_name: str) -> DocumentBlob:
        normalized_object_name = self._normalize_object_name(object_name)
        self._ensure_bucket_exists()
        try:
            stat = self._client.stat_object(self._documents_bucket, normalized_object_name)
            response = self._client.get_object(self._documents_bucket, normalized_object_name)
        except S3Error as exc:
            if getattr(exc, "code", "") in {"NoSuchKey", "NoSuchObject", "NoSuchVersion", "ResourceNotFound"}:
                raise DocumentNotFoundError(f"Document '{normalized_object_name}' was not found") from exc
            raise DocumentRepositoryError(str(exc)) from exc

        try:
            content = response.read()
        finally:
            response.close()
            response.release_conn()

        content_type = getattr(stat, "content_type", None) or mimetypes.guess_type(normalized_object_name)[0] or "application/octet-stream"
        return DocumentBlob(
            object_name=normalized_object_name,
            bucket=self._documents_bucket,
            content=content,
            content_type=content_type,
            etag=getattr(stat, "etag", None),
            last_modified=getattr(stat, "last_modified", None),
            size_bytes=getattr(stat, "size", 0) or len(content),
        )

    def upload_document(
        self,
        *,
        filename: str,
        content_stream: BinaryIO,
        size_bytes: int,
        content_type: str | None = None,
        object_name: str | None = None,
    ) -> DocumentMetadata:
        self._ensure_bucket_exists()
        resolved_object_name = self._normalize_object_name(object_name or filename)
        resolved_content_type = content_type or mimetypes.guess_type(resolved_object_name)[0] or "application/octet-stream"

        if size_bytes < 0:
            raise DocumentRepositoryError("Document size must be greater than or equal to zero")

        try:
            self._client.put_object(
                self._documents_bucket,
                resolved_object_name,
                data=content_stream,
                length=size_bytes,
                content_type=resolved_content_type,
            )
            stat = self._client.stat_object(self._documents_bucket, resolved_object_name)
        except S3Error as exc:
            raise DocumentRepositoryError(str(exc)) from exc

        return self._build_metadata_from_object(resolved_object_name, stat, content_type=resolved_content_type)

    def delete_document(self, object_name: str) -> DocumentDeleteResponse:
        normalized_object_name = self._normalize_object_name(object_name)
        self._ensure_bucket_exists()

        try:
            self._client.stat_object(self._documents_bucket, normalized_object_name)
        except S3Error as exc:
            if getattr(exc, "code", "") in {"NoSuchKey", "NoSuchObject", "NoSuchVersion", "ResourceNotFound"}:
                raise DocumentNotFoundError(f"Document '{normalized_object_name}' was not found") from exc
            raise DocumentRepositoryError(str(exc)) from exc

        try:
            self._client.remove_object(self._documents_bucket, normalized_object_name)
        except S3Error as exc:
            raise DocumentRepositoryError(str(exc)) from exc

        return DocumentDeleteResponse(object_name=normalized_object_name, bucket=self._documents_bucket)

    def _build_metadata_from_object(
        self,
        object_name: str,
        obj: object,
        *,
        content_type: str | None = None,
    ) -> DocumentMetadata:
        extension = Path(object_name).suffix.lower().lstrip(".") or None
        resolved_content_type = content_type or getattr(obj, "content_type", None) or mimetypes.guess_type(object_name)[0] or "application/octet-stream"
        return DocumentMetadata(
            object_name=object_name,
            title=Path(object_name).name,
            bucket=self._documents_bucket,
            prefix=self._documents_prefix,
            size_bytes=getattr(obj, "size", 0) or 0,
            extension=extension,
            content_type=resolved_content_type,
            etag=getattr(obj, "etag", None),
            last_modified=getattr(obj, "last_modified", None),
        )

    def _normalize_object_name(self, object_name: str) -> str:
        sanitized_object_name = object_name.strip().lstrip("/")
        if not sanitized_object_name:
            raise DocumentRepositoryError("Document object name must not be empty")

        normalized_prefix = self._documents_prefix.strip("/")
        if normalized_prefix and sanitized_object_name != normalized_prefix and not sanitized_object_name.startswith(f"{normalized_prefix}/"):
            return f"{normalized_prefix}/{sanitized_object_name}"
        return sanitized_object_name

    def _ensure_bucket_exists(self) -> None:
        try:
            exists = self._client.bucket_exists(self._documents_bucket)
        except S3Error as exc:
            raise DocumentRepositoryError(str(exc)) from exc

        if not exists:
            raise DocumentRepositoryError(f"Documents bucket '{self._documents_bucket}' does not exist")
