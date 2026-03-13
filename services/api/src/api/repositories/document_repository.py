from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from minio import Minio
from minio.datatypes import Object
from minio.error import S3Error
from minio.helpers import ObjectWriteResult
from urllib3.response import HTTPResponse

from shared.contracts.documents import DocumentDeleteResponse, DocumentMetadata


class DocumentRepositoryError(RuntimeError):
    pass


class DocumentNotFoundError(DocumentRepositoryError):
    pass


@dataclass(frozen=True)
class DocumentLocation:
    bucket: str
    object_name: str


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
    def __init__(
        self,
        *,
        client: Minio,
        documents_bucket: str,
        documents_prefix: str = "",
    ) -> None:
        self._client = client
        self._documents_bucket = documents_bucket
        self._documents_prefix = documents_prefix.strip("/")

    def list_documents(self) -> list[DocumentMetadata]:
        self._ensure_bucket_exists()
        documents = [
            self._build_metadata(self._to_location(obj.object_name), obj)
            for obj in self._iter_document_objects()
        ]
        documents.sort(key=lambda doc: doc.object_name)
        return documents

    def get_document(self, object_name: str) -> DocumentBlob:
        location = self._resolve_location(object_name)
        self._ensure_bucket_exists()

        response: HTTPResponse | None = None
        try:
            stat = self._stat_object(location)
            response = self._client.get_object(location.bucket, location.object_name)
            content = response.read()
        except S3Error as exc:
            raise self._map_storage_error(exc, location.object_name) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

        content_type = self._resolve_content_type(location.object_name, getattr(stat, "content_type", None))
        return DocumentBlob(
            object_name=location.object_name,
            bucket=location.bucket,
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
        location = self._resolve_location(object_name or filename)
        resolved_content_type = self._resolve_content_type(location.object_name, content_type)
        self._validate_upload_size(size_bytes)

        try:
            self._put_object(
                location=location,
                content_stream=content_stream,
                size_bytes=size_bytes,
                content_type=resolved_content_type,
            )
            stat = self._stat_object(location)
        except S3Error as exc:
            raise self._map_storage_error(exc, location.object_name) from exc

        return self._build_metadata(location, stat, content_type=resolved_content_type)

    def delete_document(self, object_name: str) -> DocumentDeleteResponse:
        location = self._resolve_location(object_name)
        self._ensure_bucket_exists()

        try:
            self._stat_object(location)
            self._client.remove_object(location.bucket, location.object_name)
        except S3Error as exc:
            raise self._map_storage_error(exc, location.object_name) from exc

        return DocumentDeleteResponse(object_name=location.object_name, bucket=location.bucket)

    def _iter_document_objects(self) -> list[Object]:
        objects: list[Object] = []
        for obj in self._client.list_objects(
            self._documents_bucket,
            prefix=self._list_prefix(),
            recursive=True,
        ):
            if not obj.is_dir:
                objects.append(obj)
        return objects

    def _put_object(
        self,
        *,
        location: DocumentLocation,
        content_stream: BinaryIO,
        size_bytes: int,
        content_type: str,
    ) -> ObjectWriteResult:
        return self._client.put_object(
            location.bucket,
            location.object_name,
            data=content_stream,
            length=size_bytes,
            content_type=content_type,
        )

    def _stat_object(self, location: DocumentLocation) -> object:
        return self._client.stat_object(location.bucket, location.object_name)

    def _build_metadata(
        self,
        location: DocumentLocation,
        obj: object,
        *,
        content_type: str | None = None,
    ) -> DocumentMetadata:
        extension = Path(location.object_name).suffix.lower().lstrip(".") or None
        resolved_content_type = self._resolve_content_type(location.object_name, content_type or getattr(obj, "content_type", None))
        return DocumentMetadata(
            object_name=location.object_name,
            title=Path(location.object_name).name,
            bucket=location.bucket,
            prefix=self._documents_prefix,
            size_bytes=getattr(obj, "size", 0) or 0,
            extension=extension,
            content_type=resolved_content_type,
            etag=getattr(obj, "etag", None),
            last_modified=getattr(obj, "last_modified", None),
        )

    def _resolve_location(self, object_name: str) -> DocumentLocation:
        sanitized_object_name = object_name.strip().lstrip("/")
        if not sanitized_object_name:
            raise DocumentRepositoryError("Document object name must not be empty")

        if self._documents_prefix and sanitized_object_name != self._documents_prefix and not sanitized_object_name.startswith(f"{self._documents_prefix}/"):
            sanitized_object_name = f"{self._documents_prefix}/{sanitized_object_name}"

        return DocumentLocation(bucket=self._documents_bucket, object_name=sanitized_object_name)

    def _to_location(self, object_name: str) -> DocumentLocation:
        return DocumentLocation(bucket=self._documents_bucket, object_name=object_name)

    def _list_prefix(self) -> str | None:
        return self._documents_prefix or None

    def _resolve_content_type(self, object_name: str, explicit_content_type: str | None) -> str:
        return explicit_content_type or mimetypes.guess_type(object_name)[0] or "application/octet-stream"

    def _validate_upload_size(self, size_bytes: int) -> None:
        if size_bytes < 0:
            raise DocumentRepositoryError("Document size must be greater than or equal to zero")

    def _ensure_bucket_exists(self) -> None:
        try:
            exists = self._client.bucket_exists(self._documents_bucket)
        except S3Error as exc:
            raise DocumentRepositoryError(str(exc)) from exc

        if not exists:
            raise DocumentRepositoryError(f"Documents bucket '{self._documents_bucket}' does not exist")

    def _map_storage_error(self, exc: S3Error, object_name: str) -> DocumentRepositoryError:
        if getattr(exc, "code", "") in {"NoSuchKey", "NoSuchObject", "NoSuchVersion", "ResourceNotFound"}:
            return DocumentNotFoundError(f"Document '{object_name}' was not found")
        return DocumentRepositoryError(str(exc))
