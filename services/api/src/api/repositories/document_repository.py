from __future__ import annotations

from typing import BinaryIO

from minio import Minio

from api.repositories.documents import (
    DocumentBlob,
    DocumentMetadataMapper,
    DocumentNamer,
    DocumentNotFoundError,
    DocumentRepositoryError,
    DocumentStorage,
)
from shared.contracts.documents import DocumentDeleteResponse, DocumentMetadata


class DocumentRepository:
    def __init__(
        self,
        *,
        client: Minio,
        documents_bucket: str,
        documents_prefix: str = "",
    ) -> None:
        namer = DocumentNamer(
            documents_bucket=documents_bucket,
            documents_prefix=documents_prefix,
        )
        self._namer = namer
        self._storage = DocumentStorage(
            client=client,
            documents_bucket=documents_bucket,
            list_prefix=namer.list_prefix(),
        )
        self._metadata = DocumentMetadataMapper(namer)

    def list_documents(self) -> list[DocumentMetadata]:
        documents = [
            self._metadata.build(self._namer.to_location(obj.object_name), obj)
            for obj in self._storage.list_objects()
        ]
        documents.sort(key=lambda doc: doc.object_name)
        return documents

    def get_document(self, object_name: str) -> DocumentBlob:
        location = self._namer.resolve_location(object_name)
        stat = self._storage.stat_object(location)
        content = self._storage.get_object_bytes(location)
        content_type = self._namer.resolve_content_type(
            location.object_name,
            getattr(stat, "content_type", None),
        )
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
        if size_bytes < 0:
            raise DocumentRepositoryError("Document size must be greater than or equal to zero")

        location = self._namer.resolve_location(object_name or filename)
        resolved_content_type = self._namer.resolve_content_type(location.object_name, content_type)
        self._storage.put_object(
            location=location,
            content_stream=content_stream,
            size_bytes=size_bytes,
            content_type=resolved_content_type,
        )
        stat = self._storage.stat_object(location)
        return self._metadata.build(location, stat, content_type=resolved_content_type)

    def delete_document(self, object_name: str) -> DocumentDeleteResponse:
        location = self._namer.resolve_location(object_name)
        self._storage.stat_object(location)
        self._storage.remove_object(location)
        return DocumentDeleteResponse(object_name=location.object_name, bucket=location.bucket)


__all__ = [
    "DocumentBlob",
    "DocumentNotFoundError",
    "DocumentRepository",
    "DocumentRepositoryError",
]
