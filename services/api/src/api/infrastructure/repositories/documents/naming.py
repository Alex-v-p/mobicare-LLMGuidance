from __future__ import annotations

import mimetypes
from pathlib import Path

from api.infrastructure.repositories.documents.errors import InvalidDocumentError
from api.infrastructure.repositories.documents.models import DocumentLocation


class DocumentNamer:
    def __init__(self, *, documents_bucket: str, documents_prefix: str = "") -> None:
        self._documents_bucket = documents_bucket
        self._documents_prefix = documents_prefix.strip("/")

    @property
    def documents_prefix(self) -> str:
        return self._documents_prefix

    def resolve_location(self, object_name: str) -> DocumentLocation:
        sanitized_object_name = object_name.strip().lstrip("/")
        if not sanitized_object_name:
            raise InvalidDocumentError("Document object name must not be empty")

        if (
            self._documents_prefix
            and sanitized_object_name != self._documents_prefix
            and not sanitized_object_name.startswith(f"{self._documents_prefix}/")
        ):
            sanitized_object_name = f"{self._documents_prefix}/{sanitized_object_name}"

        return DocumentLocation(bucket=self._documents_bucket, object_name=sanitized_object_name)

    def to_location(self, object_name: str) -> DocumentLocation:
        return DocumentLocation(bucket=self._documents_bucket, object_name=object_name)

    def list_prefix(self) -> str | None:
        return self._documents_prefix or None

    def resolve_content_type(self, object_name: str, explicit_content_type: str | None) -> str:
        return explicit_content_type or mimetypes.guess_type(object_name)[0] or "application/octet-stream"

    def extension_for(self, object_name: str) -> str | None:
        return Path(object_name).suffix.lower().lstrip(".") or None

    def title_for(self, object_name: str) -> str:
        return Path(object_name).name
