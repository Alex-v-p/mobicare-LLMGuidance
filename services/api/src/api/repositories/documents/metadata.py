from __future__ import annotations

from shared.contracts.documents import DocumentMetadata

from api.repositories.documents.naming import DocumentNamer
from api.repositories.documents.models import DocumentLocation


class DocumentMetadataMapper:
    def __init__(self, namer: DocumentNamer) -> None:
        self._namer = namer

    def build(
        self,
        location: DocumentLocation,
        obj: object,
        *,
        content_type: str | None = None,
    ) -> DocumentMetadata:
        resolved_content_type = self._namer.resolve_content_type(
            location.object_name,
            content_type or getattr(obj, "content_type", None),
        )
        return DocumentMetadata(
            object_name=location.object_name,
            title=self._namer.title_for(location.object_name),
            bucket=location.bucket,
            prefix=self._namer.documents_prefix,
            size_bytes=getattr(obj, "size", 0) or 0,
            extension=self._namer.extension_for(location.object_name),
            content_type=resolved_content_type,
            etag=getattr(obj, "etag", None),
            last_modified=getattr(obj, "last_modified", None),
        )
