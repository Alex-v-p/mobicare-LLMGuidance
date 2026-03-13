from __future__ import annotations

from dataclasses import dataclass


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
