from __future__ import annotations

from minio.error import S3Error


class DocumentRepositoryError(RuntimeError):
    pass


class DocumentNotFoundError(DocumentRepositoryError):
    pass


def map_storage_error(exc: S3Error, object_name: str) -> DocumentRepositoryError:
    if getattr(exc, "code", "") in {"NoSuchKey", "NoSuchObject", "NoSuchVersion", "ResourceNotFound"}:
        return DocumentNotFoundError(f"Document '{object_name}' was not found")
    return DocumentRepositoryError(str(exc))
