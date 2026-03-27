from __future__ import annotations

from urllib3.exceptions import ConnectTimeoutError, MaxRetryError, NewConnectionError

from api.infrastructure.repositories.documents.errors import (
    DocumentNotFoundError,
    DocumentStorageUnavailableError,
    map_storage_error,
)
from minio.error import S3Error


def make_s3_error(code: str, message: str = "message", resource: str = "/docs/x.pdf") -> S3Error:
    return S3Error(
        code=code,
        message=message,
        resource=resource,
        request_id="req-123",
        host_id="host-123",
        response=None,
    )


def test_map_storage_error_maps_not_found_s3_codes():
    error = make_s3_error("NoSuchKey")

    mapped = map_storage_error(error, "x.pdf")

    assert isinstance(mapped, DocumentNotFoundError)


def test_map_storage_error_maps_auth_and_timeout_errors():
    auth_error = make_s3_error("AccessDenied")
    timeout_error = ConnectTimeoutError(None, None, "timeout")

    auth_mapped = map_storage_error(auth_error, "x.pdf")
    timeout_mapped = map_storage_error(timeout_error, "x.pdf")

    assert isinstance(auth_mapped, DocumentStorageUnavailableError)
    assert auth_mapped.code == "DOCUMENT_STORAGE_AUTH_FAILED"
    assert timeout_mapped.code == "DOCUMENT_STORAGE_TIMEOUT"


def test_map_storage_error_maps_connection_and_generic_errors():
    connect_error = MaxRetryError(None, "/", NewConnectionError(None, "no route"))
    generic_error = RuntimeError("boom")

    connect_mapped = map_storage_error(connect_error, "x.pdf")
    generic_mapped = map_storage_error(generic_error, "x.pdf")

    assert connect_mapped.message == "Could not reach document storage"
    assert generic_mapped.message == "Document storage request failed"