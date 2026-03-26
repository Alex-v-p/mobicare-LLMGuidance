from __future__ import annotations

from types import SimpleNamespace

import pytest

from api.repositories.documents.models import DocumentLocation
from api.repositories.documents.storage import DocumentStorage
from minio.error import S3Error


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False
        self.released = False

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.released = True


def make_s3_error(code: str, message: str = "message", resource: str = "/docs/x.pdf") -> S3Error:
    return S3Error(
        code=code,
        message=message,
        resource=resource,
        request_id="req-123",
        host_id="host-123",
        response=None,
    )


class FakeMinio:
    def __init__(self, *, bucket_exists: bool = True) -> None:
        self._bucket_exists = bucket_exists
        self.created_buckets: list[str] = []
        self.objects = [
            SimpleNamespace(object_name="guidelines/a.pdf", is_dir=False),
            SimpleNamespace(object_name="guidelines/dir/", is_dir=True),
        ]
        self.response = FakeResponse(b"abc")

    def bucket_exists(self, bucket: str) -> bool:
        return self._bucket_exists

    def make_bucket(self, bucket: str) -> None:
        self.created_buckets.append(bucket)
        self._bucket_exists = True

    def list_objects(self, bucket: str, prefix=None, recursive=True):
        return iter(self.objects)

    def stat_object(self, bucket: str, object_name: str):
        if object_name == "missing.pdf":
            raise make_s3_error("NoSuchKey", message="missing", resource=f"/{bucket}/{object_name}")
        return SimpleNamespace(size=3, etag="etag", last_modified=None, content_type="application/pdf")

    def get_object(self, bucket: str, object_name: str):
        return self.response

    def put_object(self, bucket, object_name, data, length, content_type):
        return SimpleNamespace(bucket_name=bucket, object_name=object_name)

    def remove_object(self, bucket, object_name):
        return None


def test_ensure_bucket_exists_creates_missing_bucket():
    client = FakeMinio(bucket_exists=False)
    storage = DocumentStorage(client=client, documents_bucket="docs")

    storage.ensure_bucket_exists()

    assert client.created_buckets == ["docs"]


def test_list_objects_filters_directories():
    storage = DocumentStorage(client=FakeMinio(), documents_bucket="docs", list_prefix="guidelines")

    objects = storage.list_objects()

    assert [obj.object_name for obj in objects] == ["guidelines/a.pdf"]


def test_object_exists_returns_false_for_missing_object():
    storage = DocumentStorage(client=FakeMinio(), documents_bucket="docs")

    assert storage.object_exists(DocumentLocation(bucket="docs", object_name="missing.pdf")) is False


def test_get_object_bytes_closes_response_even_on_success():
    client = FakeMinio()
    storage = DocumentStorage(client=client, documents_bucket="docs")

    content = storage.get_object_bytes(DocumentLocation(bucket="docs", object_name="guidelines/a.pdf"))

    assert content == b"abc"
    assert client.response.closed is True
    assert client.response.released is True