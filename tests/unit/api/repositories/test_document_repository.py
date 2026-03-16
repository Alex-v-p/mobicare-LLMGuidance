from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from api.repositories.document_repository import DocumentRepository
from api.repositories.documents.errors import DocumentAlreadyExistsError, InvalidDocumentError


class FakeStorage:
    def __init__(self) -> None:
        self.objects = [
            SimpleNamespace(object_name="guidelines/z.pdf", is_dir=False, size=1, etag="z", last_modified=None, content_type="application/pdf"),
            SimpleNamespace(object_name="guidelines/a.pdf", is_dir=False, size=2, etag="a", last_modified=None, content_type="application/pdf"),
        ]
        self.exists = False
        self.put_calls: list[dict[str, object]] = []

    def list_objects(self):
        return list(self.objects)

    def stat_object(self, location):
        return SimpleNamespace(size=10, etag="etag", last_modified=None, content_type=None)

    def get_object_bytes(self, location):
        return b"pdf-data"

    def object_exists(self, location):
        return self.exists

    def put_object(self, *, location, content_stream, size_bytes, content_type):
        self.put_calls.append({"location": location, "size_bytes": size_bytes, "content_type": content_type})
        return None

    def remove_object(self, location):
        return None


@pytest.fixture
def repository() -> DocumentRepository:
    repo = DocumentRepository(client=object(), documents_bucket="docs", documents_prefix="guidelines")
    repo._storage = FakeStorage()
    return repo


def test_list_documents_returns_sorted_paginated_results(repository: DocumentRepository):
    docs, total = repository.list_documents(offset=0, limit=1)

    assert total == 2
    assert len(docs) == 1
    assert docs[0].object_name == "guidelines/a.pdf"


def test_get_document_builds_blob_with_resolved_content_type(repository: DocumentRepository):
    blob = repository.get_document("a.pdf")

    assert blob.object_name == "guidelines/a.pdf"
    assert blob.content == b"pdf-data"
    assert blob.content_type == "application/pdf"
    assert blob.size_bytes == 10


def test_upload_document_rejects_negative_size(repository: DocumentRepository):
    with pytest.raises(InvalidDocumentError):
        repository.upload_document(filename="a.pdf", content_stream=io.BytesIO(b"x"), size_bytes=-1)


def test_upload_document_respects_overwrite_flag(repository: DocumentRepository):
    repository._storage.exists = True

    with pytest.raises(DocumentAlreadyExistsError):
        repository.upload_document(filename="a.pdf", content_stream=io.BytesIO(b"x"), size_bytes=1, overwrite=False)


def test_upload_document_stores_and_returns_metadata(repository: DocumentRepository):
    document = repository.upload_document(filename="a.pdf", content_stream=io.BytesIO(b"abc"), size_bytes=3)

    assert repository._storage.put_calls[0]["content_type"] == "application/pdf"
    assert document.object_name == "guidelines/a.pdf"
    assert document.size_bytes == 10
