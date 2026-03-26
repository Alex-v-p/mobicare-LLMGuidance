from __future__ import annotations

import pytest

from api.infrastructure.repositories.documents.errors import InvalidDocumentError
from api.infrastructure.repositories.documents.naming import DocumentNamer


def test_resolve_location_adds_prefix_and_strips_leading_slash():
    namer = DocumentNamer(documents_bucket="docs", documents_prefix="guidelines")

    location = namer.resolve_location("/heart/esc.pdf")

    assert location.bucket == "docs"
    assert location.object_name == "guidelines/heart/esc.pdf"


def test_resolve_location_rejects_empty_name():
    namer = DocumentNamer(documents_bucket="docs")

    with pytest.raises(InvalidDocumentError):
        namer.resolve_location("/")


def test_extension_and_title_helpers_work_for_nested_path():
    namer = DocumentNamer(documents_bucket="docs")

    assert namer.extension_for("nested/path/file.PDF") == "pdf"
    assert namer.title_for("nested/path/file.PDF") == "file.PDF"


def test_resolve_content_type_prefers_explicit_type_and_falls_back_to_guess():
    namer = DocumentNamer(documents_bucket="docs")

    assert namer.resolve_content_type("file.pdf", "application/custom") == "application/custom"
    assert namer.resolve_content_type("file.pdf", None) == "application/pdf"
    assert namer.resolve_content_type("file.unknownext", None) == "application/octet-stream"
