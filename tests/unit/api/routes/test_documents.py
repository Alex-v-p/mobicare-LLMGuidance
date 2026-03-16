from __future__ import annotations

from fastapi.testclient import TestClient

from api.dependencies import get_document_service
from api.errors import BadRequestError
from api.main import create_app
from shared.config import get_settings


class ClosingUploadService:
    def upload_document(self, **kwargs):
        raise AssertionError("should not be called")


def test_get_document_returns_expected_headers(stub_document_service):
    app = create_app()
    app.dependency_overrides[get_document_service] = lambda: stub_document_service

    with TestClient(app) as client:
        response = client.get("/documents/guidelines/esc-heart-failure.pdf")

    assert response.status_code == 200
    assert response.content == b"sample document"
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["x-document-bucket"] == "guidance-documents"
    assert response.headers["etag"] == "etag-1"
    assert "filename*=UTF-8''guidelines/esc-heart-failure.pdf" in response.headers["content-disposition"]


def test_upload_document_rejects_disallowed_extension(monkeypatch):
    app = create_app()
    app.dependency_overrides[get_document_service] = lambda: ClosingUploadService()

    original = get_settings()
    monkeypatch.setattr(
        "api.routes.documents.get_settings",
        lambda: original.model_copy(update={"document_allowed_extensions_csv": "pdf"}),
    )

    with TestClient(app) as client:
        response = client.post(
            "/documents",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "DOCUMENT_UPLOAD_INVALID"
    assert response.json()["error"]["details"]["extension"] == "txt"
