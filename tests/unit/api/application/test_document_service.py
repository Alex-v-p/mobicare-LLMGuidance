from unittest.mock import Mock

from api.application.services.document_service import DocumentService
from shared.contracts.documents import DocumentDeleteResponse, DocumentMetadata


def test_list_metadata_builds_pagination_response():
    repository = Mock()
    repository.list_documents.return_value = (
        [
            DocumentMetadata(
                object_name="guidelines/doc-1.pdf",
                title="doc-1.pdf",
                bucket="guidance-documents",
                size_bytes=10,
            )
        ],
        3,
    )
    service = DocumentService(repository=repository)

    response = service.list_metadata(offset=1, limit=1)

    assert response.count == 1
    assert response.total_count == 3
    assert response.offset == 1
    assert response.limit == 1
    assert response.has_more is True


def test_delete_document_delegates_to_repository():
    repository = Mock()
    repository.delete_document.return_value = DocumentDeleteResponse(
        object_name="guidelines/doc-1.pdf",
        bucket="guidance-documents",
    )
    service = DocumentService(repository=repository)

    response = service.delete_document("guidelines/doc-1.pdf")

    repository.delete_document.assert_called_once_with("guidelines/doc-1.pdf")
    assert response.status == "deleted"
