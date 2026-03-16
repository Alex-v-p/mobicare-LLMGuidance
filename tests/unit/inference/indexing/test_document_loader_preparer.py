from __future__ import annotations

from types import SimpleNamespace

from inference.indexing.document_loader import DocumentLoader
from inference.indexing.document_preparer import DocumentPreparationService
from shared.contracts.ingestion import IngestionOptions


class FakeDocumentStore:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def list_documents(self, *, split_pdf_pages: bool = False):
        self.calls.append(split_pdf_pages)
        return self.docs


def test_document_loader_filters_empty_and_unsupported_documents():
    docs = [
        SimpleNamespace(object_name="a.txt", title="a.txt", text="hello", metadata={"m": 1}),
        SimpleNamespace(object_name="b.csv", title="b.csv", text="ignored", metadata={}),
        SimpleNamespace(object_name="c.pdf#page-1", title="c.pdf", text="  ", metadata={}),
    ]
    loader = DocumentLoader(FakeDocumentStore(docs))

    loaded = loader.load_all(split_pdf_pages=True)

    assert len(loaded) == 1
    assert loaded[0].path == "a.txt"
    assert loaded[0].title == "a"


def test_document_preparer_enriches_page_ranges_and_builds_chunks():
    documents = [
        SimpleNamespace(
            path="guide.pdf",
            title="Guide",
            text="First page text\n\nSecond page text",
            metadata={"raw_page_texts": ["First page text", "Second page text"], "source_object_name": "guide.pdf"},
        )
    ]
    options = IngestionOptions(cleaning_strategy="basic", chunking_strategy="naive", chunking_params={"chunk_size": 20, "chunk_overlap": 5})

    source_documents, chunks = DocumentPreparationService().prepare_documents(documents=documents, options=options)

    assert len(source_documents) == 1
    assert source_documents[0].metadata["cleaning_strategy"] == "basic"
    assert "page_ranges" in source_documents[0].metadata
    assert source_documents[0].metadata["normalized_source_text"] == "First page text Second page text"
    assert len(chunks) >= 2
    assert chunks[0].source_id == "guide.pdf"
