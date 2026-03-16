from __future__ import annotations

from inference.indexing.chunking.factory import ChunkerFactory
from inference.indexing.chunking.late_chunker import LateChunker
from inference.indexing.chunking.naive_chunker import NaiveChunker
from inference.indexing.chunking.page_indexed_chunker import PageIndexedChunker
from inference.indexing.models import SourceDocument


def test_chunker_factory_creates_expected_types_and_rejects_unknown():
    assert isinstance(ChunkerFactory.create("naive", {}), NaiveChunker)
    assert isinstance(ChunkerFactory.create("page_indexed", {}), PageIndexedChunker)
    assert isinstance(ChunkerFactory.create("late", {}), LateChunker)

    try:
        ChunkerFactory.create("weird", {})
    except ValueError as exc:
        assert "Unsupported chunking strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_naive_chunker_returns_multiple_chunks_with_page_numbers():
    document = SourceDocument(
        source_id="doc-1",
        title="Guide",
        text="alpha beta gamma delta epsilon zeta",
        metadata={
            "page_ranges": [{"page_number": 1, "start": 0, "end": 35}],
            "normalized_source_text": "alpha beta gamma delta epsilon zeta",
        },
    )

    chunks = NaiveChunker(chunk_size=12, chunk_overlap=3).chunk(document)

    assert len(chunks) >= 3
    assert chunks[0].chunk_id == "doc-1-chunk-0"
    assert all(chunk.metadata["chunking_strategy"] == "naive" for chunk in chunks)
    assert all(chunk.metadata["page_number"] == 1 for chunk in chunks)


def test_page_indexed_chunker_includes_page_suffix_and_page_metadata():
    document = SourceDocument(
        source_id="doc-2",
        title="Guide",
        text="one two three four five six",
        metadata={"page_number": 4, "source_object_name": "guidelines/b.pdf"},
    )

    chunks = PageIndexedChunker(chunk_size=10, chunk_overlap=2).chunk(document)

    assert chunks[0].chunk_id.startswith("doc-2-page-4-chunk-")
    assert chunks[0].source_id == "guidelines/b.pdf"
    assert chunks[0].metadata["page_number"] == 4


def test_late_chunker_splits_paragraphs_and_large_blocks():
    document = SourceDocument(
        source_id="doc-3",
        title="Late",
        text="Para one words here.\n\n" + ("x" * 25) + "\n\nFinal para.",
        metadata={
            "normalized_source_text": "Para one words here. " + ("x" * 25) + " Final para.",
            "page_ranges": [{"page_number": 2, "start": 0, "end": 200}],
        },
    )

    chunks = LateChunker(chunk_size=15, chunk_overlap=5).chunk(document)

    assert len(chunks) >= 3
    assert all(chunk.metadata["chunking_strategy"] == "late" for chunk in chunks)
    assert all(chunk.metadata["page_number"] == 2 for chunk in chunks)
