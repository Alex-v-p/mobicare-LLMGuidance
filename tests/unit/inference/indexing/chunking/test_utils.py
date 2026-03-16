from __future__ import annotations

from inference.indexing.chunking.utils import (
    build_chunk,
    build_page_ranges,
    normalize_chunk_text,
    normalize_for_offset_matching,
    resolve_page_number,
    sliding_window_chunks,
)
from inference.indexing.models import SourceDocument


def test_normalize_helpers_collapse_whitespace():
    assert normalize_for_offset_matching("a\n\n  b\t c") == "a b c"
    assert normalize_chunk_text("  a\n b  ") == "a b"


def test_sliding_window_chunks_respects_overlap_and_non_empty():
    chunks = sliding_window_chunks("abcdefghij", chunk_size=4, chunk_overlap=2)
    assert chunks == [(0, "abcd"), (2, "cdef"), (4, "efgh"), (6, "ghij")]


def test_build_page_ranges_and_resolve_page_number_work_together():
    page_ranges, normalized = build_page_ranges(["Page one text", "Page two text"]) 
    metadata = {"page_ranges": page_ranges, "normalized_source_text": normalized}

    assert resolve_page_number(metadata, chunk_text="Page one") == 1
    assert resolve_page_number(metadata, chunk_text="two text") == 2


def test_build_chunk_uses_source_object_name_and_resolves_page_number():
    document = SourceDocument(
        source_id="doc-1",
        title="Guide",
        text="Page one text\n\nPage two text",
        metadata={
            "source_object_name": "guidelines/a.pdf",
            "page_ranges": [
                {"page_number": 1, "start": 0, "end": 13},
                {"page_number": 2, "start": 14, "end": 27},
            ],
            "normalized_source_text": "Page one text Page two text",
        },
    )

    chunk = build_chunk(document, index=0, text="Page two text", strategy="naive", start_offset=14)

    assert chunk.source_id == "guidelines/a.pdf"
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["chunking_strategy"] == "naive"
    assert chunk.metadata["page_number"] == 2
