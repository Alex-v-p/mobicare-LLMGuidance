from __future__ import annotations

from inference.indexing.cleaning.factory import CleanerFactory
from inference.indexing.cleaning.strategies import BasicCleaner, DeepCleaner, MedicalGuidelineDeepCleaner, NoOpCleaner
from inference.indexing.models import SourceDocument


def test_cleaner_factory_creates_expected_types_and_rejects_unknown():
    assert isinstance(CleanerFactory.create("none"), NoOpCleaner)
    assert isinstance(CleanerFactory.create("basic"), BasicCleaner)
    assert isinstance(CleanerFactory.create("deep"), DeepCleaner)
    assert isinstance(CleanerFactory.create("medical_guideline_deep"), MedicalGuidelineDeepCleaner)

    try:
        CleanerFactory.create("bad")
    except ValueError as exc:
        assert "Unsupported cleaning strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_basic_cleaner_normalizes_whitespace_and_newlines():
    document = SourceDocument(source_id="doc", title="Doc", text="a\x00\r\n\r\n\r\n b\t\t c", metadata={"x": 1})

    cleaned = BasicCleaner().clean(document)

    assert cleaned.text == "a\n\n b c"
    assert cleaned.metadata == {"x": 1}


def test_deep_cleaner_removes_urls_doi_page_markers_and_repeated_headers():
    text = """ESC Guidelines 2023
1 / 10
Keep this treatment advice.
www.example.com
DOI: 10.1000/test
Repeated Header

Repeated Header

Repeated Header
"""
    document = SourceDocument(source_id="doc", title="Doc", text=text, metadata={})

    cleaned = DeepCleaner().clean(document)

    assert "Keep this treatment advice." in cleaned.text
    assert "www.example.com" not in cleaned.text
    assert "DOI" not in cleaned.text
    assert "Repeated Header" not in cleaned.text
    assert "1 / 10" not in cleaned.text


def test_medical_guideline_deep_cleaner_preserves_recommendation_language():
    text = "Recommendation: ACE inhibitor should be considered for symptomatic patients.\n\nTable of contents"
    document = SourceDocument(source_id="doc", title="Doc", text=text, metadata={})

    cleaned = MedicalGuidelineDeepCleaner().clean(document)

    assert "ACE inhibitor should be considered" in cleaned.text
