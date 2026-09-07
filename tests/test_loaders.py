"""Loaders are checked against the real sample files, not fixtures of their output."""

from pathlib import Path

import pytest

from knowledge.ingest.loaders import UnsupportedFileType, load

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "sample_data"


def headings(document):
    return [block.text for block in document.blocks if block.kind == "heading"]


def test_markdown_front_matter_is_read():
    document = load(SAMPLE_DATA / "preparing-for-an-mri-scan.md")
    assert document.title == "Preparing for an MRI scan"
    assert document.category == "preparation"
    assert document.reviewed_on is not None
    assert document.source_type == "markdown"


def test_markdown_keeps_heading_levels():
    document = load(SAMPLE_DATA / "preparing-for-an-mri-scan.md")
    assert "What to bring" in headings(document)
    assert document.blocks[0].level == 1


def test_markdown_list_items_stay_with_their_paragraph():
    """A four-item list must not become four unretrievable fragments."""
    document = load(SAMPLE_DATA / "visiting-hours-and-ward-rules.md")
    assert all(len(b.text.split()) > 3 for b in document.blocks if b.kind == "text")


def test_pdf_headings_and_paragraphs_are_recovered():
    """The reflow heuristic must rebuild structure the PDF format threw away."""
    document = load(SAMPLE_DATA / "pharmacy-and-prescriptions.pdf")
    assert document.source_type == "pdf"
    found = headings(document)
    assert "Opening hours" in found
    assert "Prescription charges" in found
    # More than one paragraph, i.e. the page was not collapsed into one block.
    assert len([b for b in document.blocks if b.kind == "text"]) > 5


def test_docx_uses_real_heading_styles():
    document = load(SAMPLE_DATA / "emergency-department-what-to-expect.docx")
    assert document.source_type == "docx"
    assert "What to bring" in headings(document)


def test_checksum_is_stable():
    path = SAMPLE_DATA / "preparing-for-a-blood-test.md"
    assert load(path).checksum == load(path).checksum


def test_unsupported_extension_is_rejected():
    with pytest.raises(UnsupportedFileType):
        load(SAMPLE_DATA.parent / "README.md.notreal")
