"""
Turning files into text we can chunk.

A loader has one job: read a file and return `LoadedDocument` — clean text plus
the metadata every later stage needs. Everything downstream is file-format
blind, which is why adding a new format later means adding one function here
and nothing else.

The interesting part is what we keep. Most naive loaders return a wall of text
and throw the structure away. We keep heading levels, because headings are how
a hospital document is organised and they end up being the single most useful
signal both for chunking and for citations.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".pdf", ".docx"}


@dataclass
class Block:
    """One heading or one paragraph, in the order it appeared in the file."""

    kind: str  # "heading" or "text"
    text: str
    level: int = 0  # heading depth: 1 for "#", 2 for "##", 0 for body text


@dataclass
class LoadedDocument:
    title: str
    source_path: str
    source_type: str
    checksum: str
    blocks: list[Block] = field(default_factory=list)
    category: str = "department"
    reviewed_on: date | None = None


class UnsupportedFileType(Exception):
    pass


def checksum_of(path: Path) -> str:
    """SHA-256 of the raw bytes, so we can skip files that have not changed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> LoadedDocument:
    """Load any supported file. Dispatches on the file extension."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return load_markdown(path)
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    raise UnsupportedFileType(f"{path.name}: only {', '.join(sorted(SUPPORTED_SUFFIXES))} are supported")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

# Optional front matter at the top of a hospital document:
#
#   ---
#   category: preparation
#   reviewed_on: 2026-03-01
#   ---
#
# Both keys are optional. `category` groups documents in the admin and lets
# retrieval be filtered later; `reviewed_on` is shown next to every citation.
FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def load_markdown(path: Path) -> LoadedDocument:
    raw = path.read_text(encoding="utf-8")
    metadata, body = _split_front_matter(raw)

    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            text = " ".join(line.strip() for line in paragraph).strip()
            if text:
                blocks.append(Block(kind="text", text=text))
            paragraph.clear()

    for line in body.splitlines():
        heading = HEADING.match(line)
        if heading:
            flush()
            blocks.append(Block(kind="heading", text=heading.group(2).strip(), level=len(heading.group(1))))
        elif line.strip():
            paragraph.append(line)
        else:
            # A blank line ends a paragraph. List items stay inside the
            # paragraph they belong to, which keeps "bring these four things"
            # together as one retrievable unit instead of four fragments.
            flush()
    flush()

    title = _first_heading(blocks) or path.stem.replace("-", " ").title()
    return LoadedDocument(
        title=title,
        source_path=str(path),
        source_type="markdown",
        checksum=checksum_of(path),
        blocks=blocks,
        category=metadata.get("category", "department"),
        reviewed_on=_parse_date(metadata.get("reviewed_on")),
    )


def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    metadata = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return metadata, raw[match.end():]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _first_heading(blocks: list[Block]) -> str | None:
    return next((b.text for b in blocks if b.kind == "heading"), None)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def load_pdf(path: Path) -> LoadedDocument:
    """
    Extract text from a PDF and rebuild its paragraphs.

    A PDF has no paragraphs and no headings — only glyphs at positions. Text
    extraction gives back one line per *visual* line, so a paragraph arrives
    already broken into pieces and has to be sewn back together. Two signals do
    the sewing, and both are about line width:

      * A line that ends a sentence **and** stops well short of the right
        margin is the last line of its paragraph. A line that ends a sentence
        but runs to the margin is just a sentence boundary mid-paragraph.
      * A short line with no closing punctuation is a heading.

    This works on plainly formatted documents, which is what hospital leaflets
    are. It will not survive a two-column research paper, and it cannot read a
    scanned page at all, because a scan contains no text to extract. Both cases
    need a layout-aware parser or a vision model — see README, "Known limits".
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    blocks: list[Block] = []
    for page in reader.pages:
        blocks.extend(_reflow_page(page.extract_text() or ""))

    # The first heading on page one is the document title, not a section.
    if blocks and blocks[0].kind == "heading":
        blocks[0].level = 1

    metadata_title = (reader.metadata.title if reader.metadata else None) or ""
    title = metadata_title.strip() or _first_heading(blocks) or path.stem.replace("-", " ").title()
    return LoadedDocument(
        title=title,
        source_path=str(path),
        source_type="pdf",
        checksum=checksum_of(path),
        blocks=blocks,
    )


def _reflow_page(text: str) -> list[Block]:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    # How wide a full line of body text is on this page. Taking a high
    # percentile rather than the maximum keeps one unusually long line from
    # skewing it.
    widths = sorted(len(line) for line in lines)
    body_width = widths[min(int(len(widths) * 0.9), len(widths) - 1)]

    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(Block(kind="text", text=" ".join(paragraph)))
            paragraph.clear()

    for line in lines:
        if _looks_like_heading(line):
            flush()
            blocks.append(Block(kind="heading", text=line, level=2))
            continue

        paragraph.append(line)
        if line.endswith((".", "!", "?")) and len(line) < body_width * 0.9:
            flush()

    flush()
    return blocks


def _looks_like_heading(line: str) -> bool:
    return (
        len(line) <= 70
        and len(line.split()) <= 10
        and not line.endswith((".", ",", ";", ":", "!", "?"))
        and line[:1].isupper()
    )


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------

def load_docx(path: Path) -> LoadedDocument:
    """Word keeps real heading styles, so this loader is the most reliable one."""
    import docx

    document = docx.Document(str(path))
    blocks: list[Block] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower()
        if style.startswith("heading"):
            level = int(style.replace("heading", "").strip() or 1)
            blocks.append(Block(kind="heading", text=text, level=level))
        elif style == "title":
            blocks.append(Block(kind="heading", text=text, level=1))
        else:
            blocks.append(Block(kind="text", text=text))

    # Tables are flattened row by row. A row read as "Department | Floor | Phone"
    # keeps its columns together, which is enough for the simple tables in
    # hospital documents.
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(Block(kind="text", text=" | ".join(cells)))

    title = _first_heading(blocks) or path.stem.replace("-", " ").title()
    return LoadedDocument(
        title=title,
        source_path=str(path),
        source_type="docx",
        checksum=checksum_of(path),
        blocks=blocks,
    )
