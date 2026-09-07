"""
Turning an answer into HTML, safely.

The model writes light markdown: paragraphs, short bullet lists, some bold, and
citation markers like [1]. This renders exactly that and nothing else.

It is written by hand rather than handed to a markdown library for one reason.
The text is not fully trusted. It is generated from passages that came out of
files somebody uploaded, so a hostile or careless document could try to smuggle
markup through the model and into another patient's browser. Escaping first and
then adding a fixed set of tags makes that impossible by construction: at no
point does any input string reach the page unescaped.
"""

from __future__ import annotations

import re

from django.utils.html import escape
from django.utils.safestring import mark_safe

BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
CITATION = re.compile(r"\[(\d{1,2})\]")
BULLET = re.compile(r"^\s*[-*]\s+")


def extract_cited_indices(text: str) -> list[int]:
    """The passage numbers an answer actually refers to, in order of first use."""
    seen: list[int] = []
    for match in CITATION.finditer(text or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def render_answer(text: str) -> str:
    """
    Render an answer to HTML. The result is safe to insert into a page.

    Lines are walked in order rather than blocks being classified whole,
    because the most common shape the model produces mixes the two:

        Bring these with you:
        - your appointment letter
        - a list of your medicines

    A block-level test asks "is this a list?", gets "no" because of the first
    line, and flattens the bullets into a paragraph.
    """
    html_parts: list[str] = []
    paragraph: list[str] = []
    bullets: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            html_parts.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_bullets() -> None:
        if bullets:
            items = "".join(f"<li>{_inline(item)}</li>" for item in bullets)
            html_parts.append(f"<ul>{items}</ul>")
            bullets.clear()

    for raw_line in (text or "").strip().splitlines():
        line = raw_line.strip()
        if not line:
            # A blank line ends whatever was open.
            flush_paragraph()
            flush_bullets()
            continue
        if BULLET.match(line):
            flush_paragraph()
            bullets.append(BULLET.sub("", line))
        else:
            flush_bullets()
            paragraph.append(line)

    flush_paragraph()
    flush_bullets()
    return mark_safe("".join(html_parts))  # noqa: S308 - every input path is escaped in _inline


def _inline(text: str) -> str:
    """Escape, then re-introduce the only three inline forms we allow."""
    safe = escape(text)
    # escape() turns ** into ** untouched, so the bold pattern still matches,
    # and its captured group is already escaped.
    safe = BOLD.sub(r"<strong>\1</strong>", safe)
    safe = CITATION.sub(
        r'<sup class="citation-marker" data-source="\1">\1</sup>',
        safe,
    )
    return safe
