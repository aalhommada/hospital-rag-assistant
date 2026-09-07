"""
Cutting a document into retrievable passages.

Chunking is the highest-leverage decision in a RAG system, and the reason is
simple: a chunk is the smallest thing search can return. If the answer to
"how long before an MRI must I stop eating?" is split across two chunks, no
retriever, no reranker, and no model can put it back together.

So this chunker does not cut every N characters. It follows the document's own
structure:

  1. Headings define the boundaries. A new heading always starts a new chunk,
     because a hospital document changes subject at its headings.
  2. Paragraphs are never split. A paragraph is the author's own unit of
     meaning, and "bring these four things" survives only if it stays whole.
  3. Size is a budget, not a rule. Paragraphs accumulate until the chunk
     reaches its target; a single paragraph over the maximum is kept whole
     anyway rather than cut mid-sentence.
  4. A little overlap carries across, so a passage that opens by referring to
     the previous sentence still makes sense on its own.

Every chunk also remembers the headings above it. "Visiting hours > Intensive
care unit" tells the model what it is reading before it reads a word of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .loaders import Block


@dataclass
class ChunkDraft:
    ordinal: int
    heading_path: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def embedding_text(self) -> str:
        """
        What we actually embed.

        The heading path goes in front of the passage. A chunk reading "Between
        08:00 and 20:00, two visitors at a time" means very little on its own;
        prefixed with "Visiting hours > General wards" it lands close to the
        questions people really ask.
        """
        if self.heading_path:
            return f"{self.heading_path}\n\n{self.text}"
        return self.text


def chunk_blocks(
    blocks: list[Block],
    *,
    target_words: int = 180,
    max_words: int = 260,
    overlap_words: int = 30,
    min_chunk_words: int = 12,
) -> list[ChunkDraft]:
    """Turn a loaded document's blocks into chunks."""
    chunks: list[ChunkDraft] = []
    heading_stack: list[str] = []

    # New paragraphs waiting to be emitted.
    buffer: list[str] = []
    # The tail of the previous chunk, carried forward for context. Kept apart
    # from `buffer` so a flush with nothing new emits nothing — otherwise the
    # last chunk of every document would be a copy of the one before it.
    carry: list[str] = []

    def current_path() -> str:
        return " > ".join(heading_stack)

    def pending_words() -> int:
        return sum(len(part.split()) for part in buffer + carry)

    def flush(path: str) -> None:
        nonlocal buffer, carry
        if not buffer:
            return
        text = "\n\n".join(carry + buffer).strip()
        if not text:
            buffer = []
            return
        chunks.append(ChunkDraft(ordinal=len(chunks), heading_path=path, text=text))
        carry = _overlap_tail(text, overlap_words) if overlap_words else []
        buffer = []

    for block in blocks:
        if block.kind == "heading":
            # Subject change: close the current chunk before the heading moves,
            # and drop the overlap so it cannot leak into a different topic.
            flush(current_path())
            carry = []

            # Keep the stack at the right depth. A level-2 heading replaces the
            # previous level-2 and drops anything deeper, exactly like an
            # outline, so the path always reads top-down.
            depth = max(block.level - 1, 0)
            del heading_stack[depth:]
            heading_stack.append(block.text)
            continue

        paragraph = block.text.strip()
        if not paragraph:
            continue

        words = len(paragraph.split())

        # Adding this paragraph would overflow the budget, and we already have
        # something worth emitting: close the chunk first.
        if buffer and pending_words() + words > max_words:
            flush(current_path())

        buffer.append(paragraph)

        # Target reached — emit now rather than drifting toward the maximum.
        if pending_words() >= target_words:
            flush(current_path())

    flush(current_path())
    return _renumber(chunks, min_chunk_words)


def _overlap_tail(text: str, overlap_words: int) -> list[str]:
    """The last few words of a chunk, carried into the next one."""
    words = text.split()
    if len(words) <= overlap_words:
        return []
    return [" ".join(words[-overlap_words:])]


def _renumber(chunks: list[ChunkDraft], min_chunk_words: int) -> list[ChunkDraft]:
    """
    Drop fragments too small to answer anything, then renumber.

    A stray one-line paragraph under its own heading produces a chunk of three
    words. It can still match a query and would then occupy a slot in the
    prompt that a real passage should have had.
    """
    kept = [c for c in chunks if c.word_count >= min_chunk_words]
    for index, chunk in enumerate(kept):
        chunk.ordinal = index
    return kept
