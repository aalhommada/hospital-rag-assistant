"""The chunker is pure logic, so it is tested without a database or a model."""

from knowledge.ingest.chunker import chunk_blocks
from knowledge.ingest.loaders import Block


def words(count: int, token: str = "word") -> str:
    return " ".join([token] * count)


def test_headings_start_new_chunks():
    chunks = chunk_blocks(
        [
            Block("heading", "Visiting hours", 1),
            Block("text", words(40, "alpha")),
            Block("heading", "Parking", 1),
            Block("text", words(40, "beta")),
        ]
    )
    assert len(chunks) == 2
    assert "alpha" in chunks[0].text and "beta" not in chunks[0].text
    assert chunks[0].heading_path == "Visiting hours"
    assert chunks[1].heading_path == "Parking"


def test_heading_path_nests_and_unwinds():
    chunks = chunk_blocks(
        [
            Block("heading", "Visiting hours", 1),
            Block("heading", "Intensive care", 2),
            Block("text", words(30, "icu")),
            Block("heading", "Maternity", 2),
            Block("text", words(30, "maternity")),
        ]
    )
    assert chunks[0].heading_path == "Visiting hours > Intensive care"
    # A second level-2 heading replaces the first rather than nesting under it.
    assert chunks[1].heading_path == "Visiting hours > Maternity"


def test_paragraphs_are_never_split():
    paragraph = words(200, "single")
    chunks = chunk_blocks([Block("text", paragraph)], target_words=50, max_words=80)
    assert len(chunks) == 1
    assert chunks[0].word_count == 200


def test_chunks_respect_the_target_size():
    blocks = [Block("text", words(50, f"p{i}")) for i in range(6)]
    chunks = chunk_blocks(blocks, target_words=100, max_words=140, overlap_words=0)
    assert len(chunks) > 1
    assert all(chunk.word_count <= 140 for chunk in chunks)


def test_no_chunk_is_only_carried_over_overlap():
    """The bug this guards: the last flush emitting a copy of the previous chunk."""
    blocks = [Block("text", words(60, f"p{i}")) for i in range(4)]
    chunks = chunk_blocks(blocks, target_words=60, max_words=100, overlap_words=25)
    texts = [chunk.text for chunk in chunks]
    assert len(texts) == len(set(texts))


def test_overlap_does_not_cross_a_heading():
    chunks = chunk_blocks(
        [
            Block("heading", "Section one", 1),
            Block("text", words(60, "alpha")),
            Block("heading", "Section two", 1),
            Block("text", words(60, "beta")),
        ],
        target_words=50,
        overlap_words=20,
    )
    assert "alpha" not in chunks[1].text


def test_embedding_text_carries_the_heading_path():
    chunks = chunk_blocks(
        [Block("heading", "Parking charges", 1), Block("text", words(30, "cost"))]
    )
    assert chunks[0].embedding_text.startswith("Parking charges")


def test_tiny_fragments_are_dropped():
    chunks = chunk_blocks([Block("heading", "Note", 1), Block("text", "Too short.")])
    assert chunks == []
