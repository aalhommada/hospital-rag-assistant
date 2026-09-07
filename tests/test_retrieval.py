"""End-to-end retrieval against a real Postgres with real embeddings."""

import pytest

from knowledge.models import Chunk, Document
from knowledge.retrieval import hybrid_search, keyword_search

pytestmark = pytest.mark.django_db


def test_ingestion_produces_indexed_chunks(sample_documents):
    assert Document.objects.count() == 2
    assert Chunk.objects.count() > 10
    # Every chunk must have both search representations, or hybrid search is
    # quietly running on one leg.
    assert not Chunk.objects.filter(search_vector__isnull=True).exists()
    assert all(len(chunk.embedding) == 384 for chunk in Chunk.objects.all())


def test_ingestion_is_idempotent(sample_documents):
    from pathlib import Path

    from knowledge.ingest.pipeline import ingest_file

    before = Chunk.objects.count()
    result = ingest_file(
        Path(__file__).resolve().parent.parent / "sample_data" / "visiting-hours-and-ward-rules.md"
    )
    assert result.status == "unchanged"
    assert Chunk.objects.count() == before


def test_paraphrase_is_found_without_shared_words(sample_documents):
    """The case keyword search cannot handle."""
    results = hybrid_search("am I allowed to bring a bouquet when I come and see someone")
    assert results
    assert any("flowers" in result.chunk.text.lower() for result in results)


def test_exact_token_is_found(sample_documents):
    """The case vector search is weak at."""
    results = hybrid_search("020 7946 0400")
    assert any("020 7946 0400" in result.chunk.text for result in results)


def test_off_topic_questions_retrieve_nothing(sample_documents):
    assert hybrid_search("what is the capital of France") == []
    assert hybrid_search("how do I bake sourdough bread") == []


def test_the_floor_is_what_rejects_them(sample_documents):
    """Dropping the floor returns passages, proving the gate is doing the work."""
    assert hybrid_search("what is the capital of France", min_similarity=0.0) != []


def test_empty_query_returns_nothing(sample_documents):
    assert hybrid_search("   ") == []


def test_category_filter_restricts_the_search(sample_documents):
    results = hybrid_search("what are the visiting hours", categories=["access"])
    assert all(result.chunk.document.category == "access" for result in results)


def test_keyword_search_survives_punctuation(sample_documents):
    """websearch_to_tsquery must not raise on what people actually type."""
    assert keyword_search("what's the parking?! (4 hours)", limit=5) is not None


def test_results_are_ordered_by_fused_score(sample_documents):
    results = hybrid_search("visiting hours on the children's ward")
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
