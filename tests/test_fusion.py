"""Reciprocal Rank Fusion is arithmetic on ranks, so it needs no database."""

from types import SimpleNamespace

from knowledge.retrieval import reciprocal_rank_fusion


def chunk(pk: int):
    """RRF only ever touches `.pk`, so a stand-in is enough."""
    return SimpleNamespace(pk=pk)


def test_agreement_beats_a_single_first_place():
    a, b, c = chunk(1), chunk(2), chunk(3)
    fused = reciprocal_rank_fusion({"vector": [a, b], "keyword": [c, b]}, k=60)
    # b is second in both lists; a and c are first in one and absent from the other.
    assert fused[0].chunk.pk == 2
    assert fused[0].score > fused[1].score


def test_a_chunk_only_one_search_found_still_scores():
    a, b = chunk(1), chunk(2)
    fused = reciprocal_rank_fusion({"vector": [a], "keyword": [b]}, k=60)
    assert {result.chunk.pk for result in fused} == {1, 2}
    assert all(result.score > 0 for result in fused)


def test_ranks_are_recorded_for_explainability():
    a = chunk(1)
    fused = reciprocal_rank_fusion({"vector": [a], "keyword": [a]}, k=60)
    assert fused[0].vector_rank == 1
    assert fused[0].keyword_rank == 1
    assert fused[0].found_by == "both"


def test_found_by_reports_the_single_source():
    a = chunk(1)
    assert reciprocal_rank_fusion({"vector": [a]}, k=60)[0].found_by == "meaning"
    assert reciprocal_rank_fusion({"keyword": [a]}, k=60)[0].found_by == "keywords"


def test_score_matches_the_formula():
    a = chunk(1)
    fused = reciprocal_rank_fusion({"vector": [chunk(9), a]}, k=60)
    assert fused[1].chunk.pk == 1
    assert fused[1].score == 1 / (60 + 2)


def test_empty_input_is_not_an_error():
    assert reciprocal_rank_fusion({"vector": [], "keyword": []}) == []
