"""
Finding the passages that answer a question.

Two searches run over the same table, and their results are merged.

**Vector search** compares meaning. It finds "you must not eat for six hours
beforehand" from the question "can I have breakfast first?", because the two
sentences sit close together in embedding space even though they share no
words. It is weak at exact tokens — ward names, phone numbers, drug names.

**Keyword search** compares words, using Postgres' own full-text index. It is
excellent at exactly the things vector search is weak at, and useless at
paraphrase.

Neither is good enough alone, and each fails where the other succeeds — which
is the whole argument for running both. That is hybrid search, and it is the
cheapest quality improvement available in RAG.

Merging them is the interesting part. The two searches produce scores on scales
that cannot be compared: a cosine distance and a `ts_rank` are different units.
Reciprocal Rank Fusion sidesteps the problem by throwing the scores away and
keeping only the ranks:

    score(chunk) = sum over each search of  1 / (k + rank in that search)

A chunk both searches rank highly wins. A chunk one search loves and the other
has never heard of still scores respectably. k (60 by default, from the
original paper) flattens the curve so first place is not overwhelming.

## Ordering and relevance are two different jobs

RRF is very good at ordering and completely useless at judging relevance, and
mixing the two up is an easy mistake to make. Because its score depends only on
rank, the best result always scores 1/(60+1) = 0.0164 — whether the question
was "how much is parking?" or "what is the capital of France?". Thresholding on
the fused score therefore filters nothing.

So relevance is judged separately, and a chunk earns its place in the prompt by
clearing **either** of two independent bars.

**Semantic evidence.** Cosine similarity between the question and the chunk, on
an absolute scale. Measured on this corpus, questions the documents answer
score 0.68 to 0.91 and questions they do not score 0.41 to 0.56, so
`RETRIEVAL_MIN_SIMILARITY` sits at 0.65, inside the gap.

**Lexical evidence.** The chunk was returned by the keyword arm at all. This
second bar exists because the first one has a blind spot that would undo the
whole point of hybrid search: a query that is a bare identifier — a phone
number, "Hospital-Guest", a ward name — carries almost no semantic content, so
its best cosine similarity can sit *below* the floor even when the keyword arm
found the exact string. Searching for "020 7946 0400" scores 0.614 and would
otherwise be thrown away with the right answer in hand.

Trusting a keyword hit is safe here because `websearch_to_tsquery` requires
*every* term in the query to appear in the chunk. That is why off-topic
questions return nothing from the keyword arm at all — "how do I bake sourdough
bread" finds no match even though the colonoscopy leaflet mentions bread,
because no passage contains all of bake, sourdough, and bread.

Together, those two bars are what make "the documents do not cover this"
possible. Without them every question returns six chunks however irrelevant,
and a model handed six irrelevant passages will find a way to build an answer
out of them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank
from pgvector.django import CosineDistance

from knowledge.embeddings import get_embedding_provider
from knowledge.models import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """One search result, with enough detail to explain why it was chosen."""

    chunk: Chunk
    score: float
    similarity: float = 0.0
    vector_rank: int | None = None
    keyword_rank: int | None = None

    @property
    def found_by(self) -> str:
        if self.vector_rank is not None and self.keyword_rank is not None:
            return "both"
        if self.vector_rank is not None:
            return "meaning"
        return "keywords"


def cosine_similarity(a, b) -> float:
    """1.0 means identical direction, 0.0 unrelated. The inverse of pgvector's cosine distance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else 0.0


def vector_search(
    embedding: list[float], limit: int, categories: list[str] | None = None
) -> list[Chunk]:
    """
    Nearest neighbours by cosine distance, using the HNSW index.

    Takes an already-computed embedding rather than the raw question, so a
    single hybrid search embeds the question exactly once.
    """
    queryset = Chunk.objects.select_related("document")
    if categories:
        queryset = queryset.filter(document__category__in=categories)
    return list(
        queryset.annotate(distance=CosineDistance("embedding", embedding)).order_by("distance")[:limit]
    )


def keyword_search(query: str, limit: int, categories: list[str] | None = None) -> list[Chunk]:
    """
    Full-text search over the same rows.

    `websearch` is the forgiving query parser: it accepts what a person
    actually types, including quoted phrases and "or", and does not raise on
    punctuation the way the stricter parsers do.
    """
    search_query = SearchQuery(query, config="english", search_type="websearch")
    queryset = Chunk.objects.select_related("document").filter(search_vector=search_query)
    if categories:
        queryset = queryset.filter(document__category__in=categories)
    return list(
        queryset.annotate(rank=SearchRank("search_vector", search_query)).order_by("-rank")[:limit]
    )


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[Chunk]],
    *,
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Merge several ranked lists into one.

    Takes {"vector": [...], "keyword": [...]} and returns a single list ordered
    by fused score. Pure arithmetic on ranks — no database access and no
    embeddings, which is why it is trivial to unit test.
    """
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    chunks: dict[int, Chunk] = {}

    for source, chunk_list in ranked_lists.items():
        for position, chunk in enumerate(chunk_list):
            rank = position + 1
            scores[chunk.pk] = scores.get(chunk.pk, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(chunk.pk, {})[source] = rank
            chunks[chunk.pk] = chunk

    fused = [
        RetrievedChunk(
            chunk=chunks[pk],
            score=score,
            vector_rank=ranks[pk].get("vector"),
            keyword_rank=ranks[pk].get("keyword"),
        )
        for pk, score in scores.items()
    ]
    fused.sort(key=lambda r: (-r.score, r.chunk.pk))
    return fused


def hybrid_search(
    query: str,
    *,
    top_k: int | None = None,
    candidates: int | None = None,
    min_similarity: float | None = None,
    categories: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    The function the rest of the application calls.

    Retrieve wide, then narrow: each arm returns `candidates` results (30 by
    default), fusion orders them, and the similarity floor decides which
    deserve a place in the prompt. Searching wide costs almost nothing — both
    indexes are fast — while giving fusion enough to work with. Only the
    survivors are paid for, in prompt tokens.

    Returns an empty list when nothing clears either bar. That is not a
    failure; it is the signal the answering layer turns into an honest
    "I don't know".
    """
    top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
    candidates = candidates if candidates is not None else settings.RETRIEVAL_CANDIDATES
    if min_similarity is None:
        min_similarity = settings.RETRIEVAL_MIN_SIMILARITY

    query = (query or "").strip()
    if not query:
        return []

    query_embedding = get_embedding_provider().embed_query(query)

    fused = reciprocal_rank_fusion(
        {
            "vector": vector_search(query_embedding, candidates, categories),
            "keyword": keyword_search(query, candidates, categories),
        },
        k=settings.RRF_K,
    )

    # Score every candidate against the question on an absolute scale. The
    # embedding is already loaded with each row, so this is arithmetic, not a
    # database round trip.
    for result in fused:
        result.similarity = cosine_similarity(query_embedding, result.chunk.embedding)

    # Either bar admits a chunk: semantic closeness, or an exact lexical match
    # on every term of the query. See the module docstring for why the second
    # one is needed and why it is safe.
    kept = [
        r
        for r in fused[:top_k]
        if r.similarity >= min_similarity or r.keyword_rank is not None
    ]

    logger.info(
        "retrieval query=%r fused=%d kept=%d best_similarity=%.3f",
        query[:80],
        len(fused),
        len(kept),
        fused[0].similarity if fused else 0.0,
    )
    return kept
