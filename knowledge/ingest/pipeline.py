"""
The ingestion pipeline: files in, searchable chunks out.

    load  ->  chunk  ->  embed  ->  store  ->  index

Two properties are worth more than they look:

**It is idempotent.** Every document records the SHA-256 of the file it came
from. Running ingestion again over an unchanged file does nothing, so `make
ingest` is safe to run in a loop while you are editing documents, and a nightly
sync only pays for what actually changed.

**A document is replaced atomically.** When a file does change, its old chunks
are deleted and the new ones written inside one transaction. There is no moment
where a patient can retrieve half the old leaflet and half the new one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from django.contrib.postgres.search import SearchVector
from django.db import transaction
from django.utils.text import slugify

from knowledge.embeddings import get_embedding_provider
from knowledge.models import Chunk, Document

from .chunker import chunk_blocks
from .loaders import SUPPORTED_SUFFIXES, load

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    path: Path
    title: str
    chunk_count: int
    status: str  # "created", "updated", "unchanged", or "failed"
    detail: str = ""


def ingest_file(path: Path, *, force: bool = False) -> IngestResult:
    """Run one file through the whole pipeline."""
    try:
        loaded = load(path)
    except Exception as error:  # noqa: BLE001 - one bad file must not stop the run
        logger.warning("could not load %s: %s", path, error)
        return IngestResult(
            path=path, title=path.name, chunk_count=0, status="failed", detail=str(error)
        )

    slug = slugify(path.stem)[:255]
    existing = Document.objects.filter(slug=slug).first()

    # Step 1 — skip work we have already done.
    if existing and existing.checksum == loaded.checksum and not force:
        return IngestResult(
            path=path,
            title=existing.title,
            chunk_count=existing.chunks.count(),
            status="unchanged",
        )

    # Step 2 — cut the document into passages.
    drafts = chunk_blocks(
        loaded.blocks,
        target_words=settings.CHUNK_TARGET_WORDS,
        max_words=settings.CHUNK_MAX_WORDS,
        overlap_words=settings.CHUNK_OVERLAP_WORDS,
    )
    if not drafts:
        return IngestResult(
            path=path, title=loaded.title, chunk_count=0, status="failed", detail="no text found"
        )

    # Step 3 — embed them. One batched call, not one call per chunk: the
    # difference on a few hundred chunks is seconds versus minutes.
    provider = get_embedding_provider()
    vectors = provider.embed_passages([draft.embedding_text for draft in drafts])

    # Step 4 — write. Everything below happens or nothing does.
    with transaction.atomic():
        document, created = Document.objects.update_or_create(
            slug=slug,
            defaults={
                "title": loaded.title,
                "category": loaded.category,
                "source_path": loaded.source_path,
                "source_type": loaded.source_type,
                "checksum": loaded.checksum,
                "reviewed_on": loaded.reviewed_on,
            },
        )
        document.chunks.all().delete()
        Chunk.objects.bulk_create(
            [
                Chunk(
                    document=document,
                    ordinal=draft.ordinal,
                    heading_path=draft.heading_path[:512],
                    text=draft.text,
                    word_count=draft.word_count,
                    embedding=vector,
                )
                for draft, vector in zip(drafts, vectors, strict=True)
            ]
        )
        # Step 5 — build the keyword index for this document's chunks.
        refresh_search_vectors(document)

    return IngestResult(
        path=path,
        title=document.title,
        chunk_count=len(drafts),
        status="created" if created else "updated",
    )


def refresh_search_vectors(document: Document | None = None) -> int:
    """
    Fill in the Postgres full-text column.

    Heading text is weighted above body text, so a chunk under the heading
    "Parking" outranks one that merely mentions parking in passing. Doing this
    as one UPDATE per document keeps it far cheaper than a per-row trigger
    during a bulk import.
    """
    queryset = Chunk.objects.all()
    if document is not None:
        queryset = queryset.filter(document=document)
    return queryset.update(
        search_vector=SearchVector("heading_path", weight="A", config="english")
        + SearchVector("text", weight="B", config="english")
    )


def ingest_directory(directory: Path, *, force: bool = False) -> list[IngestResult]:
    """Ingest every supported file under a directory, in a stable order."""
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    return [ingest_file(path, force=force) for path in files]
