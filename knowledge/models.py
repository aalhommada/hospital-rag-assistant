"""
The knowledge base: hospital documents, and the chunks we actually search.

Two tables carry the whole retrieval layer.

`Document` is the thing a human recognises — "Preparing for an MRI scan". It
never gets searched directly; it exists so every answer can name its source and
so re-ingesting a changed file can replace exactly one document's chunks.

`Chunk` is a searchable passage. Each row stores the same passage three ways:

  text           the words, which end up in the prompt
  embedding      a 384-number vector, for "find me passages that mean this"
  search_vector  Postgres' own keyword index, for "find me passages that say this"

Storing both search representations in the same table is what makes hybrid
search cheap here — no second database, no synchronisation problem.
"""

from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models
from pgvector.django import HnswIndex, VectorField

# The width of one embedding. This number is compiled into the database column,
# so it cannot be changed with a settings edit alone — see README, "Changing the
# embedding model". It matches the default local model, BAAI/bge-small-en-v1.5.
EMBEDDING_DIMENSIONS = 384


class Document(models.Model):
    """One source file from the hospital's document set."""

    class Category(models.TextChoices):
        PREPARATION = "preparation", "Preparing for a procedure"
        VISITING = "visiting", "Visiting and wards"
        DEPARTMENT = "department", "Departments and locations"
        ADMINISTRATION = "administration", "Admission, billing, and records"
        ACCESS = "access", "Travel, parking, and accessibility"

    class SourceType(models.TextChoices):
        MARKDOWN = "markdown", "Markdown"
        PDF = "pdf", "PDF"
        DOCX = "docx", "Word document"

    title = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255, unique=True)
    category = models.CharField(
        max_length=32, choices=Category.choices, default=Category.DEPARTMENT
    )
    source_path = models.CharField(max_length=1024)
    source_type = models.CharField(max_length=16, choices=SourceType.choices)

    # SHA-256 of the raw file. Re-ingesting a file whose checksum is unchanged
    # is skipped, which keeps `make ingest` fast and idempotent.
    checksum = models.CharField(max_length=64)

    # Hospital instructions go out of date, and a stale answer about fasting
    # before surgery is worse than no answer. The review date is surfaced with
    # every citation so a patient can see how fresh the guidance is.
    reviewed_on = models.DateField(null=True, blank=True)

    ingested_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["title"]

    def __str__(self) -> str:
        return self.title


class Chunk(models.Model):
    """A single passage of a document, small enough to put in a prompt."""

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")

    # Position within the document, so chunks can be shown back in reading order.
    ordinal = models.PositiveIntegerField()

    # The headings above this passage, joined with " > ". Carried into the
    # prompt because "Visiting hours > Intensive care" tells the model far more
    # about what it is reading than the passage alone does.
    heading_path = models.CharField(max_length=512, blank=True)

    text = models.TextField()
    word_count = models.PositiveIntegerField(default=0)

    embedding = VectorField(dimensions=EMBEDDING_DIMENSIONS)

    # Populated after insert with a single bulk UPDATE — see ingest/pipeline.py.
    search_vector = SearchVectorField(null=True, blank=True)

    class Meta:
        ordering = ["document_id", "ordinal"]
        constraints = [
            models.UniqueConstraint(fields=["document", "ordinal"], name="unique_chunk_ordinal")
        ]
        indexes = [
            # HNSW is an approximate index: it trades a little recall for search
            # that stays fast as the table grows. vector_cosine_ops must match
            # the distance function used at query time (CosineDistance).
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
            # The keyword half of hybrid search.
            GinIndex(name="chunk_search_vector_gin", fields=["search_vector"]),
        ]

    def __str__(self) -> str:
        return f"{self.document.title} #{self.ordinal}"

    @property
    def citation_label(self) -> str:
        """What a patient sees under an answer."""
        if self.heading_path:
            return f"{self.document.title} — {self.heading_path}"
        return self.document.title
