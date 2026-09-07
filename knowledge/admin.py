"""
Admin for the knowledge base.

Worth having for a reason beyond convenience: the people who own hospital
documents are not engineers. Django's admin gives them a place to see which
leaflets are indexed, how many passages each produced, when each was last
reviewed, and — through the re-ingest action — a way to push a corrected file
live without anyone opening a terminal.
"""

from pathlib import Path

from django.contrib import admin, messages
from django.db.models import Count

from knowledge.ingest.pipeline import ingest_file
from knowledge.models import Chunk, Document


class ChunkInline(admin.TabularInline):
    model = Chunk
    extra = 0
    can_delete = False
    fields = ("ordinal", "heading_path", "word_count", "preview")
    readonly_fields = fields
    max_num = 0  # view only: chunks are produced by ingestion, never typed in

    @admin.display(description="Text")
    def preview(self, obj: Chunk) -> str:
        return obj.text[:160] + ("…" if len(obj.text) > 160 else "")


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "category", "source_type", "chunk_total", "reviewed_on", "ingested_at")
    list_filter = ("category", "source_type")
    search_fields = ("title", "source_path")
    readonly_fields = ("checksum", "ingested_at", "source_path", "source_type")
    inlines = [ChunkInline]
    actions = ["reingest"]

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(_chunks=Count("chunks"))

    @admin.display(description="Chunks", ordering="_chunks")
    def chunk_total(self, obj) -> int:
        return obj._chunks

    @admin.action(description="Re-ingest the selected documents from disk")
    def reingest(self, request, queryset) -> None:
        done, missing = 0, []
        for document in queryset:
            path = Path(document.source_path)
            if not path.exists():
                missing.append(document.title)
                continue
            ingest_file(path, force=True)
            done += 1
        if done:
            self.message_user(request, f"Re-ingested {done} document(s).", messages.SUCCESS)
        if missing:
            self.message_user(
                request,
                "Source file missing for: " + ", ".join(missing),
                messages.WARNING,
            )


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("document", "ordinal", "heading_path", "word_count")
    list_filter = ("document__category",)
    search_fields = ("text", "heading_path", "document__title")
    # The embedding is 384 floats — useless on screen and slow to render.
    exclude = ("embedding", "search_vector")
    readonly_fields = ("document", "ordinal", "heading_path", "text", "word_count")
