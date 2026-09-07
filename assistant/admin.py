"""
Admin for conversations.

This is the audit trail. Everything is read-only on purpose: the value of a
transcript is that nobody edited it. Being able to open a conversation, see how
each question was routed, and read the exact passages behind an answer is what
makes it possible to investigate a complaint months later.
"""

from django.contrib import admin

from .models import Citation, Conversation, Message


class CitationInline(admin.TabularInline):
    model = Citation
    extra = 0
    can_delete = False
    max_num = 0
    fields = ("ordinal", "document_title", "heading_path", "similarity", "excerpt")
    readonly_fields = fields


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    can_delete = False
    max_num = 0
    fields = ("created_at", "role", "route", "refused", "text")
    readonly_fields = fields


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "started_at", "last_active_at", "message_count")
    inlines = [MessageInline]
    readonly_fields = ("session_key", "started_at", "last_active_at")

    @admin.display(description="Messages")
    def message_count(self, obj) -> int:
        return obj.messages.count()


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("created_at", "conversation", "role", "route", "refused", "snippet")
    list_filter = ("role", "route", "refused")
    search_fields = ("text", "search_query")
    inlines = [CitationInline]
    readonly_fields = (
        "conversation",
        "role",
        "text",
        "route",
        "search_query",
        "refused",
        "created_at",
    )

    @admin.display(description="Text")
    def snippet(self, obj) -> str:
        return obj.text[:90] + ("…" if len(obj.text) > 90 else "")
