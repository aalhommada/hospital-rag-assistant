"""
Conversation history, and the evidence behind every answer.

Storing the citations rather than only the answer text is what makes this
auditable. Six months after the fact you can still see which passage of which
document produced a given sentence — which is the difference between a hospital
being able to defend an answer and having to apologise for it.

Note that `Citation` keeps its own copy of the document title and heading. The
foreign key alone is not enough: re-ingesting a changed document deletes its
chunks and creates new ones, so a citation that only pointed at a chunk id would
quietly lose its meaning the next time a leaflet was updated.
"""

from __future__ import annotations

from django.db import models

from knowledge.models import Chunk


class Conversation(models.Model):
    """One patient's chat. Tied to a browser session, with no login required."""

    session_key = models.CharField(max_length=64, db_index=True)
    started_at = models.DateTimeField(auto_now_add=True)
    last_active_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-last_active_at"]

    def __str__(self) -> str:
        return f"Conversation {self.pk} ({self.messages.count()} messages)"


class Message(models.Model):
    class Role(models.TextChoices):
        USER = "user", "Patient"
        ASSISTANT = "assistant", "Assistant"

    class Route(models.TextChoices):
        """How the question was handled. Recorded so routing can be measured."""

        INFORMATION = "information", "Answered from documents"
        APPOINTMENT = "appointment", "Handled by the booking tools"
        CLINICAL = "clinical", "Refused — clinical advice"
        EMERGENCY = "emergency", "Refused — urgent, redirected"
        UNSUPPORTED = "unsupported", "Refused — outside the documents"
        ERROR = "error", "Something went wrong"

    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=16, choices=Role.choices)
    text = models.TextField(blank=True)

    # Assistant messages only.
    route = models.CharField(max_length=16, choices=Route.choices, blank=True)
    search_query = models.TextField(blank=True, help_text="The standalone query actually searched")
    refused = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "pk"]

    def __str__(self) -> str:
        return f"{self.get_role_display()}: {self.text[:60]}"

    @property
    def is_pending(self) -> bool:
        return self.role == self.Role.ASSISTANT and not self.text and not self.refused


class Citation(models.Model):
    """One numbered source under an answer."""

    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name="citations")

    # SET_NULL, not CASCADE: re-ingesting a document must not erase the history
    # of what was said and why.
    chunk = models.ForeignKey(Chunk, on_delete=models.SET_NULL, null=True, blank=True)

    ordinal = models.PositiveSmallIntegerField(help_text="The [1] the answer refers to")
    document_title = models.CharField(max_length=255)
    heading_path = models.CharField(max_length=512, blank=True)
    excerpt = models.TextField(blank=True)
    similarity = models.FloatField(default=0.0)
    reviewed_on = models.DateField(null=True, blank=True)

    class Meta:
        ordering = ["message_id", "ordinal"]
        constraints = [
            models.UniqueConstraint(fields=["message", "ordinal"], name="unique_citation_ordinal")
        ]

    def __str__(self) -> str:
        return f"[{self.ordinal}] {self.document_title}"

    @property
    def label(self) -> str:
        return (
            f"{self.document_title} — {self.heading_path}"
            if self.heading_path
            else self.document_title
        )
