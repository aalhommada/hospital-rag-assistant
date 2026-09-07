"""The web layer: page, form post, and the streaming endpoint."""

import json

import pytest
from django.urls import reverse

from assistant.models import Conversation, Message

pytestmark = pytest.mark.django_db


def sse_payloads(response):
    body = b"".join(response.streaming_content).decode()
    return [json.loads(line[len("data: ") :]) for line in body.splitlines() if line.startswith("data: ")]


def test_the_chat_page_renders(client):
    response = client.get(reverse("assistant:chat"))
    assert response.status_code == 200
    assert b"Patient information assistant" in response.content
    # The safety notice must be on the page before anyone types anything.
    assert b"In an emergency call" in response.content


def test_asking_records_the_question_and_returns_both_bubbles(client):
    response = client.post(reverse("assistant:ask"), {"question": "what are the visiting hours"})
    assert response.status_code == 200
    assert b"data-stream-url" in response.content
    message = Message.objects.get()
    assert message.role == Message.Role.USER
    assert message.text == "what are the visiting hours"


def test_an_empty_question_is_ignored(client):
    response = client.post(reverse("assistant:ask"), {"question": "   "})
    assert response.status_code == 204
    assert Message.objects.count() == 0


def test_a_very_long_question_is_truncated_not_rejected(client):
    client.post(reverse("assistant:ask"), {"question": "x" * 5000})
    assert len(Message.objects.get().text) == 1000


def test_streaming_someone_elses_question_is_a_404(client):
    other = Conversation.objects.create(session_key="somebody-else")
    stranger = Message.objects.create(
        conversation=other, role=Message.Role.USER, text="private question"
    )
    assert client.get(reverse("assistant:stream", args=[stranger.pk])).status_code == 404


def test_an_already_answered_question_replays_instead_of_regenerating(client, monkeypatch):
    """A reconnecting EventSource must not book a second appointment."""
    from assistant import views

    client.post(reverse("assistant:ask"), {"question": "what are the visiting hours"})
    question = Message.objects.get(role=Message.Role.USER)
    conversation = question.conversation
    Message.objects.create(
        conversation=conversation,
        role=Message.Role.ASSISTANT,
        text="Visiting is 14:00 to 16:00.",
        route=Message.Route.INFORMATION,
    )

    def explode(*args, **kwargs):
        raise AssertionError("respond() must not run for an answered question")

    monkeypatch.setattr(views, "respond", explode)

    payloads = sse_payloads(client.get(reverse("assistant:stream", args=[question.pk])))
    assert payloads[0]["type"] == "replace"
    assert "14:00" in payloads[0]["html"]
    assert payloads[-1]["type"] == "end"


def test_the_stream_is_sent_as_server_sent_events(client, monkeypatch):
    from assistant import views

    client.post(reverse("assistant:ask"), {"question": "hello"})
    question = Message.objects.get(role=Message.Role.USER)

    monkeypatch.setattr(
        views, "respond", lambda c, q: iter([{"type": "status", "text": "Working…"}])
    )
    response = client.get(reverse("assistant:stream", args=[question.pk]))

    assert response["Content-Type"] == "text/event-stream"
    assert response["Cache-Control"] == "no-cache"
    assert response["X-Accel-Buffering"] == "no"
    assert sse_payloads(response)[0]["text"] == "Working…"


def test_conversations_are_kept_apart_by_session(client):
    from django.test import Client

    client.post(reverse("assistant:ask"), {"question": "first browser"})
    Client().post(reverse("assistant:ask"), {"question": "second browser"})
    assert Conversation.objects.count() == 2
