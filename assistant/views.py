"""
The web layer: three views, server-rendered.

    GET  /                      the chat page
    POST /ask/                  record a question, return the two bubbles
    GET  /stream/<id>/          stream the answer for that question

There is no JavaScript framework here and no JSON API. HTMX posts the form and
swaps in the HTML Django rendered; a few lines of vanilla JavaScript consume a
Server-Sent Events stream for the token-by-token part, because that is the one
thing a form post cannot do.

The split between `ask` and `stream` exists because a browser cannot stream a
response to a POST it made through a form. So the POST records the question and
returns immediately, and the answer arrives on a separate GET that streams.
"""

from __future__ import annotations

import json
import logging

from django.conf import settings
from django.http import Http404, HttpRequest, HttpResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from .answering import respond
from .models import Conversation, Message
from .rendering import render_answer

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 1000


def _conversation(request: HttpRequest) -> Conversation:
    """The conversation for this browser session, created on first use."""
    if not request.session.session_key:
        request.session.create()
    conversation, _ = Conversation.objects.get_or_create(session_key=request.session.session_key)
    return conversation


@require_GET
def chat(request: HttpRequest) -> HttpResponse:
    conversation = _conversation(request)
    messages = conversation.messages.prefetch_related("citations").order_by("created_at", "pk")
    return render(
        request,
        "assistant/chat.html",
        {
            "conversation": conversation,
            "messages": [_presented(message) for message in messages],
            "hospital_name": settings.HOSPITAL_NAME,
            "emergency_number": settings.HOSPITAL_EMERGENCY_NUMBER,
        },
    )


@require_POST
def ask(request: HttpRequest) -> HttpResponse:
    """Record the question and return both bubbles. The answer follows on the stream."""
    conversation = _conversation(request)
    question = (request.POST.get("question") or "").strip()[:MAX_QUESTION_LENGTH]

    if not question:
        return HttpResponse("", status=204)

    message = Message.objects.create(
        conversation=conversation, role=Message.Role.USER, text=question
    )
    return render(request, "assistant/_exchange.html", {"question": message})


@require_GET
def stream(request: HttpRequest, message_id: int) -> StreamingHttpResponse:
    """Stream the answer to one question as Server-Sent Events."""
    conversation = _conversation(request)
    try:
        question = conversation.messages.get(pk=message_id, role=Message.Role.USER)
    except Message.DoesNotExist as error:
        raise Http404("No such question in this conversation") from error

    response = StreamingHttpResponse(
        _events(conversation, question), content_type="text/event-stream"
    )
    response["Cache-Control"] = "no-cache"
    # Tells nginx not to buffer, which would otherwise hold the whole stream
    # back and deliver it in one lump.
    response["X-Accel-Buffering"] = "no"
    return response


def _events(conversation: Conversation, question: Message):
    """Adapt the answering engine's events into the SSE wire format."""
    # Guard against a reconnecting EventSource re-running a whole answer — and
    # re-booking an appointment. If this question has already been answered,
    # replay the stored answer instead of generating a new one.
    existing = (
        conversation.messages.filter(
            role=Message.Role.ASSISTANT, created_at__gte=question.created_at
        )
        .order_by("created_at", "pk")
        .first()
    )
    if existing is not None:
        yield _sse({"type": "replace", **_rendered(existing)})
        yield _sse({"type": "end"})
        return

    for event in respond(conversation, question.text):
        if event["type"] == "done":
            message = Message.objects.prefetch_related("citations").get(pk=event["message_id"])
            yield _sse({"type": "replace", **_rendered(message)})
        else:
            yield _sse(event)

    yield _sse({"type": "end"})


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _rendered(message: Message) -> dict:
    from django.template.loader import render_to_string

    return {
        "html": render_to_string("assistant/_answer.html", {"message": _presented(message)}),
    }


def _presented(message: Message) -> Message:
    """Attach the rendered HTML so templates never call the renderer themselves."""
    message.rendered = render_answer(message.text)
    return message
