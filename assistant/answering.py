"""
The engine: one question in, a stream of events out.

    route ──┬─ emergency        → fixed reply, instantly
            ├─ clinical advice  → fixed refusal
            ├─ other            → greeting
            ├─ appointment      → tool loop against the appointment book
            └─ information      → retrieve → ground → stream a cited answer

Everything is a generator so the browser can show progress. A grounded answer
takes a few seconds — routing, then embedding, then two index lookups, then
generation — and a patient staring at a blank box has no idea whether it is
working. Emitting "Searching the hospital's documents…" costs nothing and
changes how the wait feels entirely.

Three design points are worth naming.

**Emergencies never reach a model.** The reply is a constant. It has to be the
same every time, instant, and not contingent on a model behaving well today.

**Retrieval failure is a first-class outcome, not an error.** When nothing
clears the similarity floor the engine does not ask the model to try anyway.
It says so. That single decision is most of what separates an assistant a
hospital could deploy from a demo.

**Nothing is written to the database until the answer is complete.** A message
that half-streamed and then failed leaves no half-answer in the transcript.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import anthropic
from django.conf import settings
from django.db import transaction

from appointments.tools import TOOL_DEFINITIONS, execute_tool
from knowledge.retrieval import hybrid_search

from . import prompts
from .llm import LLMNotConfigured, describe_refusal, stream_messages
from .models import Citation, Conversation, Message
from .rendering import extract_cited_indices
from .router import route

logger = logging.getLogger(__name__)

# A booking conversation should need two or three tool calls. More than this
# means the model is stuck in a loop, and stopping is better than spending.
MAX_TOOL_ITERATIONS = 5


def respond(conversation: Conversation, question: str) -> Iterator[dict[str, Any]]:
    """
    Answer one question, yielding events as it goes.

    Events are dictionaries with a "type":
      status  — a short progress note for the user
      token   — a fragment of the answer
      done    — the finished Message id
      error   — something went wrong; "text" is safe to show
    """
    # Most specific first. Collapsing these into one `except` would lose the
    # difference between "retry in a moment" and "an operator must fix a
    # setting", which is the only thing the person reading the message can act
    # on.
    try:
        yield from _respond(conversation, question)
    except LLMNotConfigured as error:
        logger.error("Claude is not configured: %s", error)
        yield {"type": "error", "text": str(error)}
    except anthropic.AuthenticationError:
        logger.error("Claude rejected the API key")
        yield {
            "type": "error",
            "text": "The assistant is not set up correctly — its API key was rejected. "
            f"Please call the switchboard on {settings.HOSPITAL_SWITCHBOARD}.",
        }
    except anthropic.RateLimitError:
        logger.warning("Claude rate limit reached")
        yield {
            "type": "error",
            "text": "The assistant is busy at the moment. Please try again in a minute.",
        }
    except (anthropic.APIConnectionError, anthropic.APITimeoutError):
        logger.warning("could not reach Claude")
        yield {
            "type": "error",
            "text": "I could not reach the answering service. Please try again, or call the "
            f"switchboard on {settings.HOSPITAL_SWITCHBOARD}.",
        }
    except Exception:  # noqa: BLE001 - the browser must always get a final event
        logger.exception("answering failed")
        yield {
            "type": "error",
            "text": (
                "Something went wrong on our side. Please try again, or call the switchboard "
                f"on {settings.HOSPITAL_SWITCHBOARD}."
            ),
        }


def _respond(conversation: Conversation, question: str) -> Iterator[dict[str, Any]]:
    yield {"type": "status", "text": "Reading your question…"}
    decision = route(question, history=_history(conversation))

    if decision.intent == "emergency":
        yield from _fixed_reply(
            conversation, question, prompts.emergency_reply(), Message.Route.EMERGENCY, decision
        )
        return

    if decision.intent == "clinical_advice":
        yield from _fixed_reply(
            conversation, question, prompts.clinical_reply(), Message.Route.CLINICAL, decision
        )
        return

    if decision.intent == "other":
        yield from _fixed_reply(
            conversation, question, prompts.greeting_reply(), Message.Route.UNSUPPORTED, decision
        )
        return

    if decision.intent == "appointment":
        yield from _handle_booking(conversation, question, decision)
        return

    yield from _handle_information(conversation, question, decision)


# ---------------------------------------------------------------------------
# Information: the RAG path
# ---------------------------------------------------------------------------


def _handle_information(conversation, question, decision) -> Iterator[dict[str, Any]]:
    yield {"type": "status", "text": "Searching the hospital's documents…"}
    retrieved = hybrid_search(decision.search_query)

    if not retrieved:
        # Nothing cleared the similarity floor. Saying so is the correct
        # answer, and it costs neither a model call nor a risk of invention.
        yield from _fixed_reply(
            conversation,
            question,
            prompts.unsupported_reply(),
            Message.Route.UNSUPPORTED,
            decision,
        )
        return

    yield {
        "type": "status",
        "text": f"Found {len(retrieved)} passage{'s' if len(retrieved) > 1 else ''}. Writing an answer…",
    }

    collected: list[str] = []
    with stream_messages(
        model=settings.CLAUDE_ANSWER_MODEL,
        max_tokens=settings.CLAUDE_ANSWER_MAX_TOKENS,
        system=prompts.answering_system(),
        messages=[
            {
                "role": "user",
                "content": prompts.build_answering_user_message(question, retrieved),
            }
        ],
    ) as stream:
        for fragment in stream.text_stream:
            collected.append(fragment)
            yield {"type": "token", "text": fragment}
        final = stream.get_final_message()

    text = "".join(collected).strip()

    # A refusal arrives as a normal 200 with no content, so it has to be
    # checked for explicitly rather than discovered as an empty answer.
    if final.stop_reason == "refusal" or not text:
        logger.warning(
            "answer declined: %s", describe_refusal(final.stop_reason, final.stop_details)
        )
        yield from _fixed_reply(
            conversation, question, prompts.clinical_reply(), Message.Route.CLINICAL, decision
        )
        return

    message = _save_answer(
        conversation,
        question,
        text,
        Message.Route.INFORMATION,
        decision,
        retrieved=retrieved,
    )
    yield {"type": "done", "message_id": message.pk}


# ---------------------------------------------------------------------------
# Appointments: the tool path
# ---------------------------------------------------------------------------


def _handle_booking(conversation, question, decision) -> Iterator[dict[str, Any]]:
    yield {"type": "status", "text": "Checking the appointment book…"}

    messages: list[dict[str, Any]] = _history(conversation) + [
        {"role": "user", "content": question}
    ]
    collected: list[str] = []

    for _iteration in range(MAX_TOOL_ITERATIONS):
        turn_text: list[str] = []
        with stream_messages(
            model=settings.CLAUDE_ANSWER_MODEL,
            max_tokens=settings.CLAUDE_ANSWER_MAX_TOKENS,
            system=prompts.booking_system(),
            tools=TOOL_DEFINITIONS,
            messages=messages,
        ) as stream:
            for fragment in stream.text_stream:
                turn_text.append(fragment)
                yield {"type": "token", "text": fragment}
            final = stream.get_final_message()

        collected.extend(turn_text)

        if final.stop_reason != "tool_use":
            break

        # Run every tool the model asked for, then hand all the results back in
        # one user message — splitting them across messages teaches the model to
        # stop asking for tools in parallel.
        messages.append({"role": "assistant", "content": final.content})
        results = []
        for block in final.content:
            if block.type != "tool_use":
                continue
            yield {"type": "status", "text": _tool_status(block.name)}
            logger.info("tool call %s %s", block.name, block.input)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": execute_tool(block.name, dict(block.input)),
                }
            )
        messages.append({"role": "user", "content": results})
    else:
        logger.warning("booking loop hit the iteration limit")

    text = "".join(collected).strip() or (
        "I could not complete that booking. Please call the appointments line on "
        "020 7946 0100 and they will help."
    )
    message = _save_answer(conversation, question, text, Message.Route.APPOINTMENT, decision)
    yield {"type": "done", "message_id": message.pk}


def _tool_status(tool_name: str) -> str:
    return {
        "find_available_slots": "Looking for free appointments…",
        "book_appointment": "Booking your appointment…",
    }.get(tool_name, "Working…")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _fixed_reply(conversation, question, text, route_value, decision) -> Iterator[dict[str, Any]]:
    """Emit a canned reply as if it had streamed, so the UI has one code path."""
    for line in text.splitlines(keepends=True):
        yield {"type": "token", "text": line}
    message = _save_answer(conversation, question, text, route_value, decision, refused=True)
    yield {"type": "done", "message_id": message.pk}


@transaction.atomic
def _save_answer(
    conversation: Conversation,
    question: str,
    text: str,
    route_value: str,
    decision,
    *,
    retrieved=None,
    refused: bool = False,
) -> Message:
    message = Message.objects.create(
        conversation=conversation,
        role=Message.Role.ASSISTANT,
        text=text,
        route=route_value,
        search_query=decision.search_query if decision else "",
        refused=refused,
    )

    if retrieved:
        # Record only the passages the answer actually cited. Listing all six
        # retrieved passages under a two-sentence answer implies the answer used
        # them, which is not true and makes the citations less trustworthy.
        cited = extract_cited_indices(text)
        for ordinal in cited:
            if not 1 <= ordinal <= len(retrieved):
                continue  # a citation number the prompt never offered
            result = retrieved[ordinal - 1]
            Citation.objects.create(
                message=message,
                chunk=result.chunk,
                ordinal=ordinal,
                document_title=result.chunk.document.title,
                heading_path=result.chunk.heading_path,
                excerpt=result.chunk.text[:400],
                similarity=result.similarity,
                reviewed_on=result.chunk.document.reviewed_on,
            )

    conversation.save(update_fields=["last_active_at"])
    return message


def _history(conversation: Conversation) -> list[dict[str, str]]:
    """Previous turns, in the shape the API expects."""
    return [
        {"role": message.role, "content": message.text}
        for message in conversation.messages.exclude(text="").order_by("created_at", "pk")
    ]
