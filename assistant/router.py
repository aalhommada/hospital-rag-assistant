"""
Deciding what kind of question just arrived.

One cheap call does two jobs at once.

**Classification.** Not every message is a retrieval problem. "Book me a chest
clinic slot" is an action. "Should I stop my warfarin?" must be refused. Sending
all three down the same pipeline is how assistants end up answering clinical
questions out of a parking leaflet.

**Query rewriting.** People ask follow-up questions. "And for the children's
ward?" is meaningless to a search engine on its own, but obvious in context.
The router resolves it into a standalone query before anything is searched.
This is the whole of "conversational RAG": the conversation is used to build the
query, and only the query is used to search.

The result is a typed object, not free text, because the next thing the code
does is branch on it. `messages.parse` with a Pydantic model makes the API
guarantee the shape, so there is no JSON parsing and no defensive `.get()`.
"""

from __future__ import annotations

import logging
from typing import Literal

from django.conf import settings
from pydantic import BaseModel, Field

from .llm import get_client
from .prompts import ROUTER_SYSTEM

logger = logging.getLogger(__name__)

# How many previous turns the router sees. Enough to resolve "and for the
# children's ward?", short enough to stay cheap and to keep an old topic from
# dragging a new question sideways.
HISTORY_TURNS = 6


class RouteDecision(BaseModel):
    """The router's answer. The API validates the response against this schema."""

    intent: Literal["emergency", "clinical_advice", "appointment", "information", "other"]
    search_query: str = Field(
        description="Standalone search query with pronouns resolved, or empty string"
    )
    reason: str = Field(description="One short sentence explaining the classification")


def route(question: str, history: list[dict] | None = None) -> RouteDecision:
    """Classify a message and produce a standalone search query for it."""
    messages: list[dict] = []
    for turn in (history or [])[-HISTORY_TURNS:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    response = get_client().messages.parse(
        model=settings.CLAUDE_ROUTER_MODEL,
        max_tokens=settings.CLAUDE_ROUTER_MAX_TOKENS,
        system=ROUTER_SYSTEM,
        messages=messages,
        output_format=RouteDecision,
    )

    decision = response.parsed_output
    if decision is None:
        # The model declined to answer even this. Treat an unclassifiable
        # message as clinical: refusing something harmless is a small cost,
        # answering something clinical is not.
        logger.warning("router returned no parsed output (stop_reason=%s)", response.stop_reason)
        return RouteDecision(
            intent="clinical_advice",
            search_query="",
            reason="Could not classify the message; refusing on the safe side.",
        )

    # The model is asked for an empty query on non-information intents, but a
    # sensible default costs nothing and removes a failure mode.
    if decision.intent == "information" and not decision.search_query.strip():
        decision.search_query = question

    logger.info("routed intent=%s query=%r", decision.intent, decision.search_query[:80])
    return decision
