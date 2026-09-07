"""
One place that knows how to talk to Claude.

Everything model-related is concentrated here so the rest of the application
never imports the SDK: the router asks for a classification, the answering layer
asks for a stream, and neither knows which model produced it.

Two settings are worth understanding.

**Refusal fallbacks.** A hospital corpus sits next to topics that safety
classifiers watch closely — medicines, doses, procedures. A request can come
back with `stop_reason == "refusal"` and no content at all. With fallbacks on,
the API retries the same request on a fallback model inside the same call, so a
borderline question about fasting before an operation still gets an answer
grounded in the hospital's own leaflet. Turn it off with
`CLAUDE_ENABLE_FALLBACKS=false` if you would rather see refusals directly.

**Model choice.** Both models default to `claude-opus-5`. The router does a
small classification job and is an obvious candidate for a cheaper model — but
that is a decision to make with measurements from `manage.py evaluate`, not by
guessing, which is why the default does not make it for you.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import anthropic
from django.conf import settings

logger = logging.getLogger(__name__)

# The beta flag that turns on server-side refusal fallbacks.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMNotConfigured(RuntimeError):
    """Raised when no API key is present, so the UI can say so plainly."""


# The value shipped in .env.example. Catching it here turns a confusing 401
# into a sentence that says what to do.
PLACEHOLDER_KEY = "sk-ant-..."


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    key = settings.ANTHROPIC_API_KEY.strip()
    if not key or key == PLACEHOLDER_KEY:
        raise LLMNotConfigured(
            "No Claude API key is configured. Copy .env.example to .env and set "
            "ANTHROPIC_API_KEY to a key from https://console.anthropic.com/settings/keys"
        )
    return anthropic.Anthropic(api_key=key)


def fallback_kwargs() -> dict:
    """Extra request arguments for refusal fallbacks, or nothing if disabled."""
    if not settings.CLAUDE_ENABLE_FALLBACKS:
        return {}
    return {"betas": [FALLBACK_BETA], "fallbacks": "default"}


def stream_messages(**kwargs):
    """
    Open a streaming request, using the beta endpoint only when fallbacks are on.

    Returns the SDK's stream context manager, so callers use it with `with`.
    """
    client = get_client()
    extra = fallback_kwargs()
    if extra:
        return client.beta.messages.stream(**kwargs, **extra)
    return client.messages.stream(**kwargs)


# Published price per million tokens, as (input, output). Used only to turn the
# token counts below into a number you can reason about — it is a convenience,
# not a billing record, and it will drift as prices change.
PRICES_PER_MTOK = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


def log_usage(label: str, model: str, usage) -> float:
    """
    Log what one call cost, and return the estimate in dollars.

    Worth having because model choice is the single biggest cost lever in this
    project, and it should be decided from numbers rather than intuition. Run
    the app with `INFO` logging and every call prints its own price.

    Note what the numbers include: with adaptive thinking on, reasoning tokens
    are billed as output even though they never reach the patient. That is
    invisible in the transcript and very visible on the bill.
    """
    if usage is None:
        return 0.0

    read = getattr(usage, "input_tokens", 0) or 0
    written = getattr(usage, "output_tokens", 0) or 0
    cached = getattr(usage, "cache_read_input_tokens", 0) or 0

    input_price, output_price = PRICES_PER_MTOK.get(model, (0.0, 0.0))
    # Cached input is billed at roughly a tenth of the normal rate.
    cost = (read * input_price + cached * input_price * 0.1 + written * output_price) / 1_000_000

    logger.info(
        "usage %s model=%s in=%d cached=%d out=%d est=$%.5f",
        label,
        model,
        read,
        cached,
        written,
        cost,
    )
    return cost


def describe_refusal(stop_reason: str | None, stop_details) -> str:
    """A log line for a refused request. Never shown to a patient."""
    if stop_reason != "refusal":
        return ""
    category = getattr(stop_details, "category", None)
    return f"model declined (category={category!r})"
