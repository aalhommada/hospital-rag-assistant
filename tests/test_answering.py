"""
The orchestration, with Claude stubbed out.

The model is replaced so these tests are fast, free, and deterministic. What is
being checked is not how well Claude writes — it is that the code around it
routes correctly, refuses when it should, records the right citations, and never
sends a clinical question to be answered from a parking leaflet. That logic is
ours, so it is ours to test.
"""

from types import SimpleNamespace

import pytest

from assistant import answering
from assistant.models import Conversation, Message
from assistant.router import RouteDecision

pytestmark = pytest.mark.django_db


class FakeStream:
    """Stands in for the SDK's streaming context manager."""

    def __init__(self, text: str, stop_reason: str = "end_turn", content=None):
        self._text = text
        self._stop_reason = stop_reason
        self._content = content or []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        # Split into a few fragments so the streaming path is genuinely exercised.
        for index in range(0, len(self._text), 12):
            yield self._text[index : index + 12]

    def get_final_message(self):
        return SimpleNamespace(
            stop_reason=self._stop_reason, content=self._content, stop_details=None
        )


@pytest.fixture
def conversation(db):
    return Conversation.objects.create(session_key="test-session")


def stub(monkeypatch, *, intent: str, answer: str = "", stop_reason: str = "end_turn", query: str = ""):
    monkeypatch.setattr(
        answering,
        "route",
        lambda question, history=None: RouteDecision(
            intent=intent, search_query=query or question, reason="stubbed"
        ),
    )
    monkeypatch.setattr(
        answering, "stream_messages", lambda **kwargs: FakeStream(answer, stop_reason)
    )


def collect(events):
    return "".join(e["text"] for e in events if e["type"] == "token")


def answer_message(conversation):
    return conversation.messages.filter(role=Message.Role.ASSISTANT).latest("pk")


# --- refusals -------------------------------------------------------------

def test_an_emergency_is_redirected_without_calling_the_model(conversation, monkeypatch, sample_documents):
    def explode(**kwargs):
        raise AssertionError("the model must not be called for an emergency")

    monkeypatch.setattr(answering, "stream_messages", explode)
    monkeypatch.setattr(
        answering,
        "route",
        lambda q, history=None: RouteDecision(intent="emergency", search_query="", reason="x"),
    )

    text = collect(answering.respond(conversation, "I have crushing chest pain"))
    assert "999" in text
    message = answer_message(conversation)
    assert message.route == Message.Route.EMERGENCY
    assert message.refused


def test_a_clinical_question_is_refused_and_never_searched(conversation, monkeypatch, sample_documents):
    def explode(*args, **kwargs):
        raise AssertionError("a clinical question must not reach retrieval")

    monkeypatch.setattr(answering, "hybrid_search", explode)
    stub(monkeypatch, intent="clinical_advice")

    text = collect(answering.respond(conversation, "Should I stop my warfarin?"))
    assert "cannot answer that" in text
    message = answer_message(conversation)
    assert message.route == Message.Route.CLINICAL
    assert message.refused


def test_nothing_retrieved_means_an_honest_i_do_not_know(conversation, monkeypatch, sample_documents):
    stub(monkeypatch, intent="information", answer="should never be used")
    monkeypatch.setattr(answering, "hybrid_search", lambda *a, **k: [])

    text = collect(answering.respond(conversation, "what is the capital of France"))
    assert "could not find that" in text
    message = answer_message(conversation)
    assert message.route == Message.Route.UNSUPPORTED
    assert message.citations.count() == 0


def test_a_model_refusal_falls_back_to_the_clinical_reply(conversation, monkeypatch, sample_documents):
    stub(monkeypatch, intent="information", answer="", stop_reason="refusal", query="visiting hours")

    text = collect(answering.respond(conversation, "something borderline"))
    assert "cannot answer that" in text
    assert answer_message(conversation).route == Message.Route.CLINICAL


# --- the grounded path ----------------------------------------------------

def test_a_grounded_answer_is_stored_with_its_citations(conversation, monkeypatch, sample_documents):
    stub(
        monkeypatch,
        intent="information",
        answer="Visiting is 14:00 to 16:00 [1] and again in the evening [2].",
        query="visiting hours",
    )

    events = list(answering.respond(conversation, "what are the visiting hours"))
    assert any(e["type"] == "status" for e in events)
    assert events[-1]["type"] == "done"

    message = answer_message(conversation)
    assert message.route == Message.Route.INFORMATION
    assert not message.refused
    assert message.search_query == "visiting hours"
    assert [c.ordinal for c in message.citations.all()] == [1, 2]
    assert all(c.document_title for c in message.citations.all())


def test_only_cited_passages_are_recorded(conversation, monkeypatch, sample_documents):
    """Six passages go into the prompt; if the answer cites one, one is shown."""
    stub(monkeypatch, intent="information", answer="Visiting is 14:00 to 16:00 [1].", query="visiting hours")
    list(answering.respond(conversation, "what are the visiting hours"))
    assert answer_message(conversation).citations.count() == 1


def test_a_citation_number_that_was_never_offered_is_ignored(conversation, monkeypatch, sample_documents):
    stub(monkeypatch, intent="information", answer="Made up [99].", query="visiting hours")
    list(answering.respond(conversation, "what are the visiting hours"))
    assert answer_message(conversation).citations.count() == 0


def test_citations_snapshot_the_document_title(conversation, monkeypatch, sample_documents):
    """Re-ingesting a document must not erase the history of what was cited."""
    from knowledge.models import Document

    stub(monkeypatch, intent="information", answer="Visiting is 14:00 [1].", query="visiting hours")
    list(answering.respond(conversation, "what are the visiting hours"))
    citation = answer_message(conversation).citations.first()
    title = citation.document_title

    Document.objects.all().delete()
    citation.refresh_from_db()
    assert citation.chunk is None
    assert citation.document_title == title


# --- failure handling -----------------------------------------------------

def test_a_missing_api_key_produces_an_actionable_message(conversation, monkeypatch, sample_documents):
    from assistant.llm import LLMNotConfigured

    def unconfigured(*args, **kwargs):
        raise LLMNotConfigured("No Claude API key is configured.")

    monkeypatch.setattr(answering, "route", unconfigured)
    events = list(answering.respond(conversation, "hello"))
    assert events[-1]["type"] == "error"
    assert "API key" in events[-1]["text"]


def test_an_unexpected_failure_still_ends_the_stream(conversation, monkeypatch, sample_documents):
    def boom(*args, **kwargs):
        raise RuntimeError("database on fire")

    monkeypatch.setattr(answering, "route", boom)
    events = list(answering.respond(conversation, "hello"))
    assert events[-1]["type"] == "error"
    assert "database on fire" not in events[-1]["text"]  # internals stay internal
