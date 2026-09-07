"""
`manage.py evaluate`

Scores the assistant against `evaluation/questions.yaml`.

Two things are measured, and they can be run separately on purpose:

**Retrieval** needs no API key. For every question the documents answer, did
search find the right document, and did the passages it returned actually
contain the fact needed to answer? And for questions nothing answers, did
search correctly return nothing? This is the loop you want while tuning
chunking, embeddings, or the similarity floor — it is free and it takes seconds.

**Routing** needs a key, and runs with `--with-router`. It checks that clinical
questions are refused, emergencies are redirected, and booking requests reach
the tools rather than the search index.

The similarity report at the end is what you use to re-tune
RETRIEVAL_MIN_SIMILARITY: it prints the range for questions that should match
and the range for questions that should not. A healthy corpus shows a clear gap
between them, and the setting belongs inside it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from knowledge.retrieval import hybrid_search

QUESTIONS_FILE = Path(settings.BASE_DIR) / "evaluation" / "questions.yaml"

# Router intents that must never be answered from documents.
REFUSAL_INTENTS = {"clinical_advice", "emergency"}


class Command(BaseCommand):
    help = "Score retrieval, and optionally routing, against the labelled question set."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--with-router",
            action="store_true",
            help="Also check routing accuracy. Needs ANTHROPIC_API_KEY and costs one call per question.",
        )
        parser.add_argument(
            "--questions", default=str(QUESTIONS_FILE), help="Path to the question set"
        )
        parser.add_argument(
            "--verbose-failures",
            action="store_true",
            help="Show what was retrieved for each failure",
        )

    def handle(self, *args, **options) -> None:
        path = Path(options["questions"])
        if not path.exists():
            raise CommandError(f"{path} does not exist")
        rows = yaml.safe_load(path.read_text())

        self.stdout.write(self.style.MIGRATE_HEADING(f"\nRetrieval — {len(rows)} questions\n"))
        retrieval = self._score_retrieval(rows, options["verbose_failures"])

        if options["with_router"]:
            self.stdout.write(self.style.MIGRATE_HEADING("\nRouting\n"))
            self._score_routing(rows)

        self._similarity_report(retrieval)

    # -- retrieval ---------------------------------------------------------

    def _score_retrieval(self, rows: list[dict], verbose: bool) -> dict:
        answerable = [r for r in rows if r["intent"] == "information"]
        out_of_scope = [r for r in rows if r["intent"] == "out_of_scope"]

        found, grounded, failures = 0, 0, []
        answerable_similarities, out_of_scope_similarities = [], []

        for row in answerable:
            hits = hybrid_search(row["question"])
            best = max((h.similarity for h in hits), default=0.0)
            answerable_similarities.append(best)

            slugs = {h.chunk.document.slug for h in hits}
            document_found = row["document"] in slugs
            found += document_found

            phrase = str(row.get("must_contain", "")).lower()
            has_phrase = (
                any(phrase in h.chunk.text.lower() for h in hits) if phrase else document_found
            )
            grounded += has_phrase

            if not (document_found and has_phrase):
                failures.append((row, hits, document_found, has_phrase))

        refused_correctly = 0
        for row in out_of_scope:
            hits = hybrid_search(row["question"])
            # Measure how close the corpus came, even though the gate rejected it.
            probe = hybrid_search(row["question"], min_similarity=0.0)
            out_of_scope_similarities.append(max((h.similarity for h in probe), default=0.0))
            if not hits:
                refused_correctly += 1
            else:
                failures.append((row, hits, False, False))

        self._line("Right document retrieved", found, len(answerable))
        self._line("Answer actually present in the passages", grounded, len(answerable))
        self._line(
            "Out-of-scope questions correctly rejected", refused_correctly, len(out_of_scope)
        )

        if failures:
            self.stdout.write(self.style.WARNING(f"\n{len(failures)} question(s) to look at:"))
            for row, hits, document_found, has_phrase in failures:
                reason = self._failure_reason(row, document_found, has_phrase)
                self.stdout.write(f"  - {row['question']}\n      {reason}")
                if verbose:
                    for hit in hits[:3]:
                        self.stdout.write(
                            f"        {hit.similarity:.3f} {hit.chunk.document.slug} :: {hit.chunk.heading_path}"
                        )

        return {
            "answerable": answerable_similarities,
            "out_of_scope": out_of_scope_similarities,
        }

    def _failure_reason(self, row: dict, document_found: bool, has_phrase: bool) -> str:
        if row["intent"] == "out_of_scope":
            return "expected no passages above the similarity floor, but got some"
        if not document_found:
            return f"expected document '{row['document']}' was not retrieved"
        return f"document found, but no passage contained {row.get('must_contain')!r}"

    # -- routing -----------------------------------------------------------

    def _score_routing(self, rows: list[dict]) -> None:
        from assistant.router import route

        # out_of_scope questions are handled by the retrieval gate, not the
        # router, so any non-refusal classification is acceptable for them.
        checkable = [r for r in rows if r["intent"] != "out_of_scope"]
        correct, mistakes, unsafe = 0, [], 0

        for row in checkable:
            decision = route(row["question"])
            expected = "information" if row["intent"] == "information" else row["intent"]
            if decision.intent == expected:
                correct += 1
                continue
            mistakes.append((row, decision))
            # The mistake that matters: something that should have been refused
            # was sent to be answered.
            if row["intent"] in REFUSAL_INTENTS and decision.intent not in REFUSAL_INTENTS:
                unsafe += 1

        self._line("Routed correctly", correct, len(checkable))
        if unsafe:
            self.stdout.write(
                self.style.ERROR(
                    f"  {unsafe} question(s) that should have been refused were not. "
                    "Fix this before anything else."
                )
            )
        for row, decision in mistakes:
            self.stdout.write(
                f"  - {row['question']}\n"
                f"      expected {row['intent']}, got {decision.intent} — {decision.reason}"
            )

    # -- reporting ---------------------------------------------------------

    def _line(self, label: str, value: int, total: int) -> None:
        if total == 0:
            return
        percentage = 100 * value / total
        style = self.style.SUCCESS if percentage >= 90 else self.style.WARNING
        self.stdout.write(f"  {label:<44} {style(f'{value}/{total}  ({percentage:.0f}%)')}")

    def _similarity_report(self, similarities: dict) -> None:
        answerable = similarities["answerable"]
        out_of_scope = similarities["out_of_scope"]
        if not answerable or not out_of_scope:
            return

        floor = settings.RETRIEVAL_MIN_SIMILARITY
        lowest_answerable = min(answerable)
        highest_noise = max(out_of_scope)

        self.stdout.write(self.style.MIGRATE_HEADING("\nSimilarity floor\n"))
        self.stdout.write(
            f"  Questions the documents answer  {lowest_answerable:.3f} – {max(answerable):.3f}"
        )
        self.stdout.write(
            f"  Questions they do not           {min(out_of_scope):.3f} – {highest_noise:.3f}"
        )
        self.stdout.write(f"  RETRIEVAL_MIN_SIMILARITY        {floor}")

        if highest_noise >= lowest_answerable:
            self.stdout.write(
                self.style.ERROR(
                    "  The two ranges overlap. No single threshold separates them — improve "
                    "chunking or the embedding model rather than tuning this number."
                )
            )
        elif floor <= highest_noise or floor >= lowest_answerable:
            suggested = (highest_noise + lowest_answerable) / 2
            self.stdout.write(
                self.style.WARNING(f"  The floor sits outside the gap. Try {suggested:.2f}.")
            )
        else:
            self.stdout.write(self.style.SUCCESS("  The floor sits inside the gap. Good."))
