"""
`manage.py ingest_documents sample_data`

The entry point for getting hospital documents into the knowledge base. Point
it at a directory or at individual files.
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from knowledge.ingest.loaders import SUPPORTED_SUFFIXES
from knowledge.ingest.pipeline import IngestResult, ingest_directory, ingest_file
from knowledge.models import Chunk, Document

STATUS_ORDER = ["created", "updated", "unchanged", "failed"]


class Command(BaseCommand):
    help = "Load, chunk, embed, and index documents into the knowledge base."

    def add_arguments(self, parser) -> None:
        parser.add_argument("paths", nargs="+", help="Files or directories to ingest")
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete every existing document first, then ingest from scratch",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-ingest even when the file's checksum has not changed",
        )

    def handle(self, *args, **options) -> None:
        targets = [Path(p) for p in options["paths"]]
        for target in targets:
            if not target.exists():
                raise CommandError(f"{target} does not exist")

        if options["reset"]:
            deleted = Document.objects.count()
            Document.objects.all().delete()
            self.stdout.write(self.style.WARNING(f"Removed {deleted} existing document(s)"))

        results: list[IngestResult] = []
        for target in targets:
            if target.is_dir():
                results.extend(ingest_directory(target, force=options["force"]))
            elif target.suffix.lower() in SUPPORTED_SUFFIXES:
                results.append(ingest_file(target, force=options["force"]))
            else:
                self.stdout.write(self.style.WARNING(f"skipping unsupported file {target}"))

        self._report(results)

    def _report(self, results: list[IngestResult]) -> None:
        if not results:
            self.stdout.write(self.style.WARNING("Nothing to ingest."))
            return

        width = max(len(r.title) for r in results)
        for result in sorted(results, key=lambda r: (STATUS_ORDER.index(r.status), r.title)):
            style = {
                "created": self.style.SUCCESS,
                "updated": self.style.SUCCESS,
                "unchanged": self.style.HTTP_INFO,
                "failed": self.style.ERROR,
            }[result.status]
            line = f"{result.title:<{width}}  {result.status:<9}  {result.chunk_count:>3} chunks"
            if result.detail:
                line += f"  ({result.detail})"
            self.stdout.write(style(line))

        failed = sum(1 for r in results if r.status == "failed")
        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Knowledge base now holds {Document.objects.count()} documents "
                f"and {Chunk.objects.count()} chunks."
            )
        )
        if failed:
            self.stdout.write(self.style.ERROR(f"{failed} file(s) failed — see the list above."))
