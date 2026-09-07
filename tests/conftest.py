"""Shared fixtures. `.env` is loaded so tests hit the same database as the app."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@pytest.fixture
def sample_documents(db):
    """Ingest two short documents into the test database."""
    from knowledge.ingest.pipeline import ingest_file

    ingest_file(BASE_DIR / "sample_data" / "visiting-hours-and-ward-rules.md")
    ingest_file(BASE_DIR / "sample_data" / "parking-transport-and-accessibility.md")


@pytest.fixture
def booking_book(db):
    """A department with one clinician and two free slots."""
    from datetime import timedelta

    from django.utils import timezone

    from appointments.models import Clinician, Department, Slot

    department = Department.objects.create(
        name="Cardiology", slug="cardiology", phone="020 7946 0210"
    )
    clinician = Clinician.objects.create(full_name="Dr Amara Okafor", department=department)
    start = timezone.now() + timedelta(days=2)
    slots = [
        Slot.objects.create(clinician=clinician, starts_at=start),
        Slot.objects.create(clinician=clinician, starts_at=start + timedelta(hours=1)),
    ]
    return {"department": department, "clinician": clinician, "slots": slots}
