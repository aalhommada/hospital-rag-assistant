"""
The booking side of the assistant.

Nothing here is AI. It is an ordinary appointment book: departments, clinicians,
the slots they offer, and the bookings made against them. It exists to make one
boundary visible.

Answering "what should I bring to my MRI?" is retrieval — search text, ground an
answer in it, cite it. Answering "book me a chest clinic slot on Thursday" is
not retrieval at all. Nothing is searched. A row is written, and afterwards the
world is different: a slot that was free is taken.

Those two jobs need different machinery, different failure handling, and
different guarantees. Confusing them is the most common mistake in systems like
this one — an "AI booking agent" that is really a search engine with optimism.
"""

from __future__ import annotations

import secrets
from datetime import timedelta

from django.db import models
from django.utils import timezone


class Department(models.Model):
    name = models.CharField(max_length=120, unique=True)
    slug = models.SlugField(max_length=120, unique=True)
    zone = models.CharField(max_length=60, blank=True, help_text="e.g. Green Zone, level 2")
    phone = models.CharField(max_length=40, blank=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Clinician(models.Model):
    full_name = models.CharField(max_length=120)
    department = models.ForeignKey(Department, on_delete=models.CASCADE, related_name="clinicians")
    specialty = models.CharField(max_length=120, blank=True)

    class Meta:
        ordering = ["full_name"]
        constraints = [
            models.UniqueConstraint(
                fields=["full_name", "department"], name="unique_clinician_per_department"
            )
        ]

    def __str__(self) -> str:
        return f"{self.full_name} ({self.department.name})"


class Slot(models.Model):
    """One bookable appointment time."""

    clinician = models.ForeignKey(Clinician, on_delete=models.CASCADE, related_name="slots")
    starts_at = models.DateTimeField()
    duration_minutes = models.PositiveIntegerField(default=20)
    is_booked = models.BooleanField(default=False)

    class Meta:
        ordering = ["starts_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["clinician", "starts_at"], name="unique_slot_per_clinician"
            )
        ]
        indexes = [models.Index(fields=["starts_at", "is_booked"], name="slot_availability_idx")]

    def __str__(self) -> str:
        return f"{self.clinician.full_name} at {self.starts_at:%d %b %Y %H:%M}"

    @property
    def ends_at(self):
        return self.starts_at + timedelta(minutes=self.duration_minutes)

    @property
    def is_bookable(self) -> bool:
        return not self.is_booked and self.starts_at > timezone.now()


class Appointment(models.Model):
    """A slot that somebody has taken."""

    class Status(models.TextChoices):
        BOOKED = "booked", "Booked"
        CANCELLED = "cancelled", "Cancelled"

    slot = models.OneToOneField(Slot, on_delete=models.PROTECT, related_name="appointment")
    patient_name = models.CharField(max_length=120)
    contact_number = models.CharField(max_length=40)
    reason = models.CharField(max_length=255, blank=True)
    reference = models.CharField(max_length=12, unique=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.BOOKED)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.reference} — {self.patient_name}"

    @staticmethod
    def generate_reference() -> str:
        """A short human-readable booking reference, e.g. RG-7K2M9Q."""
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no I/O/0/1 — misread over the phone
        return "RG-" + "".join(secrets.choice(alphabet) for _ in range(6))
