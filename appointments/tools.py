"""
The two things the assistant is allowed to *do*, as opposed to answer.

A tool is a function with a JSON schema attached. The schema goes to Claude in
the request; Claude replies with the name of a tool and arguments; this module
runs the real Python function and hands the result back. Claude never touches
the database — it asks, and this code decides.

Both tools are declared `strict`, so the API guarantees the arguments validate
against the schema before they reach us. That removes a whole class of defensive
parsing, but it does not remove the checks below: a schema can promise that
`slot_id` is an integer, never that the slot is still free.

Design rules that matter more than the code:

* **Reads are wide, writes are narrow.** Searching for free slots is harmless
  and takes fuzzy input. Booking one takes an exact slot id that the model can
  only have obtained from a previous search.
* **The write is guarded in the database, not in the prompt.** `select_for_update`
  means two people asking for the same slot at the same moment cannot both get
  it, however the conversation went.
* **A failed tool returns an explanation, not an exception.** "That slot has
  just been taken, here are three others" is something the model can recover
  from. A traceback is not.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Any

from django.db import transaction
from django.utils import timezone

from .models import Appointment, Department, Slot

logger = logging.getLogger(__name__)

MAX_SLOTS_RETURNED = 8
BOOKING_HORIZON_DAYS = 60


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "find_available_slots",
        "description": (
            "Search the appointment book for free slots. Use this whenever someone asks about "
            "booking, changing, or the availability of an appointment. Always call this before "
            "book_appointment — it returns the slot_id values that booking requires. "
            "Returns at most 8 slots, soonest first."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "Department name, e.g. 'Cardiology'. Partial names match.",
                },
                "clinician_name": {
                    "type": ["string", "null"],
                    "description": "Optional. Part of a clinician's name, if the patient asked for someone specific.",
                },
                "earliest_date": {
                    "type": ["string", "null"],
                    "description": "Optional ISO date (YYYY-MM-DD). Do not offer anything before this day.",
                },
                "latest_date": {
                    "type": ["string", "null"],
                    "description": "Optional ISO date (YYYY-MM-DD). Do not offer anything after this day.",
                },
            },
            "required": ["department", "clinician_name", "earliest_date", "latest_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "book_appointment",
        "description": (
            "Take a specific free slot. Only call this once the patient has confirmed the exact "
            "slot and has given their full name and a contact telephone number. Never guess "
            "either of them, and never invent a slot_id — use one returned by find_available_slots."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "slot_id": {
                    "type": "integer",
                    "description": "The slot_id from a find_available_slots result.",
                },
                "patient_name": {
                    "type": "string",
                    "description": "The patient's full name, exactly as they gave it.",
                },
                "contact_number": {
                    "type": "string",
                    "description": "A telephone number the hospital can reach them on.",
                },
                "reason": {
                    "type": "string",
                    "description": "A short reason for the appointment, in the patient's own words.",
                },
            },
            "required": ["slot_id", "patient_name", "contact_number", "reason"],
            "additionalProperties": False,
        },
    },
]


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run a tool by name and return text for the model to read."""
    if name == "find_available_slots":
        return find_available_slots(**arguments)
    if name == "book_appointment":
        return book_appointment(**arguments)
    return f"Unknown tool {name!r}. Available tools: find_available_slots, book_appointment."


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def find_available_slots(
    department: str,
    clinician_name: str | None = None,
    earliest_date: str | None = None,
    latest_date: str | None = None,
) -> str:
    matched = Department.objects.filter(name__icontains=department.strip())
    if not matched.exists():
        known = ", ".join(Department.objects.values_list("name", flat=True))
        return f"No department matches {department!r}. Departments that take bookings: {known}."

    slots = Slot.objects.select_related("clinician__department").filter(
        clinician__department__in=matched,
        is_booked=False,
        starts_at__gt=timezone.now(),
        starts_at__lt=timezone.now() + timedelta(days=BOOKING_HORIZON_DAYS),
    )

    if clinician_name:
        slots = slots.filter(clinician__full_name__icontains=clinician_name.strip())

    window = _date_window(earliest_date, latest_date)
    if window.error:
        return window.error
    if window.start:
        slots = slots.filter(starts_at__gte=window.start)
    if window.end:
        slots = slots.filter(starts_at__lt=window.end)

    found = list(slots.order_by("starts_at")[:MAX_SLOTS_RETURNED])
    if not found:
        return (
            f"No free slots in {department} matching those constraints. "
            "Suggest widening the date range, or offer to call the department directly."
        )

    lines = [f"{len(found)} free slot(s):"]
    for slot in found:
        local = timezone.localtime(slot.starts_at)
        lines.append(
            f"- slot_id={slot.pk} | {local:%A %d %B %Y at %H:%M} | "
            f"{slot.clinician.full_name} | {slot.clinician.department.name} | "
            f"{slot.duration_minutes} minutes"
        )
    return "\n".join(lines)


class _Window:
    def __init__(self, start: datetime | None, end: datetime | None, error: str = "") -> None:
        self.start, self.end, self.error = start, end, error


def _date_window(earliest: str | None, latest: str | None) -> _Window:
    """Turn optional ISO date strings into an aware datetime range."""
    try:
        start = _start_of_day(date.fromisoformat(earliest)) if earliest else None
        # `latest` is inclusive for a human, so the exclusive bound is the next day.
        end = _start_of_day(date.fromisoformat(latest) + timedelta(days=1)) if latest else None
    except ValueError:
        return _Window(None, None, "Dates must be in YYYY-MM-DD form, for example 2026-09-14.")
    if start and end and end <= start:
        return _Window(None, None, "The latest date must not be before the earliest date.")
    return _Window(start, end)


def _start_of_day(day: date) -> datetime:
    return timezone.make_aware(datetime.combine(day, time.min), timezone.get_current_timezone())


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def book_appointment(slot_id: int, patient_name: str, contact_number: str, reason: str = "") -> str:
    patient_name = (patient_name or "").strip()
    contact_number = (contact_number or "").strip()

    # The schema guarantees these fields are present; it cannot guarantee they
    # are not empty strings, and an appointment nobody can be contacted about
    # is worse than no appointment.
    if not patient_name:
        return "Cannot book without the patient's full name. Ask for it, then call this tool again."
    if not contact_number:
        return (
            "Cannot book without a contact telephone number. Ask for it, then call this tool again."
        )

    try:
        with transaction.atomic():
            # Lock the row. Two conversations racing for the last slot now
            # queue here, and the second one gets the honest error below.
            slot = (
                Slot.objects.select_for_update()
                .select_related("clinician__department")
                .get(pk=slot_id)
            )

            if slot.is_booked:
                return (
                    f"Slot {slot_id} has just been taken by someone else. "
                    "Call find_available_slots again and offer the patient a different time."
                )
            if slot.starts_at <= timezone.now():
                return f"Slot {slot_id} is in the past and cannot be booked."

            slot.is_booked = True
            slot.save(update_fields=["is_booked"])

            appointment = Appointment.objects.create(
                slot=slot,
                patient_name=patient_name,
                contact_number=contact_number,
                reason=(reason or "").strip()[:255],
                reference=Appointment.generate_reference(),
            )
    except Slot.DoesNotExist:
        return (
            f"There is no slot with id {slot_id}. Only use slot_id values returned by "
            "find_available_slots — never invent one."
        )

    local = timezone.localtime(slot.starts_at)
    logger.info("booked appointment %s for slot %s", appointment.reference, slot.pk)
    return (
        f"Booked. Reference {appointment.reference} for {appointment.patient_name} "
        f"with {slot.clinician.full_name} ({slot.clinician.department.name}) on "
        f"{local:%A %d %B %Y at %H:%M}. Tell the patient the reference and remind them to "
        f"call {slot.clinician.department.phone or 'the department'} if they need to change it."
    )
