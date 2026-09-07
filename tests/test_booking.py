"""The booking tools. Every path the model can drive them down."""

from datetime import timedelta

import pytest
from django.utils import timezone

from appointments.models import Appointment, Slot
from appointments.tools import execute_tool, find_available_slots

pytestmark = pytest.mark.django_db


def test_free_slots_are_listed_with_ids(booking_book):
    result = find_available_slots("Cardiology", None, None, None)
    assert "slot_id=" in result
    assert "Dr Amara Okafor" in result


def test_partial_department_names_match(booking_book):
    assert "slot_id=" in find_available_slots("cardio", None, None, None)


def test_unknown_department_lists_the_real_ones(booking_book):
    result = find_available_slots("Astrology", None, None, None)
    assert "No department matches" in result
    assert "Cardiology" in result


def test_booked_slots_disappear_from_the_list(booking_book):
    slot = booking_book["slots"][0]
    execute_tool(
        "book_appointment",
        {
            "slot_id": slot.pk,
            "patient_name": "Alex Fisher",
            "contact_number": "07700 900123",
            "reason": "review",
        },
    )
    assert f"slot_id={slot.pk} " not in find_available_slots("Cardiology", None, None, None)


def test_booking_creates_an_appointment_with_a_reference(booking_book):
    slot = booking_book["slots"][0]
    result = execute_tool(
        "book_appointment",
        {
            "slot_id": slot.pk,
            "patient_name": "Alex Fisher",
            "contact_number": "07700 900123",
            "reason": "review",
        },
    )
    appointment = Appointment.objects.get()
    assert appointment.reference in result
    assert appointment.patient_name == "Alex Fisher"
    assert Slot.objects.get(pk=slot.pk).is_booked


def test_the_same_slot_cannot_be_booked_twice(booking_book):
    slot = booking_book["slots"][0]
    payload = {
        "slot_id": slot.pk,
        "patient_name": "A B",
        "contact_number": "07700 900123",
        "reason": "x",
    }
    execute_tool("book_appointment", payload)
    second = execute_tool("book_appointment", {**payload, "patient_name": "C D"})
    assert "just been taken" in second
    assert Appointment.objects.count() == 1


def test_an_invented_slot_id_is_refused(booking_book):
    result = execute_tool(
        "book_appointment",
        {"slot_id": 999999, "patient_name": "A B", "contact_number": "07700 900123", "reason": "x"},
    )
    assert "no slot with id" in result
    assert Appointment.objects.count() == 0


def test_booking_without_a_name_is_refused(booking_book):
    result = execute_tool(
        "book_appointment",
        {
            "slot_id": booking_book["slots"][0].pk,
            "patient_name": "  ",
            "contact_number": "07700 900123",
            "reason": "x",
        },
    )
    assert "full name" in result
    assert Appointment.objects.count() == 0


def test_booking_without_a_number_is_refused(booking_book):
    result = execute_tool(
        "book_appointment",
        {
            "slot_id": booking_book["slots"][0].pk,
            "patient_name": "A B",
            "contact_number": "",
            "reason": "x",
        },
    )
    assert "telephone number" in result
    assert Appointment.objects.count() == 0


def test_past_slots_are_never_offered_or_booked(booking_book):
    stale = Slot.objects.create(
        clinician=booking_book["clinician"], starts_at=timezone.now() - timedelta(days=1)
    )
    assert f"slot_id={stale.pk} " not in find_available_slots("Cardiology", None, None, None)
    result = execute_tool(
        "book_appointment",
        {
            "slot_id": stale.pk,
            "patient_name": "A B",
            "contact_number": "07700 900123",
            "reason": "x",
        },
    )
    assert "in the past" in result


def test_a_malformed_date_is_explained_not_raised(booking_book):
    assert "YYYY-MM-DD" in find_available_slots("Cardiology", None, "next tuesday", None)


def test_an_inverted_date_range_is_rejected(booking_book):
    result = find_available_slots("Cardiology", None, "2026-10-10", "2026-10-01")
    assert "must not be before" in result


def test_an_unknown_tool_name_is_reported(booking_book):
    assert "Unknown tool" in execute_tool("delete_everything", {})
