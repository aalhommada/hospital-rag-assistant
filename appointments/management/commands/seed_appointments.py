"""
`manage.py seed_appointments`

Fills the appointment book with departments, clinicians, and free slots so the
booking half of the assistant has something real to work against. Safe to run
repeatedly: existing rows are matched, not duplicated, and slots that already
exist are left alone — including any that have been booked.
"""

from datetime import datetime, time, timedelta

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from django.utils.text import slugify

from appointments.models import Appointment, Clinician, Department, Slot

DEPARTMENTS = [
    ("Cardiology", "Green Zone, level 2", "020 7946 0210"),
    ("Respiratory medicine", "Green Zone, level 2", "020 7946 0215"),
    ("Dermatology", "Green Zone, ground floor", "020 7946 0220"),
    ("Physiotherapy", "Blue Zone, ground floor", "020 7946 0230"),
    ("Endoscopy", "Green Zone, level 2", "020 7946 0188"),
]

CLINICIANS = [
    ("Dr Amara Okafor", "Cardiology", "General cardiology"),
    ("Dr Peter Lindqvist", "Cardiology", "Heart rhythm"),
    ("Dr Sian Roberts", "Respiratory medicine", "Asthma and COPD"),
    ("Dr Hana Yusuf", "Dermatology", "General dermatology"),
    ("Nadia Kaur", "Physiotherapy", "Musculoskeletal physiotherapy"),
    ("Dr Tomas Bergman", "Endoscopy", "Gastroenterology"),
]

# Clinic times, weekdays only.
CLINIC_HOURS = [9, 10, 11, 14, 15, 16]
DAYS_AHEAD = 21


class Command(BaseCommand):
    help = "Create sample departments, clinicians, and bookable appointment slots."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete all appointments and slots first, then recreate them",
        )

    @transaction.atomic
    def handle(self, *args, **options) -> None:
        if options["reset"]:
            Appointment.objects.all().delete()
            Slot.objects.all().delete()
            self.stdout.write(self.style.WARNING("Cleared existing slots and appointments"))

        departments = {}
        for name, zone, phone in DEPARTMENTS:
            department, _ = Department.objects.update_or_create(
                slug=slugify(name), defaults={"name": name, "zone": zone, "phone": phone}
            )
            departments[name] = department

        clinicians = []
        for full_name, department_name, specialty in CLINICIANS:
            clinician, _ = Clinician.objects.update_or_create(
                full_name=full_name,
                department=departments[department_name],
                defaults={"specialty": specialty},
            )
            clinicians.append(clinician)

        created = self._create_slots(clinicians)

        self.stdout.write(
            self.style.SUCCESS(
                f"{Department.objects.count()} departments, {Clinician.objects.count()} clinicians, "
                f"{Slot.objects.filter(is_booked=False).count()} free slots "
                f"({created} created just now)."
            )
        )

    def _create_slots(self, clinicians: list[Clinician]) -> int:
        """One clinic day per clinician every third day, starting tomorrow."""
        tz = timezone.get_current_timezone()
        today = timezone.localdate()
        wanted: list[Slot] = []

        for index, clinician in enumerate(clinicians):
            for offset in range(1, DAYS_AHEAD + 1):
                day = today + timedelta(days=offset)
                if day.weekday() >= 5:  # no weekend clinics
                    continue
                if (offset + index) % 3 != 0:  # each clinician runs every third day
                    continue
                for hour in CLINIC_HOURS:
                    starts_at = timezone.make_aware(datetime.combine(day, time(hour=hour)), tz)
                    wanted.append(
                        Slot(clinician=clinician, starts_at=starts_at, duration_minutes=20)
                    )

        # ignore_conflicts leaves already-created (and possibly booked) slots
        # untouched, which is what makes this command safe to re-run.
        before = Slot.objects.count()
        Slot.objects.bulk_create(wanted, ignore_conflicts=True)
        return Slot.objects.count() - before
