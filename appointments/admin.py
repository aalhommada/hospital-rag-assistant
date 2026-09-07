from django.contrib import admin

from .models import Appointment, Clinician, Department, Slot


@admin.register(Department)
class DepartmentAdmin(admin.ModelAdmin):
    list_display = ("name", "zone", "phone")
    prepopulated_fields = {"slug": ("name",)}


@admin.register(Clinician)
class ClinicianAdmin(admin.ModelAdmin):
    list_display = ("full_name", "department", "specialty")
    list_filter = ("department",)
    search_fields = ("full_name",)


@admin.register(Slot)
class SlotAdmin(admin.ModelAdmin):
    list_display = ("starts_at", "clinician", "duration_minutes", "is_booked")
    list_filter = ("is_booked", "clinician__department", "clinician")
    date_hierarchy = "starts_at"


@admin.register(Appointment)
class AppointmentAdmin(admin.ModelAdmin):
    list_display = ("reference", "patient_name", "slot", "status", "created_at")
    list_filter = ("status", "slot__clinician__department")
    search_fields = ("reference", "patient_name", "contact_number")
    readonly_fields = ("reference", "created_at")
