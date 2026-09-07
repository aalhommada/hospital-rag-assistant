from django.urls import path

from . import views

app_name = "assistant"

urlpatterns = [
    path("", views.chat, name="chat"),
    path("ask/", views.ask, name="ask"),
    path("stream/<int:message_id>/", views.stream, name="stream"),
]
