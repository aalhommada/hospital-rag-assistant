"""
Settings for the hospital assistant.

Everything that changes between machines is read from the environment, so the
same code runs locally and in production. Copy `.env.example` to `.env` and
edit that file — `manage.py` loads it before Django starts.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


# ---------------------------------------------------------------------------
# Core Django
# ---------------------------------------------------------------------------

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "insecure-development-key-change-me")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if h.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.postgres",
    "knowledge",
    "appointments",
    "assistant",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("POSTGRES_DB", "hospital_rag"),
        "USER": os.environ.get("POSTGRES_USER", "hospital"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "hospital"),
        "HOST": os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        "PORT": os.environ.get("POSTGRES_PORT", "5433"),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.environ.get("DJANGO_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple": {"format": "{levelname} {name}: {message}", "style": "{"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}

# ---------------------------------------------------------------------------
# Embeddings
#
# EMBEDDING_DIMENSIONS must match the width of the vectors your provider
# returns, because it is baked into the database column by a migration.
# Changing provider therefore means changing this number and creating a new
# migration — see README, "Changing the embedding model".
# ---------------------------------------------------------------------------

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSIONS = env_int("EMBEDDING_DIMENSIONS", 384)
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_TARGET_WORDS = env_int("CHUNK_TARGET_WORDS", 180)
CHUNK_MAX_WORDS = env_int("CHUNK_MAX_WORDS", 260)
CHUNK_OVERLAP_WORDS = env_int("CHUNK_OVERLAP_WORDS", 30)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RETRIEVAL_CANDIDATES = env_int("RETRIEVAL_CANDIDATES", 30)
RETRIEVAL_TOP_K = env_int("RETRIEVAL_TOP_K", 6)
# Cosine similarity a chunk must reach to be allowed into the prompt, unless
# the keyword arm matched it exactly (see knowledge/retrieval.py). Measured on
# this corpus: answerable questions score 0.68-0.91, unanswerable ones
# 0.41-0.56, so 0.65 sits in the gap. Re-measure after changing the embedding
# model or the documents — `manage.py evaluate` reports both sides.
RETRIEVAL_MIN_SIMILARITY = env_float("RETRIEVAL_MIN_SIMILARITY", 0.65)
# Constant in the Reciprocal Rank Fusion formula 1 / (RRF_K + rank).
# 60 is the value from the original paper and works well without tuning.
RRF_K = env_int("RRF_K", 60)

# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_ANSWER_MODEL = os.environ.get("CLAUDE_ANSWER_MODEL", "claude-opus-5")
CLAUDE_ROUTER_MODEL = os.environ.get("CLAUDE_ROUTER_MODEL", "claude-opus-5")
CLAUDE_ENABLE_FALLBACKS = env_bool("CLAUDE_ENABLE_FALLBACKS", True)
# Generous enough that adaptive thinking plus a cited answer never truncates,
# while the prompt keeps the visible answer short.
CLAUDE_ANSWER_MAX_TOKENS = env_int("CLAUDE_ANSWER_MAX_TOKENS", 8000)
CLAUDE_ROUTER_MAX_TOKENS = env_int("CLAUDE_ROUTER_MAX_TOKENS", 2000)

# Shown to patients whenever the assistant refuses, so the refusal is a
# handover rather than a dead end.
HOSPITAL_NAME = os.environ.get("HOSPITAL_NAME", "Riverside General Hospital")
HOSPITAL_SWITCHBOARD = os.environ.get("HOSPITAL_SWITCHBOARD", "+44 20 7946 0000")
HOSPITAL_EMERGENCY_NUMBER = os.environ.get("HOSPITAL_EMERGENCY_NUMBER", "999")
