# Every target assumes the virtualenv at .venv. Run `make install` first.
PY := .venv/bin/python
PIP := .venv/bin/pip

# Django 6.1 needs Python 3.12 or newer. Pick the newest interpreter on this
# machine that qualifies, rather than whatever `python3` happens to point at —
# an older `python3` does not fail, it silently resolves Django to 5.2.
PYTHON ?= $(shell for p in python3.14 python3.13 python3.12 python3; do \
	command -v $$p >/dev/null 2>&1 && \
	$$p -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null && \
	echo $$p && break; done)

# Compose v2 is a docker subcommand; older installs ship the standalone binary.
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

.PHONY: help install db-up db-down migrate seed ingest run test lint format evaluate evaluate-routing outdated reset

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install dependencies
ifeq ($(PYTHON),)
	$(error No Python 3.12+ found. Install one, e.g. `uv python install 3.14`, or `sudo apt install python3.13`, then re-run `make install`)
endif
	@echo "using $(PYTHON) ($$($(PYTHON) --version))"
	$(PYTHON) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

db-up: ## Start Postgres with pgvector
	$(COMPOSE) up -d
	@echo "waiting for postgres..."
	@until $(COMPOSE) exec -T db pg_isready -q >/dev/null 2>&1; do sleep 1; done
	@echo "database ready on port $${POSTGRES_PORT:-5433}"

db-down: ## Stop Postgres (data is kept in a named volume)
	$(COMPOSE) down

migrate: ## Apply database migrations
	$(PY) manage.py migrate

seed: migrate ingest ## Migrate, ingest the sample documents, and create appointment slots
	$(PY) manage.py seed_appointments

ingest: ## Load, chunk, embed, and index everything in sample_data/
	$(PY) manage.py ingest_documents sample_data

run: ## Start the development server on http://127.0.0.1:8000
	$(PY) manage.py runserver

test: ## Run the test suite
	.venv/bin/pytest -q

lint: ## Check formatting and lint rules
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format: ## Apply formatting
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

outdated: ## List dependencies with a newer release available
	$(PIP) list --outdated

evaluate: ## Score retrieval against the labelled question set (no API key needed)
	$(PY) manage.py evaluate

evaluate-routing: ## Also score routing accuracy (needs ANTHROPIC_API_KEY, one call per question)
	$(PY) manage.py evaluate --with-router

reset: ## Drop every document and chunk, then ingest again from scratch
	$(PY) manage.py ingest_documents sample_data --reset
