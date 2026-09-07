# Every target assumes the virtualenv at .venv. Run `make install` first.
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help install db-up db-down migrate seed ingest run test lint format evaluate reset

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

db-up: ## Start Postgres with pgvector
	docker compose up -d
	@echo "waiting for postgres..."
	@until docker compose exec -T db pg_isready -q; do sleep 1; done
	@echo "database ready on port $${POSTGRES_PORT:-5433}"

db-down: ## Stop Postgres (data is kept in a named volume)
	docker compose down

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

evaluate: ## Score retrieval and routing against the labelled question set
	$(PY) manage.py evaluate

reset: ## Drop every document and chunk, then ingest again from scratch
	$(PY) manage.py ingest_documents sample_data --reset
