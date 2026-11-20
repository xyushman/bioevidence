# BioEvidence Makefile

.PHONY: help setup lint test test-cov check migrate ingest quality chunks embed train-embed train-rerank train-lora eval eval-ci forecast api web up down clean

# Default target
help:
	@echo "BioEvidence - Biomedical Evidence Copilot"
	@echo ""
	@echo "Available targets:"
	@echo "  setup         - Install dependencies and pre-commit hooks"
	@echo "  lint          - Run ruff and mypy"
	@echo "  test          - Run pytest"
	@echo "  test-cov      - Run pytest with coverage"
	@echo "  check         - Run lint and test together"
	@echo "  migrate       - Run Alembic migrations"
	@echo "  ingest        - Run all ingestors (ctgov, pubmed, openfda)"
	@echo "  quality       - Run data quality checks"
	@echo "  chunks        - Build chunks and splits"
	@echo "  embed         - Build embeddings"
	@echo "  train-embed   - Fine-tune embeddings"
	@echo "  train-rerank  - Fine-tune reranker"
	@echo "  train-lora    - Train LoRA extractor"
	@echo "  eval          - Run full evaluation"
	@echo "  eval-ci       - Run CI evaluation gate"
	@echo "  forecast      - Run forecasting pipeline"
	@echo "  api           - Start FastAPI server"
	@echo "  web           - Start React dev server"
	@echo "  up            - Start Docker Compose services"
	@echo "  down          - Stop Docker Compose services"
	@echo "  clean         - Clean build artifacts"

# Install dependencies and setup
setup:
	uv sync --all-extras --frozen
	uv run pre-commit install
	mkdir -p data/raw/ctgov data/raw/pubmed data/raw/openfda data/processed data/splits data/goldens results/retrieval results/extraction results/agent results/forecast reports mlruns models

# Linting
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

# Testing
test:
	uv run pytest

check: lint test

test-cov:
	uv run pytest --cov=src/bioevidence --cov-report=term-missing --cov-report=html

# Database
migrate:
	uv run alembic upgrade head

# Ingestion
ingest:
	uv run python -m bioevidence ingest ctgov
	uv run python -m bioevidence ingest pubmed
	uv run python -m bioevidence ingest openfda

ingest-ctgov:
	uv run python -m bioevidence ingest ctgov

ingest-pubmed:
	uv run python -m bioevidence ingest pubmed

ingest-openfda:
	uv run python -m bioevidence ingest openfda

# Quality
quality:
	uv run python -m bioevidence quality run

# Chunking and splits
chunks:
	uv run python -m bioevidence chunk build
	uv run python -m bioevidence splits make

# Embeddings
embed:
	uv run python -m bioevidence embed build --model $(EMBEDDING_MODEL)

# Training
train-embed:
	uv run python -m bioevidence train embed

train-rerank:
	uv run python -m bioevidence train rerank

train-lora:
	uv run python -m bioevidence train lora

# Evaluation
eval:
	uv run python -m bioevidence eval run

eval-ci:
	uv run python -m bioevidence eval ci

# Forecasting
forecast:
	uv run python -m bioevidence forecast run

# API and Web
api:
	uv run uvicorn bioevidence.api.main:app --host $(API_HOST) --port $(API_PORT) --reload

web:
	cd web && npm run dev

# Docker
up:
	docker compose up -d

down:
	docker compose down

# Clean
clean:
	rm -rf dist build *.egg-info .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true