# AGENTS.md: instructions for the coding agent

You are building the BioEvidence project end to end. Follow these rules strictly.

## Workflow
1. Read `README.md`, then every file in `docs/` before writing code.
2. Execute `docs/09-TASKS.md` strictly in order. Tick a task only when its acceptance criteria pass.
3. After each milestone: run `make lint test`, commit with a conventional message (`feat:`, `fix:`, `docs:`, `exp:`).
4. If a spec is ambiguous, choose the simplest option, record it in `docs/DECISIONS.md` (date, decision, reason), continue.
5. Never fabricate metrics. Every number in docs/README must come from a file in `results/`.

## Conventions
- Python 3.11, `uv` for deps, `pyproject.toml`. Package: `src/bioevidence/`.
- Type hints everywhere, `pydantic` v2 for schemas/config, `ruff` + `pytest`.
- Config via `configs/*.yaml` + `.env` (never commit secrets; keep `.env.example` updated).
- All randomness seeded (`seed=42` default). All experiments log to MLflow (`mlruns/` gitignored) AND write a JSON
  to `results/<experiment>/<timestamp>.json` (committed, small).
- CLI via Typer: `python -m bioevidence <command>`; Makefile wraps commands.
- Idempotent pipelines: re-running ingestion must not duplicate rows (upsert on natural keys).
- Logging with `structlog` or stdlib `logging`; no `print` in library code.
- Keep modules small; one responsibility each. Tests next to behavior, not implementation.

## Hard rules
- LLM provider is pluggable (`LLM_PROVIDER=gemini|anthropic|openai`) behind `bioevidence.llm.client.LLMClient`. Cache every
  LLM call on disk (hash of model+prompt+params) so reruns are free and deterministic.
- Respect API rate limits (NCBI: 3 req/s without key, 10 with; openFDA: use bulk downloads when possible).
- The SQL tool in the agent must use a read-only DB role, statement timeout, allowlisted tables, AST validation. See `06-AGENT.md`.
- Train/dev/test splits are made ONCE by document/trial id with a fixed seed and stored in `data/splits/`. No leakage:
  test-set chunks/queries must never appear in any training data. Add a test that asserts this.
- Heavy training runs in Colab/Kaggle: provide `notebooks/` or `scripts/train_*.py` that run on a free T4 (16 GB). Keep
  configs small (bge-small, 4-bit 3B, seq len <= 1024).
- Medical disclaimer in UI and README: research demo, not medical advice.

## Definition of done (per feature)
Code + unit tests + config + short doc section + metrics file (if model-related) + `make lint test` green.

## Do NOT
- Do not use heavy orchestration frameworks (LangChain agents, etc.). Use plain Python, the provider SDK, `sentence-transformers`,
  `peft`, `trl`. LangChain-style abstractions hide the ML work that this project must show.
- Do not scrape sites; use official APIs/bulk downloads only.
- Do not skip the baseline. Every improvement is measured against it.
