# Decisions log
(Agent: append `YYYY-MM-DD | decision | reason` here whenever you resolve an ambiguity.)

2026-11-20 | Track `uv.lock` and use `uv sync --all-extras --frozen` in setup/CI | Reproducible installs are required for a clean-clone build; the lock file was previously ignored.
2026-11-20 | Keep strict mypy for completed modules and temporarily quarantine modules belonging to later milestones | The execution plan is strictly ordered; quarantined modules are re-enabled as their milestones are completed instead of weakening the global type-checking policy.
2026-11-20 | Add `scripts/check.py` as a Windows-compatible equivalent of `make lint test` | The development shell has no `make`; CI and Unix developers keep using the Makefile, while local verification remains reproducible.
2026-11-20 | Use CT.gov API v2 `filter.advanced=AREA[StudyType]INTERVENTIONAL AND AREA[StartDate]RANGE[...]` and comma-separated `filter.overallStatus` | The v2 API rejects the legacy `filter.studyType`/`filter.startDate` parameters and repeated status parameters; official response messages confirmed the supported syntax.
2026-11-20 | Use PubMed `esearch(usehistory=y)` plus batched `efetch` instead of one-shot ID lists | History keys avoid very long URLs and keep large abstract ingestion within NCBI request-size limits.
2026-11-20 | Partition PubMed ingestion into daily publication-date windows | NCBI allows only the first 9,999 esearch matches and rejects efetch `retstart` values beyond that window; daily partitions keep every query within the official limit while still reaching the volume target.
2026-11-20 | Link openFDA labels by normalized intervention name and write deduplicated quarantine JSONL files | This keeps the ingest idempotent while making rejected records auditable without polluting the database.
2026-11-20 | Filter openFDA search terms to simple drug-like names and skip non-404 HTTP errors per term | CT.gov intervention names include regimens and quoted labels that produce invalid openFDA queries; a bad term must not abort the whole ingest.
2026-11-20 | Rank openFDA search terms by intervention frequency and fall back to a bounded `openfda.generic_name:*` catalog scan | Alphabetically sorted trial identifiers matched almost no approved labels; frequency ranking plus a catalog scan keeps the API fallback deterministic while reaching the linked-label target.
2026-11-20 | Use a deterministic offline tokenizer fallback when the configured Hugging Face tokenizer is not cached | Chunking must be reproducible and offline-capable; the fallback preserves stable token bounds without silently hitting the network.
2026-11-20 | Persist Python enum values with `values_callable` | PostgreSQL enum labels are lowercase (`primary`, `trial`, `train`) and ORM inserts must match the Alembic-created types.
2026-11-20 | Generate the demo dataset with seed `42` | All randomness is seeded by project convention and sample outputs must be byte-stable enough for CI assertions.
