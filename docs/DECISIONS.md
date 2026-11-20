# Decisions log
(Agent: append `YYYY-MM-DD | decision | reason` here whenever you resolve an ambiguity.)

2026-11-20 | Track `uv.lock` and use `uv sync --all-extras --frozen` in setup/CI | Reproducible installs are required for a clean-clone build; the lock file was previously ignored.
2026-11-20 | Keep strict mypy for completed modules and temporarily quarantine modules belonging to later milestones | The execution plan is strictly ordered; quarantined modules are re-enabled as their milestones are completed instead of weakening the global type-checking policy.
2026-11-20 | Add `scripts/check.py` as a Windows-compatible equivalent of `make lint test` | The development shell has no `make`; CI and Unix developers keep using the Makefile, while local verification remains reproducible.
