# 08 API, UI, Deployment

## FastAPI (`api/main.py`)
- `POST /chat` {message, session_id?, system?: "A0".."A5"} -> agent response schema (see 06). Streaming (SSE) optional.
- `POST /search` {query, k, filters} -> hits with scores + which stage produced them.
- `GET /trial/{nct_id}` -> structured trial + extracted fields.
- `POST /forecast` (optional) -> predicted duration + interval + top SHAP features.
- `GET /eval/ablation` -> latest ablation JSON for the dashboard. `GET /health`, `GET /metrics` (Prometheus, optional).
- Pydantic request/response models, CORS config, request-id middleware, per-IP rate limit (slowapi), timeouts, structured logs.
- Tests with `httpx.AsyncClient` and a stub LLM client (no network).

## React UI (`web/`, Vite + TS + Tailwind)
Keep it small and clean (3 views):
1. **Chat**: message list, citations panel on the right (click `[n]` highlights source snippet + link to source URL), tools-used badges,
   "show SQL" toggle, abstention styled distinctly, system selector (A0 vs A5) to demo the difference live.
2. **Trial explorer**: filter by phase/condition/year, table, detail drawer with extracted fields and forecast.
3. **Eval dashboard**: ablation table + bar chart (Recall@5, faithfulness), latency chart, error-analysis category counts (reads `/eval/ablation`).
Persistent medical disclaimer banner. Loading/error states. No auth.

## Docker / local
- `docker-compose.yml`: `db` (pgvector/pgvector:pg16), `api`, `web`. Healthchecks. Volumes for pgdata. `.env.example` for keys.
- `Makefile` targets: `setup, up, down, migrate, ingest, quality, chunks, embed, train-embed, train-rerank, train-lora, eval, eval-ci, forecast, api, web, lint, test`.
- Multi-stage Dockerfile for API (slim image, non-root user). Models loaded from `models/` or HF Hub at startup, cached.

## CI (`.github/workflows/ci.yml`)
Jobs: lint (ruff), test (pytest with Postgres service container + fixtures), `eval-ci` quality gate (cached LLM calls committed in `tests/fixtures/llm_cache/`),
web build/typecheck, Docker build, Trivy image scan. Cache uv and HF models.

## Optional AWS (bonus, do last)
Terraform in `infra/terraform/`: VPC, EC2 (or ECS Fargate) running compose/api, RDS Postgres with pgvector, S3 for artifacts, SSM for secrets,
GitHub OIDC role (no long-lived keys). Provide `terraform plan` output in README and a `make destroy` reminder (cost control). Optional Langfuse for traces.

## Acceptance criteria
- `docker compose up` -> UI reachable, chat answers with citations using the demo dataset (provide a tiny seed dataset `data/sample/` so reviewers can run without full ingestion).
- CI green; gate fails when a deliberately degraded retriever is committed (prove it once, note in README).
