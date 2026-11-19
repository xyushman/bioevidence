# 09 Tasks: Execution Plan

Work top to bottom. Tick `[x]` only when the acceptance criterion is met. Commit after each task.

## Milestone 0: Scaffold (day 1)
- [ ] M0.1 Repo scaffold per `01-PROJECT_SPEC.md`: uv project, ruff, pytest, Makefile, pre-commit, `.env.example`, `.gitignore`, CI skeleton. AC: `make lint test` runs (even with 1 dummy test).
- [ ] M0.2 `docker-compose.yml` with pgvector Postgres, Alembic set up, `db migrate` creates schema from `02`. AC: tables + indexes exist.
- [ ] M0.3 `LLMClient` (gemini/anthropic/openai) + disk cache + stub client for tests. AC: second identical call is a cache hit (test).

## Milestone 1: Data (week 1)
- [ ] M1.1 CT.gov ingestor with pagination, raw cache, parser, upsert. AC: >= 20k trials, fixtures-based tests.
- [ ] M1.2 PubMed ingestor (esearch+efetch, rate limit). AC: >= 80k abstracts.
- [ ] M1.3 openFDA label ingestor. AC: >= 3k labels linked to intervention names.
- [ ] M1.4 Quality checks + `reports/data_quality.md` + quarantine. AC: zero critical failures, idempotent rerun test.
- [ ] M1.5 Chunker + `splits make` (by doc id) + leakage test. AC: chunks table populated, splits committed.
- [ ] M1.6 Sample dataset `data/sample/` (~500 trials, 2k abstracts, 200 labels) for demos/CI.

## Milestone 2: Eval foundation + baseline (week 1-2)
- [ ] M2.1 Retrieval metrics + unit tests on toy data.
- [ ] M2.2 Retrieval golden set (300 queries from TEST docs) + 60 human-verified. AC: JSONL committed, verification rate recorded.
- [ ] M2.3 A0 baseline (bge-small + pgvector), A1a BM25, A1b dense. AC: `results/retrieval/baseline.json`.
- [ ] M2.4 A2 hybrid with RRF. AC: beats both A1s on dev, or documented.
- [ ] M2.5 Generation + QA golden set (100, hand-reviewed) + judge + faithfulness/citation/abstention metrics. AC: A0 end-to-end numbers saved.

## Milestone 3: Model training (week 2-3)
- [ ] M3.1 Synthetic pair generation + filtering + hard-negative mining (TRAIN docs only). AC: >= 20k pairs, stats report.
- [ ] M3.2 Embedding fine-tune script + Colab notebook. AC: dev Recall@5 improves over base; saved checkpoint + MLflow run.
- [ ] M3.3 Re-embed chunks (`embedding_ft`) and A3 eval on TEST. AC: results JSON, CI intervals.
- [ ] M3.4 Reranker (off-the-shelf A4 + fine-tuned A4b), latency benchmark. AC: tradeoff table.
- [ ] M3.5 Full A5 pipeline + `eval/ablation.py` generating `results/ablation.md`. AC: complete table.

## Milestone 4: Extractor (week 3-4)
- [ ] M4.1 Extraction schema, structural gold labels, teacher silver labels (3k), 200 human-verified test set.
- [ ] M4.2 QLoRA training script + Colab notebook. AC: runs on T4, adapter saved.
- [ ] M4.3 Evaluate 4 systems (base zero-shot, few-shot, API, LoRA) with cost/latency. AC: `results/extraction/summary.md`.

## Milestone 5: Agent (week 4)
- [ ] M5.1 `sql_guard` + read-only DB role + injection tests. AC: all tests pass.
- [ ] M5.2 Tools + agent loop + prompts + tracing. AC: demo script handles 4 query types.
- [ ] M5.3 Tool-selection (60), security (15), end-to-end QA by category. AC: `results/agent/`.

## Milestone 6: Forecasting (week 4)
- [ ] M6.1 Features + time split + baselines + GBM. AC: metrics JSON.
- [ ] M6.2 SHAP, calibration, `reports/forecasting.md`.

## Milestone 7: Product + polish (week 5)
- [ ] M7.1 FastAPI endpoints + tests with stub LLM.
- [ ] M7.2 React UI (chat + citations, trial explorer, eval dashboard).
- [ ] M7.3 CI: lint, test, eval-ci gate, Docker build, Trivy. Prove gate fails on degraded retriever.
- [ ] M7.4 Error analysis (30 failures) + one fix with measured delta. AC: `reports/error_analysis.md`.
- [ ] M7.5 Final README from `10-REPORT_TEMPLATE.md`, architecture diagram (mermaid), 2-minute demo script, HF model cards.
- [ ] M7.6 (Bonus) Terraform AWS deploy, Langfuse tracing.

## Cut list if time is short (drop in this order)
Terraform -> trial explorer UI -> neural-net forecaster -> Matryoshka/e5 extras -> constrained decoding. NEVER cut: baseline, golden sets, ablation table, embedding fine-tune, LoRA extractor eval, SQL guard.
