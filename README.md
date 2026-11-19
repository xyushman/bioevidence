# BioEvidence: Biomedical Evidence Copilot (model-centric)

A production-style RAG + agent system over public biomedical data (ClinicalTrials.gov, openFDA drug labels, PubMed),
where the **main story is model work**: baseline -> hybrid retrieval -> fine-tuned embeddings -> fine-tuned reranker ->
LoRA-tuned extractor -> tool-calling agent -> trial-duration forecasting, with a numeric **ablation table** for every step.

## Why this project
Targets AI/Data engineer roles (e.g. Amgen AI and Data Intern). Maps to: LLMs, RAG, agents, Python + SQL, ML model
development and evaluation, forecasting, data quality/governance, pipelines, APIs, React UI, cloud/DevOps.

## Doc index (read in this order)
| File | Purpose |
|---|---|
| `AGENTS.md` | Rules, conventions, definition of done for the coding agent. READ FIRST. |
| `docs/01-PROJECT_SPEC.md` | Goals, scope, architecture, repo layout, tech stack |
| `docs/02-DATA_PIPELINE.md` | Ingestion, cleaning, quality checks, schema, chunking |
| `docs/03-RETRIEVAL_AND_TRAINING.md` | Baseline RAG, hybrid, embedding fine-tune, reranker |
| `docs/04-EVALUATION.md` | Golden set, metrics, LLM judge, ablation protocol |
| `docs/05-LORA_EXTRACTOR.md` | QLoRA structured-extraction model |
| `docs/06-AGENT.md` | Tool-calling agent, SQL safety, tool-selection eval |
| `docs/07-FORECASTING.md` | Trial duration prediction |
| `docs/08-API_UI_DEPLOY.md` | FastAPI, React UI, Docker, CI, optional AWS |
| `docs/09-TASKS.md` | Milestone checklist with acceptance criteria (the execution plan) |
| `docs/10-REPORT_TEMPLATE.md` | Final README/results/resume-bullet template |

## Success criteria (headline)
1. Reproducible: `make setup && make ingest && make eval` works from a clean clone.
2. Ablation table with real numbers for every model change.
3. Fine-tuned embeddings beat the off-the-shelf baseline on Recall@5 / MRR on a held-out set.
4. LoRA extractor evaluated against a zero-shot API LLM on accuracy, cost, latency.
5. Agent answers with citations, refuses when evidence is missing, SQL tool is provably read-only.
6. CI blocks merges when eval metrics regress.
