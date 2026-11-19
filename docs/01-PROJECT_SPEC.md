# 01 Project Spec

## Problem
Researchers and analysts need grounded answers about drugs and clinical trials ("Which phase 3 trials for X reported
overall survival as primary endpoint?", "What are the contraindications of Y?"). Answers must cite sources and say
"insufficient evidence" when appropriate.

## Scope (keep it small and finish it)
- Therapeutic scope: oncology, inflammation/immunology, cardiovascular, bone/osteoporosis (Amgen-relevant). Configurable list of
  condition keywords in `configs/scope.yaml`.
- Volume targets: ~20-30k trials (interventional, 2005+), ~100k PubMed abstracts (MeSH-filtered), openFDA labels for
  drugs appearing in the selected trials (~3-5k labels).
- English only. Text only (no images/PDF parsing).

## Non-goals
Authentication, multi-tenancy, real-time ingestion, medical advice, fine-tuning the generator LLM for chat.

## Architecture
```
 Sources (CT.gov v2 API, PubMed E-utilities, openFDA bulk)
        |  ingest (Typer CLI, idempotent upserts, raw JSON cached to data/raw/)
        v
 Quality layer (pydantic validation + Great Expectations-style checks -> reports/data_quality.md)
        v
 Postgres 16 + pgvector
   - trials, trial_interventions, trial_outcomes, conditions        (structured, SQL tool)
   - pubmed_articles, drug_labels                                    (documents)
   - chunks (text, metadata, embedding vector, tsvector)             (retrieval)
        v
 Retrieval: BM25/tsvector + dense (pgvector) -> RRF fusion -> cross-encoder reranker
        v
 Agent (tool loop): search_evidence | sql_query | compare_trials | extract_trial_fields
        v
 FastAPI (/chat, /search, /trial/{nct}, /health) -> React UI (chat, citations panel, eval dashboard)

 Offline ML track: synthetic data gen -> embedding fine-tune -> reranker fine-tune -> QLoRA extractor -> forecasting
 Eval track: golden set -> metrics -> ablation table -> CI gate
```

## Tech stack
Python 3.11, uv, FastAPI, SQLAlchemy 2 + psycopg 3, Postgres 16 + pgvector, `sentence-transformers`, `transformers`, `peft`,
`trl`, `bitsandbytes`, `rank_bm25` (for offline eval) / Postgres FTS (for serving), `xgboost`/`lightgbm`, `shap`, `scikit-learn`,
MLflow, Typer, pytest, ruff, Docker Compose, React + Vite + TypeScript + Tailwind, GitHub Actions. Optional: Terraform (AWS), Langfuse.

## Repo layout
```
bioevidence/
  AGENTS.md  README.md  Makefile  pyproject.toml  docker-compose.yml  .env.example
  configs/        scope.yaml  retrieval.yaml  train_embed.yaml  train_rerank.yaml  train_lora.yaml  agent.yaml
  docs/           (these specs) DECISIONS.md
  data/           raw/ processed/ splits/ goldens/   (raw + processed gitignored; goldens + splits committed)
  results/        <experiment>/*.json  ablation.md   (committed)
  reports/        data_quality.md  error_analysis.md
  src/bioevidence/
    config.py  llm/client.py  llm/cache.py
    ingest/{ctgov.py,pubmed.py,openfda.py}
    quality/{schemas.py,checks.py}
    db/{models.py,migrations/,session.py}
    chunking/chunker.py
    retrieval/{bm25.py,dense.py,hybrid.py,rerank.py,pipeline.py}
    training/{synth_pairs.py,hard_negatives.py,train_embed.py,train_rerank.py}
    extraction/{schema.py,silver_labels.py,train_lora.py,infer.py}
    agent/{tools.py,sql_guard.py,loop.py,prompts.py}
    forecasting/{features.py,train.py,explain.py}
    eval/{golden.py,retrieval_metrics.py,gen_metrics.py,judge.py,ablation.py,run.py}
    api/{main.py,schemas.py}
  web/            React app
  notebooks/      colab_train_embed.ipynb  colab_train_lora.ipynb (thin wrappers around scripts)
  infra/          terraform/ (optional)
  tests/
  .github/workflows/ci.yml
```
