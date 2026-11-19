# 02 Data Pipeline

## Sources
1. **ClinicalTrials.gov API v2**: `GET https://clinicaltrials.gov/api/v2/studies` with `query.cond`, `filter.overallStatus`,
   `pageSize=1000`, `pageToken` pagination. Request needed fields only. Keep interventional studies, start date >= 2005.
2. **PubMed E-utilities**: `esearch` (term from scope + MeSH, `usehistory=y`) then `efetch` in batches of 200, XML -> parse
   PMID, title, abstract, journal, year, MeSH terms. Use `NCBI_API_KEY` if set. Skip records without abstract.
3. **openFDA drug labels**: prefer bulk download JSON from open.fda.gov/apis/downloads; fallback `api.fda.gov/drug/label.json`.
   Keep drugs whose generic/brand name matches intervention names from selected trials. Sections to keep: indications_and_usage,
   dosage_and_administration, contraindications, warnings_and_cautions, adverse_reactions, drug_interactions, clinical_studies.

Cache raw responses to `data/raw/<source>/` (gzip JSON). Ingestion reads cache first. Every record keeps `source_url`, `fetched_at`.

## Schema (Postgres)
```
trials(nct_id PK, title, brief_summary, detailed_description, eligibility_text, phase, overall_status, study_type,
       allocation, masking, enrollment, enrollment_type, start_date, completion_date, sponsor, sponsor_class,
       min_age_years, max_age_years, sex, countries text[], num_arms, source_url, fetched_at)
trial_conditions(nct_id FK, condition, condition_mesh)
trial_interventions(nct_id FK, name, type, description)
trial_outcomes(nct_id FK, kind ENUM[primary,secondary], measure, time_frame, description)
pubmed_articles(pmid PK, title, abstract, journal, year, mesh_terms text[], source_url, fetched_at)
drug_labels(set_id PK, brand_names text[], generic_names text[], sections jsonb, effective_date, source_url)
chunks(id PK, doc_type ENUM[trial,pubmed,label], doc_id, section, text, token_count, metadata jsonb,
       embedding vector(384), tsv tsvector GENERATED, split ENUM[train,dev,test])
```
Indexes: HNSW on `chunks.embedding` (cosine), GIN on `tsv`, btree on `trials(phase, start_date, overall_status)`.
Use Alembic migrations. Embedding dimension is config-driven (384 for bge-small; re-embedding is a CLI command).

## Cleaning rules
- Normalize phases ("PHASE2/PHASE3" -> list), dates to ISO (partial dates -> first of month, flag `date_imputed`).
- Parse age strings ("18 Years") to numeric. Strip HTML/markup, collapse whitespace.
- Dedupe PubMed by PMID, labels by set_id (keep latest effective_date).
- Drop trials with empty summary AND empty description.

## Data quality checks (write `reports/data_quality.md` automatically)
Pydantic validation at ingest + checks in `quality/checks.py`:
- Not-null: nct_id, title, at least one text field.
- Uniqueness: PKs. Referential integrity: interventions/outcomes -> trials.
- Ranges: enrollment >= 0, completion_date >= start_date for completed trials, year within 1950..current.
- Completeness table per column (% null). Text length distribution. Language sanity (non-English flag).
- Row counts per source and per run; diff vs previous run (new/updated/dropped).
- Fail the pipeline if critical checks fail; warn on soft checks. Quarantine bad rows to `data/processed/rejected_*.jsonl` with reason.

## Chunking
- Trials: chunks per section (summary, description, eligibility inclusion, eligibility exclusion, outcomes). Prefix each chunk with
  a header: `[Trial NCT0123 | Phase 3 | Condition: ...]` so chunks are self-describing.
- PubMed: title + abstract as 1 chunk (split by 256 tokens w/ 32 overlap only if longer; abstracts rarely are).
- Labels: per section; split long sections at paragraph boundaries to <= 350 tokens, overlap 40.
- Store `token_count` using the embedding model tokenizer. Max chunk 512 tokens.

## Splits (create once, commit ids)
Split by **document id** (not chunk): 80/10/10 train/dev/test for trials and PubMed, stratified by doc_type. Stored in
`data/splits/doc_splits.json`. Embedding/reranker/LoRA training uses ONLY train docs. Eval golden set uses ONLY test docs (+ dev for tuning).

## CLI
`ingest ctgov|pubmed|openfda`, `quality run`, `db migrate`, `chunk build`, `embed build --model <name>`, `splits make`.

## Acceptance criteria
- Clean clone -> `make ingest` populates DB with counts within targets; second run adds 0 duplicates.
- `reports/data_quality.md` generated with completeness table and zero critical failures.
- Tests: parser unit tests with saved fixture JSON/XML for each source; chunker tests; split-leakage test.
