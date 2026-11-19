# 04 Evaluation

Evaluation is the core of this project. Build it BEFORE the fancy models.

## Golden sets (all in `data/goldens/`, committed as JSONL)
1. **Retrieval set** (synthetic + verified): 300 queries from TEST docs (disjoint from training queries). Fields: query, gold_chunk_ids, style.
   Human-verify a random 60 (record pass rate).
2. **QA set** (hand-reviewed, 100 questions) from TEST docs. Categories and counts:
   - 30 factual lookup (label sections, e.g. contraindications)
   - 25 trial-structured (answerable via SQL, e.g. counts/filters)
   - 20 multi-hop / comparison (two trials or trial + label)
   - 20 unanswerable (no supporting evidence in corpus)
   - 5 adversarial (asks for medical advice, prompt injection inside a chunk)
   Fields: question, category, gold_answer (short), gold_citations (chunk/trial ids), answerable (bool). Generate drafts with an LLM,
   then manually review and edit every item. Record reviewer notes.
3. **Extraction set** (200 trials): hand-verified fields for `05-LORA_EXTRACTOR.md`.
4. **Tool-selection set** (60 prompts): prompt, expected tool(s), expected args pattern.

## Retrieval metrics
Recall@{1,5,10,20}, MRR@10, nDCG@10, Hit@5; latency p50/p95. Report per doc_type and per query style. Bootstrap 95% CI (1000 resamples).

## Generation metrics
- **Citation accuracy**: % of cited chunks that exist in retrieved set; % of answers whose citations contain a gold citation.
- **Faithfulness**: LLM judge decomposes the answer into claims, checks each against cited chunks -> % supported. Judge model must differ
  from the generator when possible. Prompt + rubric stored in `eval/judge.py`. Human spot-check 30 answers, report agreement (Cohen's kappa).
- **Answer correctness**: LLM judge vs gold_answer (0/1/partial) + exact-match for numeric answers.
- **Abstention**: precision/recall of "insufficient evidence" on unanswerable vs answerable questions. Hallucination rate = unsupported
  claims / total claims, and % of unanswerable questions answered anyway.
- **Cost/latency**: tokens in/out, $ per query (from price table in config), end-to-end p50/p95.

## Ablation protocol (`eval/ablation.py` -> `results/ablation.md`)
| System | Config | Recall@5 | MRR@10 | nDCG@10 | Faithfulness | Citation acc | Abstain F1 | p95 ms |
Rows: A0 baseline, A1a BM25, A1b dense, A2 hybrid, A3 +FT embeddings, A4 +reranker (off-the-shelf), A4b +FT reranker, A5 full.
Fixed seeds, same questions, same generator/prompt/temperature across rows. Only one variable changes per row. Include CIs.

## CI gate
`make eval-ci` runs a small fixed subset (50 retrieval queries, 20 QA, cached LLM calls) and fails if Recall@5 < threshold in
`configs/eval_gate.yaml` (set to baseline-A5 minus 2 pts) or faithfulness drops by > 3 pts vs `results/main_baseline.json`.

## Error analysis (`reports/error_analysis.md`)
Sample 30 failures from A5 (QA + retrieval). Label each: retrieval miss, bad chunking, wrong filter, reranker error, generator hallucination,
over-abstention, gold-label error. Show counts, 5 detailed examples (query, retrieved, answer, root cause), and the fix applied + metric delta.

## Acceptance criteria
- `make eval` produces `results/ablation.md` + JSON, deterministic given cache.
- Judge prompts versioned; human agreement reported.
- Tests: metric functions verified on toy cases (hand-computed MRR/nDCG).
