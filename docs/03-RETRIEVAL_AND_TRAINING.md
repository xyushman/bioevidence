# 03 Retrieval and Model Training

All retrievers implement `Retriever.search(query, k, filters) -> list[Hit(chunk_id, score, text, metadata)]`.
Every config below is a named system (A0..A5) evaluated by `04-EVALUATION.md`.

## A0: Naive baseline (must exist first)
Dense retrieval, `BAAI/bge-small-en-v1.5` (off the shelf), top-k cosine over pgvector, no rerank. Generator gets top-5.

## A1: BM25 only and dense only (sanity references)
Report both; hybrid should beat each.

## A2: Hybrid
BM25 (Postgres FTS `ts_rank_cd` or `rank_bm25` offline) + dense, fused with Reciprocal Rank Fusion (k=60). Tune candidate pool
size (50/100) on dev. Metadata filters: doc_type, phase, year range.

## A3: Fine-tuned embeddings
**Synthetic data** (`training/synth_pairs.py`):
- Sample ~20k chunks from TRAIN docs only, stratified by doc_type.
- For each, prompt an LLM for 2 queries in different styles (keyword-style, natural question, clinician-phrased). Prompt must forbid
  copying long spans (avoid lexical leakage). Filter: drop queries with >60% token overlap with the chunk, dedupe, round-trip check
  (the base model must retrieve the source chunk in top-50, else keep as "hard" but flag).
- Output `data/processed/synth_pairs.jsonl` {query, positive_chunk_id, style}.
**Hard negatives** (`hard_negatives.py`): mine with base model, rank 10-50 (skip top-10 to avoid false negatives), 3 per query,
exclude chunks from same trial/article as the positive.
**Training** (`train_embed.py`): `sentence-transformers` v3 trainer, `MultipleNegativesRankingLoss` (+ hard negatives as triplets),
bge-small, batch 64 (use CachedMNRL if memory-tight), lr 2e-5, 1-3 epochs, warmup 10%, fp16, seed 42, query instruction prefix
"Represent this sentence for searching relevant passages: " handled consistently at train and inference. Evaluate on dev every
N steps with `InformationRetrievalEvaluator`; keep best checkpoint. Optionally try `e5-small-v2` or Matryoshka loss as an extra ablation.
Re-embed all chunks with the new model, write to a second column/table (`embedding_ft`) so baseline stays available.
Push model to HF Hub (optional) with a model card (data, metrics, limits).

## A4: Reranker
Two-stage: retrieve 50 -> cross-encoder -> top-5.
- Off-the-shelf: `BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2` (report both).
- Fine-tune MiniLM cross-encoder on (query, pos, hard negs from fine-tuned retriever), BCE or listwise loss, 1-2 epochs.
- Measure latency (p50/p95 ms for 50 docs on CPU and on GPU) and the accuracy/latency tradeoff. Batch inference.

## A5: Full system
Fine-tuned embeddings + hybrid + fine-tuned reranker + query metadata filtering. Generator prompt with numbered citations `[1]`.

## Generation
`generate(query, hits)`: system prompt enforces: answer only from provided context, cite chunk ids, say
"Insufficient evidence in the retrieved sources." when context lacks the answer, no medical advice. Output JSON
{answer, citations[chunk_id], confidence_note}. Temperature 0. Max 5 chunks (config).

## Acceptance criteria
- `python -m bioevidence retrieve eval --system A0..A5` reproduces numbers in `results/retrieval/`.
- A3 beats A2/A0 on Recall@5 and MRR@10 on the TEST queries; if not, document honestly in error analysis and iterate
  (data quality, negatives, lr) before moving on.
- Tests: RRF fusion unit test, hard-negative exclusion test, no-test-doc-in-train test.
