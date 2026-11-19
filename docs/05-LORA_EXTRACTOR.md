# 05 QLoRA Structured Extractor

Goal: a small open model that turns messy trial text into validated JSON, compared against a zero-shot API LLM on accuracy, cost, latency.

## Task
Input: trial `brief_summary + detailed_description + eligibility_text + outcomes text` (structured metadata fields REMOVED to avoid shortcut leakage).
Output JSON (pydantic `TrialExtraction`):
```
{
 "primary_endpoint": str,
 "endpoint_category": enum[overall_survival, progression_free_survival, response_rate, safety, pk_pd, biomarker, patient_reported, other],
 "intervention_names": [str],
 "intervention_class": enum[small_molecule, biologic, vaccine, device, behavioral, other],
 "randomized": bool,
 "blinding": enum[none, single, double, triple_plus, unclear],
 "population_age_group": enum[pediatric, adult, older_adult, mixed],
 "key_inclusion": [str] (max 3),
 "key_exclusion": [str] (max 3)
}
```
Unknown -> null / "unclear". Never invent.

## Labels
- **Gold-from-structure where possible**: `randomized` (allocation), `blinding` (masking), `intervention_names`, `population_age_group`
  (min/max age) can be derived from structured fields, giving free, reliable labels.
- **Silver via teacher LLM** for `primary_endpoint`, `endpoint_category`, `intervention_class`, `key_inclusion/exclusion`: ~3,000 TRAIN trials,
  teacher = strongest available API model, temperature 0, schema-constrained output, cached. Validate silver vs structured overlap and report agreement.
- **Test set**: 200 TEST trials hand-verified (see 04). Report inter-annotator or teacher-vs-human agreement.
- Split 2,700 train / 300 val (from TRAIN docs). Test is never touched during training.

## Training (`extraction/train_lora.py`)
- Base: `Qwen/Qwen2.5-3B-Instruct` (alt: `meta-llama/Llama-3.2-3B-Instruct`). 4-bit NF4 (bitsandbytes), compute dtype fp16 (T4) / bf16.
- LoRA: r=16, alpha=32, dropout 0.05, target all linear proj (q,k,v,o,gate,up,down). `trl` SFTTrainer, chat template, loss on completion only.
- Max seq len 1024 (truncate long descriptions smartly: keep summary, outcomes, first part of eligibility), lr 2e-4 cosine, 2-3 epochs,
  effective batch 16 via grad accumulation, gradient checkpointing. Log to MLflow. Fit on free Colab T4.
- Save adapter; optional merge + push to HF Hub with model card.

## Inference
`extraction/infer.py`: batched generation with greedy decoding, JSON parsed + validated by pydantic; on failure, one retry with repair prompt;
track invalid-JSON rate. Optional constrained decoding (`outlines`) as an extra row.

## Evaluation (`results/extraction/`)
Systems: (1) base 3B zero-shot, (2) base 3B few-shot, (3) API LLM zero-shot, (4) LoRA 3B.
Metrics per field: exact match (enums/bools), token-level F1 (strings), set F1 (lists); macro average; valid-JSON rate;
latency/trial; cost per 1k trials (API tokens price vs GPU-seconds estimate). Include a per-field error table and 10 error examples.

## Acceptance criteria
- Train script runs end-to-end on a T4 in < 90 min with the default config.
- Results table with all 4 systems and CIs. LoRA beats base 3B by a clear margin; document honestly vs API model.
- Extractor exposed to the agent as `extract_trial_fields(nct_id)` (uses adapter if available, else API fallback).
