# 10 Final Report / README Template

The final `README.md` of the repo should contain, in order (fill with real numbers from `results/`, never invented):

1. **One-paragraph pitch** + screenshot/GIF of chat with citations.
2. **Headline results** (3-4 bullets with numbers):
   - Recall@5: A0 X -> A5 Y (+Z pts); MRR@10 ...
   - Fine-tuned embeddings: +N pts Recall@5 on held-out test, trained on 20k synthetic pairs.
   - LoRA 3B extractor: F1 = a vs API LLM b at c x lower cost per 1k trials.
   - Faithfulness / abstention numbers; forecasting MAE vs baseline.
3. **Architecture diagram** (mermaid) and data sources.
4. **Ablation table** (copy of `results/ablation.md`) + one paragraph of interpretation.
5. **Extraction results table** (4 systems).
6. **Agent**: tools, SQL safety design, tool-selection and security eval results.
7. **Forecasting**: metric table, SHAP plot, limitations (right-censoring).
8. **Error analysis**: categories + top 3 lessons.
9. **Reproduce**: setup, `make` targets, sample dataset quick start, GPU notes (Colab links).
10. **Limitations and ethics**: not medical advice, data licensing/attribution, synthetic-label bias, small human-verified sets.
11. **Repo map** and links to model cards.

## Resume bullets (fill X/Y after results exist)
- Built a biomedical RAG + agent system over 3 public sources (CT.gov, PubMed, openFDA); fine-tuned bge embeddings on 20k synthetic
  query-passage pairs with hard negatives, raising Recall@5 from X to Y on held-out test queries.
- QLoRA-tuned Qwen2.5-3B for structured trial-field extraction: Z% macro-F1 vs W% for a zero-shot API model at N x lower cost per 1k trials.
- Designed an evaluation harness (retrieval, faithfulness, citation accuracy, abstention) with CI quality gate; read-only SQL agent with AST validation.
- Forecasted clinical-trial duration with gradient boosting (MAE A months vs B baseline) using time-based validation and SHAP analysis.

## Interview talking points
Why synthetic data + how you prevented leakage; why hard negatives skip top-10; hybrid vs dense tradeoffs; LLM-judge bias and how you validated it;
what failed and what you changed; cost/latency tradeoff of reranker and LoRA vs API; right-censoring in duration forecasting.
