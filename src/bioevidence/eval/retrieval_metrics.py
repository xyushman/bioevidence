"""Retrieval metrics for BioEvidence."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RetrievalMetrics:
    """Container for retrieval metrics."""

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    hit_at_5: float = 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Recall@k."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute MRR@k (Mean Reciprocal Rank)."""
    if not relevant:
        return 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute nDCG@k."""
    if not relevant:
        return 0.0

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)

    # IDCG (ideal DCG)
    ideal_relevant = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Hit@k (1 if any relevant in top-k, else 0)."""
    if not relevant:
        return 0.0
    return 1.0 if any(doc_id in relevant for doc_id in retrieved[:k]) else 0.0


def compute_metrics(
    retrieved: list[str],
    relevant: set[str],
    ks: list[int] = None,
) -> RetrievalMetrics:
    """Compute all retrieval metrics for a single query."""
    if ks is None:
        ks = [1, 5, 10, 20]
    metrics = RetrievalMetrics()

    for k in ks:
        if k == 1:
            metrics.recall_at_1 = recall_at_k(retrieved, relevant, k)
        elif k == 5:
            metrics.recall_at_5 = recall_at_k(retrieved, relevant, k)
            metrics.hit_at_5 = hit_at_k(retrieved, relevant, k)
        elif k == 10:
            metrics.recall_at_10 = recall_at_k(retrieved, relevant, k)
            metrics.mrr_at_10 = mrr_at_k(retrieved, relevant, k)
            metrics.ndcg_at_10 = ndcg_at_k(retrieved, relevant, k)
        elif k == 20:
            metrics.recall_at_20 = recall_at_k(retrieved, relevant, k)

    return metrics


def aggregate_metrics(
    all_metrics: list[RetrievalMetrics],
) -> RetrievalMetrics:
    """Aggregate metrics across multiple queries (mean)."""
    if not all_metrics:
        return RetrievalMetrics()

    agg = RetrievalMetrics()
    n = len(all_metrics)

    agg.recall_at_1 = sum(m.recall_at_1 for m in all_metrics) / n
    agg.recall_at_5 = sum(m.recall_at_5 for m in all_metrics) / n
    agg.recall_at_10 = sum(m.recall_at_10 for m in all_metrics) / n
    agg.recall_at_20 = sum(m.recall_at_20 for m in all_metrics) / n
    agg.mrr_at_10 = sum(m.mrr_at_10 for m in all_metrics) / n
    agg.ndcg_at_10 = sum(m.ndcg_at_10 for m in all_metrics) / n
    agg.hit_at_5 = sum(m.hit_at_5 for m in all_metrics) / n

    return agg


def bootstrap_ci(
    all_metrics: list[RetrievalMetrics],
    metric_name: str,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric."""
    if not all_metrics:
        return (0.0, 0.0)

    values = [getattr(m, metric_name) for m in all_metrics]
    n = len(values)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return (float(lower), float(upper))


def evaluate_retrieval(
    queries: list[dict[str, Any]],
    retriever,
    ks: list[int] = None,
) -> dict[str, Any]:
    """Evaluate retrieval system on a set of queries."""
    if ks is None:
        ks = [1, 5, 10, 20]
    all_metrics = []

    for query in queries:
        query_text = query["query"]
        gold_chunks = set(query["gold_chunk_ids"])

        # Retrieve
        hits = retriever.search(query_text, k=max(ks))
        retrieved = [hit.chunk_id for hit in hits]

        # Compute metrics
        metrics = compute_metrics(retrieved, gold_chunks, ks)
        all_metrics.append(metrics)

    # Aggregate
    agg = aggregate_metrics(all_metrics)

    # Bootstrap CIs
    cis = {}
    for metric_name in [
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "mrr_at_10",
        "ndcg_at_10",
        "hit_at_5",
    ]:
        cis[metric_name] = bootstrap_ci(all_metrics, metric_name)

    return {
        "aggregate": agg,
        "cis": cis,
        "per_query": all_metrics,
        "n_queries": len(queries),
    }
