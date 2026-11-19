"""Evaluation module for BioEvidence."""

from bioevidence.eval.generation_metrics import (
    GenerationMetrics,
    LLMJudge,
    QAEvalResult,
    aggregate_qa_metrics,
    evaluate_qa,
)
from bioevidence.eval.golden import (
    GoldenQuery,
    create_qa_golden_set,
    generate_synthetic_queries,
    load_golden_set,
    save_golden_set,
)
from bioevidence.eval.retrieval_metrics import (
    RetrievalMetrics,
    compute_metrics,
    evaluate_retrieval,
)

__all__ = [
    "GoldenQuery",
    "create_qa_golden_set",
    "generate_synthetic_queries",
    "load_golden_set",
    "save_golden_set",
    "GenerationMetrics",
    "LLMJudge",
    "QAEvalResult",
    "aggregate_qa_metrics",
    "evaluate_qa",
    "RetrievalMetrics",
    "compute_metrics",
    "evaluate_retrieval",
]
