"""Retrieval module for BioEvidence."""

from dataclasses import dataclass
from typing import Any

from bioevidence.config import get_config
from bioevidence.eval.retrieval_metrics import (
    RetrievalMetrics,
    compute_metrics,
    evaluate_retrieval,
)

__all__ = [
    "RetrievalMetrics",
    "compute_metrics",
    "evaluate_retrieval",
]
