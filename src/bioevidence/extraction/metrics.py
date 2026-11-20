"""Extraction evaluation metrics for BioEvidence."""

import json
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ExtractionMetrics:
    """Metrics for a single field extraction."""

    field: str
    token_f1: float = 0.0
    exact_match: float = 0.0
    set_f1: float = 0.0


@dataclass
class ExtractionResult:
    """Result of extraction evaluation."""

    nct_id: str
    field_metrics: dict[str, ExtractionMetrics]
    macro_f1: float = 0.0
    valid_json: bool = True
    latency_ms: float = 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between two strings."""
    if not prediction and not ground_truth:
        return 1.0
    if not prediction or not ground_truth:
        return 0.0

    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    if not pred_tokens or not gt_tokens:
        return 0.0

    intersection = pred_tokens & gt_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: Any, ground_truth: Any) -> float:
    """Exact match for categorical/boolean fields."""
    return 1.0 if prediction == ground_truth else 0.0


def set_f1(prediction: list[str], ground_truth: list[str]) -> float:
    """Set F1 for list fields."""
    if not ground_truth:
        return 1.0 if not prediction else 0.0

    pred_set = {p.lower().strip() for p in prediction if p}
    gt_set = {g.lower().strip() for g in ground_truth if g}

    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0

    intersection = pred_set & gt_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gt_set)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_field_metrics(
    field: str,
    prediction: Any,
    ground_truth: Any,
) -> ExtractionMetrics:
    """Compute metrics for a single field."""
    if field in ["primary_endpoint"]:
        return ExtractionMetrics(
            field=field,
            token_f1=token_f1(str(prediction), str(ground_truth)),
            exact_match=exact_match(prediction, ground_truth),
        )
    elif (
        field in ["endpoint_category", "intervention_class", "blinding", "population_age_group"]
        or field == "randomized"
    ):
        return ExtractionMetrics(
            field=field,
            exact_match=exact_match(prediction, ground_truth),
        )
    elif field in ["intervention_names", "key_inclusion", "key_exclusion"]:
        return ExtractionMetrics(
            field=field,
            set_f1=set_f1(
                prediction if isinstance(prediction, list) else [],
                ground_truth if isinstance(ground_truth, list) else [],
            ),
        )
    else:
        return ExtractionMetrics(field=field)


def evaluate_extraction(
    prediction: dict[str, Any],
    ground_truth: dict[str, Any],
) -> ExtractionResult:
    """Evaluate a single extraction against ground truth."""
    field_metrics = {}

    # Define expected fields
    expected_fields = [
        "primary_endpoint",
        "endpoint_category",
        "intervention_names",
        "intervention_class",
        "randomized",
        "blinding",
        "population_age_group",
        "key_inclusion",
        "key_exclusion",
    ]

    for field in expected_fields:
        pred = prediction.get(field)
        gt = ground_truth.get(field)
        field_metrics[field] = compute_field_metrics(field, pred, gt)

    # Compute macro F1 (average of field-level F1s)
    f1_scores = []
    for field, metrics in field_metrics.items():
        if field in ["primary_endpoint"]:
            f1_scores.append(metrics.token_f1)
        elif field in [
            "endpoint_category",
            "intervention_class",
            "blinding",
            "population_age_group",
            "randomized",
        ]:
            f1_scores.append(metrics.exact_match)
        elif field in ["intervention_names", "key_inclusion", "key_exclusion"]:
            f1_scores.append(metrics.set_f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Check valid JSON
    try:
        json.dumps(prediction)
        valid_json = True
    except (TypeError, ValueError):
        valid_json = False

    return ExtractionResult(
        nct_id="",
        field_metrics=field_metrics,
        macro_f1=macro_f1,
        valid_json=valid_json,
    )


def aggregate_extraction_results(results: list[ExtractionResult]) -> dict[str, Any]:
    """Aggregate extraction results across multiple trials."""
    if not results:
        return {}

    # Per-field metrics
    field_names = results[0].field_metrics.keys()
    per_field = {}

    for field in field_names:
        token_f1s = [
            r.field_metrics[field].token_f1 for r in results if r.field_metrics[field].token_f1 > 0
        ]
        exact_matches = [
            r.field_metrics[field].exact_match
            for r in results
            if hasattr(r.field_metrics[field], "exact_match")
        ]
        set_f1s = [
            r.field_metrics[field].set_f1 for r in results if r.field_metrics[field].set_f1 > 0
        ]

        per_field[field] = {
            "token_f1": sum(token_f1s) / len(token_f1s) if token_f1s else 0.0,
            "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
            "set_f1": sum(set_f1s) / len(set_f1s) if set_f1s else 0.0,
        }

    # Macro F1
    macro_f1 = sum(r.macro_f1 for r in results) / len(results)
    valid_json_rate = sum(1 for r in results if r.valid_json) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)

    return {
        "per_field": per_field,
        "macro_f1": macro_f1,
        "valid_json_rate": valid_json_rate,
        "avg_latency_ms": avg_latency,
        "total_trials": len(results),
    }


def evaluate_extraction_dataset(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a dataset of extractions."""
    results = []

    for pred, gt in zip(predictions, ground_truths, strict=False):
        result = evaluate_extraction(pred, gt)
        results.append(result)

    return aggregate_extraction_results(results)


# Threshold for passing
EXTRACTION_MACRO_F1_THRESHOLD = 0.60
EXTRACTION_VALID_JSON_THRESHOLD = 0.95


def check_extraction_thresholds(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check if extraction metrics meet thresholds."""
    failures = []

    macro_f1 = metrics.get("macro_f1", 0.0)
    if macro_f1 < EXTRACTION_MACRO_F1_THRESHOLD:
        failures.append(f"macro_f1={macro_f1:.4f} < {EXTRACTION_MACRO_F1_THRESHOLD:.4f}")

    valid_json_rate = metrics.get("valid_json_rate", 0.0)
    if valid_json_rate < EXTRACTION_VALID_JSON_THRESHOLD:
        failures.append(
            f"valid_json_rate={valid_json_rate:.4f} < {EXTRACTION_VALID_JSON_THRESHOLD:.4f}"
        )

    return len(failures) == 0, failures
