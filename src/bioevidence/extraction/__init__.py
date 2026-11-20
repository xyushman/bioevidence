"""BioEvidence extraction package."""

from bioevidence.extraction.constrained import (
    ConstrainedExtractor,
    LoRAExtractor,
    extract_trial_fields,
)
from bioevidence.extraction.metrics import (
    EXTRACTION_MACRO_F1_THRESHOLD,
    EXTRACTION_VALID_JSON_THRESHOLD,
    ExtractionMetrics,
    ExtractionResult,
    aggregate_extraction_results,
    check_extraction_thresholds,
    compute_field_metrics,
    evaluate_extraction,
    evaluate_extraction_dataset,
    exact_match,
    set_f1,
    token_f1,
)
from bioevidence.extraction.prepare_data import (
    AGE_GROUPS,
    BLINDING_TYPES,
    ENDPOINT_CATEGORIES,
    EXTRACTION_SCHEMA,
    INTERVENTION_CLASSES,
    build_trial_text,
    create_extraction_prompt,
    load_trial_data,
    prepare_extraction_dataset,
)

__all__ = [
    "ConstrainedExtractor",
    "LoRAExtractor",
    "extract_trial_fields",
    "EXTRACTION_MACRO_F1_THRESHOLD",
    "EXTRACTION_VALID_JSON_THRESHOLD",
    "ExtractionMetrics",
    "ExtractionResult",
    "aggregate_extraction_results",
    "check_extraction_thresholds",
    "compute_field_metrics",
    "evaluate_extraction",
    "evaluate_extraction_dataset",
    "exact_match",
    "set_f1",
    "token_f1",
    "ENDPOINT_CATEGORIES",
    "INTERVENTION_CLASSES",
    "BLINDING_TYPES",
    "AGE_GROUPS",
    "EXTRACTION_SCHEMA",
    "create_extraction_prompt",
    "prepare_extraction_dataset",
    "build_trial_text",
    "load_trial_data",
]
