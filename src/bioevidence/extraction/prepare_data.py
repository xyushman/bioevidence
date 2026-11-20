"""Extraction data preparation for BioEvidence."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from bioevidence.config import get_config
from bioevidence.db.models import (
    Chunk,
    DocType,
    SplitType,
    Trial,
    TrialCondition,
    TrialIntervention,
    TrialOutcome,
)
from bioevidence.db.session import get_session

logger = structlog.get_logger(__name__)


@dataclass
class ExtractionExample:
    """Training example for extraction."""

    nct_id: str
    trial_text: str
    target: dict[str, Any]


EXTRACTION_SCHEMA = {
    "primary_endpoint": "string",
    "endpoint_category": "string",
    "intervention_names": "array[string]",
    "intervention_class": "string",
    "randomized": "boolean",
    "blinding": "string",
    "population_age_group": "string",
    "key_inclusion": "array[string]",
    "key_exclusion": "array[string]",
}

ENDPOINT_CATEGORIES = [
    "overall_survival",
    "progression_free_survival",
    "disease_free_survival",
    "response_rate",
    "complete_response",
    "partial_response",
    "safety",
    "pharmacokinetics",
    "quality_of_life",
    "other",
]

INTERVENTION_CLASSES = [
    "small_molecule",
    "monoclonal_antibody",
    "antibody_drug_conjugate",
    "bispecific_antibody",
    "cell_therapy",
    "gene_therapy",
    "vaccine",
    "protein",
    "peptide",
    "other",
]

BLINDING_TYPES = [
    "double",
    "single",
    "open",
    "triple",
    "quadruple",
    "not_applicable",
]

AGE_GROUPS = [
    "pediatric",
    "adult",
    "elderly",
    "all_ages",
]


def load_trial_data(nct_id: str) -> dict[str, Any] | None:
    """Load complete trial data from database."""
    with get_session() as session:
        trial = session.get(Trial, nct_id)
        if not trial:
            return None

        conditions = session.query(TrialCondition).filter(TrialCondition.nct_id == nct_id).all()
        interventions = (
            session.query(TrialIntervention).filter(TrialIntervention.nct_id == nct_id).all()
        )
        outcomes = session.query(TrialOutcome).filter(TrialOutcome.nct_id == nct_id).all()

        return {
            "trial": trial,
            "conditions": conditions,
            "interventions": interventions,
            "outcomes": outcomes,
        }


def build_trial_text(trial_data: dict) -> str:
    """Build text representation of trial for extraction."""
    trial = trial_data["trial"]
    conditions = trial_data["conditions"]
    interventions = trial_data["interventions"]
    outcomes = trial_data["outcomes"]

    parts = [
        f"NCT ID: {trial.nct_id}",
        f"Title: {trial.title}",
        f"Phase: {trial.phase}",
        f"Status: {trial.overall_status}",
        f"Enrollment: {trial.enrollment}",
    ]

    if trial.brief_summary:
        parts.append(f"Brief Summary: {trial.brief_summary}")
    if trial.detailed_description:
        parts.append(f"Detailed Description: {trial.detailed_description}")
    if trial.eligibility_text:
        parts.append(f"Eligibility: {trial.eligibility_text}")

    if conditions:
        parts.append(f"Conditions: {', '.join(c.condition for c in conditions)}")

    if interventions:
        parts.append(f"Interventions: {', '.join(i.name for i in interventions)}")

    if outcomes:
        primary = [o.measure for o in outcomes if o.kind.value == "primary"]
        secondary = [o.measure for o in outcomes if o.kind.value == "secondary"]
        if primary:
            parts.append(f"Primary Outcomes: {', '.join(primary)}")
        if secondary:
            parts.append(f"Secondary Outcomes: {', '.join(secondary)}")

    return "\n".join(parts)


def extract_target_from_trial(trial_data: dict) -> dict[str, Any]:
    """Extract ground truth target from trial data."""
    trial = trial_data["trial"]
    outcomes = trial_data["outcomes"]
    interventions = trial_data["interventions"]

    primary_outcomes = [o for o in outcomes if o.kind.value == "primary"]
    primary_endpoint = primary_outcomes[0].measure if primary_outcomes else "Not specified"
    endpoint_category = categorize_endpoint(primary_endpoint)

    intervention_names = [i.name for i in interventions]
    intervention_class = categorize_intervention(intervention_names)

    randomized = trial.allocation is not None and "random" in trial.allocation.lower()
    blinding = trial.masking or "not_applicable"

    age_group = categorize_age_group(trial.min_age_years, trial.max_age_years)

    inclusion, exclusion = parse_eligibility(trial.eligibility_text)

    return {
        "primary_endpoint": primary_endpoint,
        "endpoint_category": endpoint_category,
        "intervention_names": intervention_names,
        "intervention_class": intervention_class,
        "randomized": randomized,
        "blinding": blinding,
        "population_age_group": age_group,
        "key_inclusion": inclusion,
        "key_exclusion": exclusion,
    }


def categorize_endpoint(endpoint: str) -> str:
    """Categorize endpoint into standard categories."""
    endpoint_lower = endpoint.lower()

    if "overall survival" in endpoint_lower or endpoint_lower.strip() == "os":
        return "overall_survival"
    elif "progression free" in endpoint_lower or endpoint_lower.strip() == "pfs":
        return "progression_free_survival"
    elif "disease free" in endpoint_lower or endpoint_lower.strip() == "dfs":
        return "disease_free_survival"
    elif "response rate" in endpoint_lower or "objective response" in endpoint_lower:
        return "response_rate"
    elif "complete response" in endpoint_lower:
        return "complete_response"
    elif "partial response" in endpoint_lower:
        return "partial_response"
    elif "safety" in endpoint_lower or "adverse" in endpoint_lower:
        return "safety"
    elif "pharmacokinetic" in endpoint_lower or "pk" in endpoint_lower:
        return "pharmacokinetics"
    elif "quality of life" in endpoint_lower or "qol" in endpoint_lower:
        return "quality_of_life"
    else:
        return "other"


def categorize_intervention(intervention_names: list[str]) -> str:
    """Categorize intervention class."""
    if not intervention_names:
        return "other"

    combined = " ".join(intervention_names).lower()

    if any(term in combined for term in ["car-t", "car t", "cell therapy", "tcell", "t-cell"]):
        return "cell_therapy"
    elif any(term in combined for term in ["gene therapy", "gene transfer", "viral vector"]):
        return "gene_therapy"
    elif any(term in combined for term in ["vaccine"]):
        return "vaccine"
    elif any(term in combined for term in ["antibody-drug conjugate", "adc", "conjugate"]):
        return "antibody_drug_conjugate"
    elif any(term in combined for term in ["bispecific", "bi-specific", "bsab"]):
        return "bispecific_antibody"
    elif any(term in combined for term in ["monoclonal antibody", "mab", "mab-", "-mab"]):
        return "monoclonal_antibody"
    elif any(term in combined for term in ["protein", "fusion protein", "recombinant"]):
        return "protein"
    elif any(term in combined for term in ["peptide"]):
        return "peptide"
    elif any(
        term in combined for term in ["small molecule", "inhibitor", "tablet", "capsule", "oral"]
    ):
        return "small_molecule"
    else:
        return "other"


def categorize_age_group(min_age: float | None, max_age: float | None) -> str:
    """Categorize age group."""
    if min_age is None and max_age is None:
        return "all_ages"
    if min_age is not None and min_age < 18:
        return "pediatric"
    if min_age is not None and min_age >= 65:
        return "elderly"
    return "adult"


def parse_eligibility(eligibility_text: str | None) -> tuple[list[str], list[str]]:
    """Parse inclusion and exclusion criteria from eligibility text."""
    if not eligibility_text:
        return [], []

    import re

    inclusion = []
    exclusion = []

    inc_patterns = [
        r"inclusion criteria[:]\s*(.*?)(?:\n\s*exclusion|\n\s*$)",
        r"inclusion[:]\s*(.*?)(?:\n\s*exclusion|\n\s*$)",
        r"criteria for inclusion[:]\s*(.*?)(?:\n\s*exclusion|\n\s*$)",
    ]

    for pattern in inc_patterns:
        match = re.search(pattern, eligibility_text, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            items = re.split(r"\n\s*[\d\-\*]\s*", text)
            inclusion = [item.strip() for item in items if item.strip() and len(item.strip()) > 10]
            break

    exc_patterns = [
        r"exclusion criteria[:]\s*(.*?)(?:\n\s*inclusion|\n\s*$)",
        r"exclusion[:]\s*(.*?)(?:\n\s*inclusion|\n\s*$)",
        r"criteria for exclusion[:]\s*(.*?)(?:\n\s*inclusion|\n\s*$)",
    ]

    for pattern in exc_patterns:
        match = re.search(pattern, eligibility_text, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            items = re.split(r"\n\s*[\d\-\*]\s*", text)
            exclusion = [item.strip() for item in items if item.strip() and len(item.strip()) > 10]
            break

    if not inclusion and not exclusion:
        lines = [line.strip() for line in eligibility_text.split("\n") if line.strip()]
        mid = len(lines) // 2
        inclusion = lines[:mid]
        exclusion = lines[mid:]

    return inclusion[:10], exclusion[:10]


def create_extraction_prompt(trial_text: str) -> str:
    """Create prompt for trial extraction."""
    schema_str = json.dumps(EXTRACTION_SCHEMA, indent=2)

    return f"""<|system|>
You are a clinical trial extraction assistant. Extract structured fields from the trial text and return valid JSON.
<|user|>
Extract the following fields from this clinical trial text:

{trial_text}

Return JSON with these fields:
{schema_str}

Constraints:
- endpoint_category must be one of: {", ".join(ENDPOINT_CATEGORIES)}
- intervention_class must be one of: {", ".join(INTERVENTION_CLASSES)}
- blinding must be one of: {", ".join(BLINDING_TYPES)}
- population_age_group must be one of: {", ".join(AGE_GROUPS)}
- randomized must be boolean (true/false)
- intervention_names, key_inclusion, key_exclusion must be arrays of strings
<|assistant|>
"""


def prepare_extraction_dataset(
    num_train: int = 2700,
    num_val: int = 300,
    num_test: int = 200,
    seed: int = 42,
) -> tuple[list[ExtractionExample], list[ExtractionExample], list[ExtractionExample]]:
    """Prepare train/val/test datasets from database trials."""
    random.seed(seed)

    with get_session() as session:
        train_trials = (
            session.query(Trial.nct_id)
            .join(Chunk, Trial.nct_id == Chunk.doc_id)
            .filter(
                Chunk.split == SplitType.TRAIN,
                Chunk.doc_type == DocType.TRIAL,
            )
            .distinct()
            .all()
        )

        val_trials = (
            session.query(Trial.nct_id)
            .join(Chunk, Trial.nct_id == Chunk.doc_id)
            .filter(
                Chunk.split == SplitType.DEV,
                Chunk.doc_type == DocType.TRIAL,
            )
            .distinct()
            .all()
        )

        test_trials = (
            session.query(Trial.nct_id)
            .join(Chunk, Trial.nct_id == Chunk.doc_id)
            .filter(
                Chunk.split == SplitType.TEST,
                Chunk.doc_type == DocType.TRIAL,
            )
            .distinct()
            .all()
        )

        train_ids = [t[0] for t in train_trials][:num_train]
        val_ids = [t[0] for t in val_trials][:num_val]
        test_ids = [t[0] for t in test_trials][:num_test]

        logger.info("dataset_split", train=len(train_ids), val=len(val_ids), test=len(test_ids))

    def process_ids(nct_ids: list[str]) -> list[ExtractionExample]:
        examples = []
        for nct_id in nct_ids:
            trial_data = load_trial_data(nct_id)
            if not trial_data:
                continue

            trial_text = build_trial_text(trial_data)
            target = extract_target_from_trial(trial_data)

            examples.append(
                ExtractionExample(
                    nct_id=nct_id,
                    trial_text=trial_text,
                    target=target,
                )
            )
        return examples

    train_examples = process_ids(train_ids)
    val_examples = process_ids(val_ids)
    test_examples = process_ids(test_ids)

    logger.info(
        "prepared_examples",
        train=len(train_examples),
        val=len(val_examples),
        test=len(test_examples),
    )

    return train_examples, val_examples, test_examples


def save_extraction_examples(examples: list[ExtractionExample], path: Path) -> None:
    """Save extraction examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {
                        "nct_id": ex.nct_id,
                        "trial_text": ex.trial_text,
                        "target": ex.target,
                    }
                )
                + "\n"
            )


def load_extraction_examples(path: Path) -> list[ExtractionExample]:
    """Load extraction examples from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(ExtractionExample(**data))
    return examples


def main():
    """Generate and save extraction datasets."""
    config = get_config()
    data_dir = Path(config.llm.cache_dir).parent.parent / "processed"

    train_examples, val_examples, test_examples = prepare_extraction_dataset(
        num_train=2700,
        num_val=300,
        num_test=200,
        seed=42,
    )

    save_extraction_examples(train_examples, data_dir / "extraction_train.jsonl")
    save_extraction_examples(val_examples, data_dir / "extraction_val.jsonl")
    save_extraction_examples(test_examples, data_dir / "extraction_test.jsonl")

    logger.info("saved_extraction_datasets")


if __name__ == "__main__":
    main()
