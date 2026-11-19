"""Golden set module for BioEvidence (file-based, no DB required)."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GoldenQuery:
    """A golden set query with ground truth."""

    query: str
    gold_chunk_ids: list[int]
    style: str  # "keyword", "natural", "clinician"
    doc_type: str  # "trial", "pubmed", "label"
    category: str | None = None  # For QA set
    gold_answer: str | None = None
    gold_citations: list[int] | None = None
    answerable: bool | None = None
    notes: str | None = None


def load_sample_data(data_dir: Path) -> dict[str, list[dict]]:
    """Load sample data from JSON files."""
    data = {}
    for name in ["trials", "pubmed", "labels"]:
        path = data_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                data[name] = json.load(f)
        else:
            data[name] = []
    return data


def generate_synthetic_queries(
    sample_data: dict[str, list[dict]],
    num_queries: int = 300,
    doc_types: list[str] | None = None,
    styles: list[str] | None = None,
    seed: int = 42,
) -> list[GoldenQuery]:
    """Generate synthetic queries from sample data using templates."""
    random.seed(seed)

    doc_types = doc_types or ["trial", "pubmed", "label"]
    styles = styles or ["keyword", "natural", "clinician"]

    queries = []

    # Distribute queries across doc types
    per_type = num_queries // len(doc_types)

    for doc_type in doc_types:
        items = (
            sample_data.get(doc_type + "s", [])
            if doc_type != "label"
            else sample_data.get("labels", [])
        )
        if not items:
            continue

        random.shuffle(items)

        # Generate multiple queries per item by cycling through styles
        queries_per_item = max(1, per_type // len(items))
        for item_idx, item in enumerate(items):
            if len(queries) >= num_queries:
                break

            for q_idx in range(queries_per_item):
                if len(queries) >= num_queries:
                    break

                style = styles[(item_idx * queries_per_item + q_idx) % len(styles)]

                if style == "keyword":
                    query = _generate_keyword_query(item, doc_type, q_idx)
                elif style == "natural":
                    query = _generate_natural_query(item, doc_type, q_idx)
                elif style == "clinician":
                    query = _generate_clinician_query(item, doc_type, q_idx)
                else:
                    query = _generate_keyword_query(item, doc_type, q_idx)

                # Use composite ID as pseudo chunk_id
                queries.append(
                    GoldenQuery(
                        query=query,
                        gold_chunk_ids=[item_idx * 100 + q_idx],
                        style=style,
                        doc_type=doc_type,
                    )
                )

    return queries


def _generate_keyword_query(item: dict, doc_type: str, variant: int = 0) -> str:
    """Generate keyword-style query."""
    keywords = []

    if doc_type == "trial":
        if "phase" in item:
            keywords.append(item["phase"].lower())
        if "title" in item:
            words = item["title"].split()
            for w in words:
                if w.lower() in [
                    "cancer",
                    "tumor",
                    "arthritis",
                    "psoriasis",
                    "heart",
                    "failure",
                    "osteoporosis",
                ]:
                    keywords.append(w.lower())
    elif doc_type == "pubmed":
        if "title" in item:
            words = item["title"].split()
            keywords.extend([w for w in words if len(w) > 4][:3])
    elif doc_type == "label":
        if "brand_names" in item and item["brand_names"]:
            keywords.extend(item["brand_names"][:2])
        if "generic_names" in item and item["generic_names"]:
            keywords.extend(item["generic_names"][:2])

    # Add variant-specific terms
    variant_terms = ["treatment", "efficacy", "safety", "outcome", "dosage"]
    if variant < len(variant_terms):
        keywords.append(variant_terms[variant])

    return " ".join(keywords[:5]) if keywords else "clinical study"


def _generate_natural_query(item: dict, doc_type: str, variant: int = 0) -> str:
    """Generate natural language query."""
    templates = {
        "trial": [
            "What were the primary outcomes of the {} trial?",
            "What is the eligibility criteria for {}?",
            "What interventions were tested in {}?",
        ],
        "pubmed": [
            "What did the study find about {}?",
            "What are the results of the {} study?",
        ],
        "label": [
            "What are the contraindications for {}?",
            "What is the dosage for {}?",
            "What are the adverse reactions of {}?",
        ],
    }

    template = random.choice(templates.get(doc_type, ["What is {}?"]))
    topic = (
        item.get("title", "this study")
        if doc_type != "label"
        else item.get("brand_names", ["this drug"])[0]
    )
    return template.format(topic)


def _generate_clinician_query(item: dict, doc_type: str, variant: int = 0) -> str:
    """Generate clinician-phrased query."""
    templates = {
        "trial": [
            "In patients with {}, what is the efficacy of {}?",
            "What are the inclusion criteria for the {} trial?",
        ],
        "pubmed": [
            "What is the evidence for {} in {}?",
        ],
        "label": [
            "What are the boxed warnings for {}?",
            "How should {} be dosed in {}?",
        ],
    }

    template = random.choice(templates.get(doc_type, ["What is the clinical evidence for {}?"]))
    topic = (
        item.get("title", "this condition")
        if doc_type != "label"
        else item.get("brand_names", ["this drug"])[0]
    )
    return template.format(topic, "treatment")


def save_golden_set(queries: list[GoldenQuery], path: Path) -> None:
    """Save golden set to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for q in queries:
            f.write(json.dumps(asdict(q)) + "\n")


def load_golden_set(path: Path) -> list[GoldenQuery]:
    """Load golden set from JSONL file."""
    queries = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            queries.append(GoldenQuery(**data))
    return queries


def create_qa_golden_set(num_questions: int = 100) -> list[GoldenQuery]:
    """Create QA golden set with categories.

    Categories:
    - 30 factual lookup (label sections, e.g. contraindications)
    - 25 trial-structured (answerable via SQL, e.g. counts/filters)
    - 20 multi-hop / comparison (two trials or trial + label)
    - 20 unanswerable (no supporting evidence in corpus)
    - 5 adversarial (asks for medical advice, prompt injection)
    """
    queries = []

    # Factual lookup (30)
    for i in range(30):
        queries.append(
            GoldenQuery(
                query=f"Factual lookup question {i}",
                gold_chunk_ids=[],
                style="factual",
                doc_type="label",
                category="factual_lookup",
                gold_answer=f"Answer for factual question {i}",
                gold_citations=[],
                answerable=True,
            )
        )

    # Trial-structured (25)
    for i in range(25):
        queries.append(
            GoldenQuery(
                query=f"Trial structured question {i}",
                gold_chunk_ids=[],
                style="sql",
                doc_type="trial",
                category="trial_structured",
                gold_answer=f"Answer for trial question {i}",
                gold_citations=[],
                answerable=True,
            )
        )

    # Multi-hop (20)
    for i in range(20):
        queries.append(
            GoldenQuery(
                query=f"Multi-hop question {i}",
                gold_chunk_ids=[],
                style="multi_hop",
                doc_type="mixed",
                category="multi_hop",
                gold_answer=f"Answer for multi-hop question {i}",
                gold_citations=[],
                answerable=True,
            )
        )

    # Unanswerable (20)
    for i in range(20):
        queries.append(
            GoldenQuery(
                query=f"Unanswerable question {i}",
                gold_chunk_ids=[],
                style="unanswerable",
                doc_type="none",
                category="unanswerable",
                gold_answer="Insufficient evidence in the retrieved sources.",
                gold_citations=[],
                answerable=False,
            )
        )

    # Adversarial (5)
    for i in range(5):
        queries.append(
            GoldenQuery(
                query=f"Adversarial question {i}",
                gold_chunk_ids=[],
                style="adversarial",
                doc_type="none",
                category="adversarial",
                gold_answer="Insufficient evidence in the retrieved sources.",
                gold_citations=[],
                answerable=False,
            )
        )

    return queries
