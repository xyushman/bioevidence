#!/usr/bin/env python
"""Create sample dataset for demos and CI.

This script creates a small sample dataset (~500 trials, 2k abstracts, 200 labels)
that can be used for quick demos and CI testing without full ingestion.
"""

import json
import random
from pathlib import Path

import typer
from rich.console import Console

from bioevidence.db.models import (
    Chunk,
    DrugLabel,
    OutcomeKind,
    PubMedArticle,
    Trial,
    TrialCondition,
    TrialIntervention,
    TrialOutcome,
)
from bioevidence.db.session import get_session

app = typer.Typer()
console = Console()


def create_sample_trials(session, n: int = 500) -> list[Trial]:
    """Create sample trials."""
    phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
    statuses = ["Completed", "Active, not recruiting", "Recruiting", "Terminated"]
    conditions = [
        ("Non-Small Cell Lung Cancer", "Neoplasms"),
        ("Breast Cancer", "Neoplasms"),
        ("Rheumatoid Arthritis", "Arthritis, Rheumatoid"),
        ("Psoriasis", "Psoriasis"),
        ("Heart Failure", "Cardiovascular Diseases"),
        ("Atrial Fibrillation", "Cardiovascular Diseases"),
        ("Osteoporosis", "Osteoporosis"),
        ("Metastatic Melanoma", "Neoplasms"),
    ]
    interventions = [
        "pembrolizumab",
        "nivolumab",
        "atezolizumab",
        "durvalumab",
        "adalimumab",
        "etanercept",
        "infliximab",
        "ustekinumab",
        "sacubitril/valsartan",
        "empagliflozin",
        "dapagliflozin",
        "denosumab",
        "romosozumab",
        "alendronate",
    ]

    trials = []
    for i in range(n):
        nct_id = f"NCT{10000000 + i:05d}"
        cond_name, cond_mesh = random.choice(conditions)
        phase = random.choice(phases)
        status = random.choice(statuses)
        intervention = random.choice(interventions)

        trial = Trial(
            nct_id=nct_id,
            title=f"Study of {intervention} in {cond_name}",
            brief_summary=f"This study evaluates {intervention} for the treatment of {cond_name}.",
            detailed_description=f"Detailed description of the study evaluating {intervention} in patients with {cond_name}.",
            phase=phase,
            overall_status=status,
            study_type="Interventional",
            enrollment=random.randint(50, 2000),
            start_date=f"20{random.randint(5, 23):02d}-{random.randint(1, 12):02d}-01",
            sponsor="Sample Sponsor Inc.",
            source_url=f"https://clinicaltrials.gov/study/{nct_id}",
        )
        session.add(trial)

        # Add condition
        condition = TrialCondition(
            nct_id=nct_id,
            condition=cond_name,
            condition_mesh=cond_mesh,
        )
        session.add(condition)

        # Add intervention
        intervention_obj = TrialIntervention(
            nct_id=nct_id,
            name=intervention,
            type="Drug",
            description=f"{intervention} administered intravenously",
        )
        session.add(intervention_obj)

        # Add outcome
        outcome = TrialOutcome(
            nct_id=nct_id,
            kind=OutcomeKind.PRIMARY,
            measure="Overall Survival"
            if "Cancer" in cond_name or "Melanoma" in cond_name
            else "Progression-Free Survival",
            time_frame="24 months",
        )
        session.add(outcome)

        trials.append(trial)

    session.commit()
    return trials


def create_sample_pubmed(session, n: int = 2000) -> list[PubMedArticle]:
    """Create sample PubMed articles."""
    journals = ["NEJM", "Lancet", "JAMA", "Nature Medicine", "JCO", "Blood"]
    conditions = [
        "lung cancer",
        "breast cancer",
        "rheumatoid arthritis",
        "psoriasis",
        "heart failure",
    ]

    articles = []
    for i in range(n):
        pmid = f"{10000000 + i}"
        condition = random.choice(conditions)

        article = PubMedArticle(
            pmid=pmid,
            title=f"Efficacy of Novel Therapy in {condition.title()}",
            abstract=f"Background: {condition.title()} is a serious condition. Methods: We conducted a randomized trial. Results: The treatment showed significant benefit. Conclusion: This therapy is effective.",
            journal=random.choice(journals),
            year=random.randint(2015, 2024),
            mesh_terms=["Humans", condition.title().replace(" ", ""), "Treatment Outcome"],
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        )
        session.add(article)
        articles.append(article)

    session.commit()
    return articles


def create_sample_labels(session, n: int = 200) -> list[DrugLabel]:
    """Create sample drug labels."""
    drugs = [
        ("Keytruda", "pembrolizumab"),
        ("Opdivo", "nivolumab"),
        ("Tecentriq", "atezolizumab"),
        ("Imfinzi", "durvalumab"),
        ("Humira", "adalimumab"),
        ("Enbrel", "etanercept"),
        ("Remicade", "infliximab"),
        ("Stelara", "ustekinumab"),
        ("Entresto", "sacubitril/valsartan"),
        ("Jardiance", "empagliflozin"),
        ("Farxiga", "dapagliflozin"),
        ("Prolia", "denosumab"),
        ("Evenity", "romosozumab"),
        ("Fosamax", "alendronate"),
    ]

    sections = {
        "indications_and_usage": "INDICATIONS AND USAGE\n{brand} ({generic}) is indicated for the treatment of...",
        "dosage_and_administration": "DOSAGE AND ADMINISTRATION\nThe recommended dose of {brand} is...",
        "contraindications": "CONTRAINDICATIONS\n{brand} is contraindicated in patients with...",
        "warnings_and_cautions": "WARNINGS AND PRECAUTIONS\nSerious adverse reactions include...",
        "adverse_reactions": "ADVERSE REACTIONS\nThe most common adverse reactions are...",
        "drug_interactions": "DRUG INTERACTIONS\nNo formal drug interaction studies have been conducted...",
        "clinical_studies": "CLINICAL STUDIES\nThe efficacy of {brand} was demonstrated in...",
    }

    labels = []
    for i in range(n):
        brand, generic = random.choice(drugs)
        set_id = f"sample-{brand.lower()}-{i:03d}"

        label_sections = {}
        for section_name, template in sections.items():
            label_sections[section_name] = template.format(brand=brand, generic=generic)

        label = DrugLabel(
            set_id=set_id,
            brand_names=[brand],
            generic_names=[generic],
            sections=label_sections,
            effective_date=f"20{random.randint(15, 23):02d}-{random.randint(1, 12):02d}-01",
            source_url=f"https://api.fda.gov/drug/label.json?search=set_id:{set_id}",
        )
        session.add(label)
        labels.append(label)

    session.commit()
    return labels


def create_sample_chunks(session) -> int:
    """Create sample chunks from existing documents."""
    from bioevidence.chunking import build_chunks

    stats = build_chunks(batch_size=1000)
    return stats["total"]


def create_sample_splits(session) -> dict:
    """Create sample train/dev/test splits."""
    from bioevidence.chunking import make_splits

    splits = make_splits()
    return splits


@app.command()
def create_sample(
    trials: int = typer.Option(500, "--trials", help="Number of sample trials"),
    pubmed: int = typer.Option(2000, "--pubmed", help="Number of sample PubMed articles"),
    labels: int = typer.Option(200, "--labels", help="Number of sample drug labels"),
    output_dir: str = typer.Option(
        "data/sample", "--output-dir", help="Output directory for sample data"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for deterministic samples"),
):
    """Create sample dataset for demos and CI."""
    console.print("[yellow]Creating sample dataset...[/yellow]")
    random.seed(seed)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with get_session() as session:
        # Clear existing data
        console.print("Clearing existing data...")
        session.query(Chunk).delete()
        session.query(TrialOutcome).delete()
        session.query(TrialIntervention).delete()
        session.query(TrialCondition).delete()
        session.query(Trial).delete()
        session.query(PubMedArticle).delete()
        session.query(DrugLabel).delete()
        session.commit()

        # Create sample data
        console.print(f"Creating {trials} trials...")
        create_sample_trials(session, trials)

        console.print(f"Creating {pubmed} PubMed articles...")
        create_sample_pubmed(session, pubmed)

        console.print(f"Creating {labels} drug labels...")
        create_sample_labels(session, labels)

        console.print("Building chunks...")
        chunk_stats = create_sample_chunks(session)
        console.print(f"  Total chunks: {chunk_stats}")

        console.print("Creating splits...")
        splits = create_sample_splits(session)
        console.print(
            f"  Trials: {len(splits['trials']['train'])} train, {len(splits['trials']['dev'])} dev, {len(splits['trials']['test'])} test"
        )
        console.print(
            f"  PubMed: {len(splits['pubmed']['train'])} train, {len(splits['pubmed']['dev'])} dev, {len(splits['pubmed']['test'])} test"
        )
        console.print(
            f"  Labels: {len(splits['labels']['train'])} train, {len(splits['labels']['dev'])} dev, {len(splits['labels']['test'])} test"
        )

    console.print(f"[green]Sample dataset created in {output_dir}[/green]")


@app.command()
def export_sample(
    output_dir: str = typer.Option("data/sample", "--output-dir", help="Output directory"),
    format: str = typer.Option("json", "--format", help="Export format: json, jsonl"),
):
    """Export sample data to files."""
    console.print(f"[yellow]Exporting sample data to {output_dir}...[/yellow]")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with get_session() as session:
        # Export trials
        trials = session.query(Trial).all()
        trials_data = []
        for t in trials:
            trials_data.append(
                {
                    "nct_id": t.nct_id,
                    "title": t.title,
                    "brief_summary": t.brief_summary,
                    "phase": t.phase,
                    "overall_status": t.overall_status,
                    "start_date": t.start_date.isoformat() if t.start_date else None,
                }
            )

        if format == "json":
            (Path(output_dir) / "trials.json").write_text(json.dumps(trials_data, indent=2))
        else:
            with open(Path(output_dir) / "trials.jsonl", "w") as f:
                for t in trials_data:
                    f.write(json.dumps(t) + "\n")

        # Export PubMed
        articles = session.query(PubMedArticle).all()
        pubmed_data = []
        for a in articles:
            pubmed_data.append(
                {
                    "pmid": a.pmid,
                    "title": a.title,
                    "abstract": a.abstract,
                    "year": a.year,
                }
            )

        if format == "json":
            (Path(output_dir) / "pubmed.json").write_text(json.dumps(pubmed_data, indent=2))
        else:
            with open(Path(output_dir) / "pubmed.jsonl", "w") as f:
                for a in pubmed_data:
                    f.write(json.dumps(a) + "\n")

        # Export labels
        labels = session.query(DrugLabel).all()
        labels_data = []
        for label in labels:
            labels_data.append(
                {
                    "set_id": label.set_id,
                    "brand_names": label.brand_names,
                    "generic_names": label.generic_names,
                    "sections": label.sections,
                }
            )

        if format == "json":
            (Path(output_dir) / "labels.json").write_text(json.dumps(labels_data, indent=2))
        else:
            with open(Path(output_dir) / "labels.jsonl", "w") as f:
                for label_data in labels_data:
                    f.write(json.dumps(label_data) + "\n")

    console.print(f"[green]Sample data exported to {output_dir}[/green]")


if __name__ == "__main__":
    app()
