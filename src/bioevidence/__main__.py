"""Main CLI entry point for BioEvidence."""

from pathlib import Path

import typer
from rich.console import Console

from bioevidence.chunking import build_chunks, get_split_stats, make_splits, verify_no_leakage
from bioevidence.config import get_config
from bioevidence.ingest import (
    ingest_ctgov as ingest_ctgov_fn,
)
from bioevidence.ingest import (
    ingest_openfda as ingest_openfda_fn,
)
from bioevidence.ingest import (
    ingest_pubmed as ingest_pubmed_fn,
)
from bioevidence.quality import run_quality_checks
from bioevidence.scripts.create_sample_data import create_sample, export_sample

app = typer.Typer(
    name="bioevidence",
    help="BioEvidence: Biomedical Evidence Copilot",
    add_completion=False,
)
console = Console()


@app.callback()
def callback():
    """BioEvidence CLI."""
    config = get_config()
    console.print(
        f"[green]BioEvidence v0.1.0[/green] - Provider: [bold]{config.llm.provider}[/bold]"
    )


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Source to ingest: ctgov, pubmed, openfda, or all"),
    max_pages: int = typer.Option(None, "--max-pages", help="Max pages for CT.gov"),
    max_articles: int = typer.Option(None, "--max-articles", help="Max articles for PubMed"),
    max_labels: int = typer.Option(None, "--max-labels", help="Max labels for openFDA"),
    rate_limit: float = typer.Option(3.0, "--rate-limit", help="Requests per second"),
    start_year: int = typer.Option(2005, "--start-year", help="Minimum start year"),
    use_bulk: bool = typer.Option(False, "--bulk", help="Use bulk download for openFDA"),
):
    """Ingest data from a source."""
    console.print(f"[yellow]Ingesting from {source}...[/yellow]")

    if source in ("ctgov", "all"):
        console.print("[blue]ClinicalTrials.gov...[/blue]")
        stats = ingest_ctgov_fn(
            max_pages=max_pages, rate_limit=rate_limit, start_date_min=f"{start_year}-01-01"
        )
        console.print(
            f"  Fetched: {stats.fetched}, Inserted: {stats.inserted}, Updated: {stats.updated}, Errors: {stats.errors}"
        )

    if source in ("pubmed", "all"):
        console.print("[blue]PubMed...[/blue]")
        stats = ingest_pubmed_fn(
            max_articles=max_articles, rate_limit=rate_limit, start_year=start_year
        )
        console.print(
            f"  Fetched: {stats.fetched}, Inserted: {stats.inserted}, Updated: {stats.updated}, Errors: {stats.errors}"
        )

    if source in ("openfda", "all"):
        console.print("[blue]openFDA...[/blue]")
        stats = ingest_openfda_fn(max_labels=max_labels, rate_limit=rate_limit, use_bulk=use_bulk)
        console.print(
            f"  Fetched: {stats.fetched}, Inserted: {stats.inserted}, Updated: {stats.updated}, Errors: {stats.errors}"
        )


@app.command()
def ingest_ctgov(
    max_pages: int = typer.Option(None, "--max-pages", help="Max pages to fetch"),
    rate_limit: float = typer.Option(3.0, "--rate-limit", help="Requests per second"),
    start_year: int = typer.Option(2005, "--start-year", help="Minimum start year"),
):
    """Ingest from ClinicalTrials.gov only."""
    console.print("[yellow]Ingesting from ClinicalTrials.gov...[/yellow]")
    stats = ingest_ctgov(
        max_pages=max_pages, rate_limit=rate_limit, start_date_min=f"{start_year}-01-01"
    )
    console.print(
        f"Fetched: {stats.fetched}, Inserted: {stats.inserted}, Updated: {stats.updated}, Errors: {stats.errors}"
    )


@app.command()
def ingest_pubmed(
    max_articles: int = typer.Option(None, "--max-articles", help="Max articles to fetch"),
    rate_limit: float = typer.Option(3.0, "--rate-limit", help="Requests per second"),
    start_year: int = typer.Option(2005, "--start-year", help="Minimum publication year"),
):
    """Ingest from PubMed only."""
    console.print("[yellow]Ingesting from PubMed...[/yellow]")
    stats = ingest_pubmed(max_articles=max_articles, rate_limit=rate_limit, start_year=start_year)
    console.print(
        f"Fetched: {stats.fetched}, Inserted: {stats.inserted}, Updated: {stats.updated}, Errors: {stats.errors}"
    )


@app.command()
def ingest_openfda(
    max_labels: int = typer.Option(None, "--max-labels", help="Max labels to fetch"),
    rate_limit: float = typer.Option(10.0, "--rate-limit", help="Requests per second"),
    use_bulk: bool = typer.Option(False, "--bulk", help="Use bulk download"),
):
    """Ingest from openFDA only."""
    console.print("[yellow]Ingesting from openFDA...[/yellow]")
    stats = ingest_openfda(max_labels=max_labels, rate_limit=rate_limit, use_bulk=use_bulk)
    console.print(
        f"Fetched: {stats.fetched}, Inserted: {stats.inserted}, Updated: {stats.updated}, Errors: {stats.errors}"
    )


@app.command()
def quality(
    action: str = typer.Argument("run", help="Action: run"),
    output: str = typer.Option(
        "reports/data_quality.md", "--output", help="Output path for report"
    ),
):
    """Run data quality checks."""
    console.print("[yellow]Running quality checks...[/yellow]")
    report = run_quality_checks(Path(output))

    console.print(f"Checks: {len(report.checks)} total")
    console.print(f"Passed: {sum(1 for c in report.checks if c.passed)}")
    console.print(f"Failed (critical): {report.critical_failures}")
    console.print(f"Warnings: {report.warnings}")

    if report.critical_failures > 0:
        console.print("[red]Quality checks FAILED[/red]")
        raise typer.Exit(code=1)
    else:
        console.print("[green]Quality checks PASSED[/green]")


@app.command()
def chunk(
    action: str = typer.Argument("build", help="Action: build"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens per chunk"),
    overlap: int = typer.Option(32, "--overlap", help="Token overlap between chunks"),
    trial_overlap: int = typer.Option(
        40, "--trial-overlap", help="Token overlap for trial sections"
    ),
    label_overlap: int = typer.Option(
        40, "--label-overlap", help="Token overlap for label sections"
    ),
    model: str = typer.Option(None, "--model", help="Embedding model for token counting"),
    batch_size: int = typer.Option(1000, "--batch-size", help="Batch size for processing"),
):
    """Build chunks from documents."""
    console.print("[yellow]Building chunks...[/yellow]")
    stats = build_chunks(
        max_tokens=max_tokens,
        overlap=overlap,
        trial_section_overlap=trial_overlap,
        label_section_overlap=label_overlap,
        model_name=model,
        batch_size=batch_size,
    )
    console.print(f"Total chunks: {stats['total']}")
    console.print(
        f"  Trials: {stats['trials']}, PubMed: {stats['pubmed']}, Labels: {stats['labels']}"
    )


@app.command()
def embed(
    action: str = typer.Argument("build", help="Action: build"),
    model: str = typer.Option(None, "--model", help="Embedding model to use"),
):
    """Build embeddings for chunks."""
    console.print("[yellow]Building embeddings...[/yellow]")
    # TODO: Implement embedding
    raise typer.Exit(code=1)


@app.command()
def train(
    component: str = typer.Argument(..., help="Component to train: embed, rerank, lora"),
):
    """Train a model component."""
    console.print(f"[yellow]Training {component}...[/yellow]")
    # TODO: Implement training
    raise typer.Exit(code=1)


@app.command()
def eval(
    mode: str = typer.Argument("run", help="Mode: run, ci"),
):
    """Run evaluation."""
    console.print(f"[yellow]Running evaluation ({mode})...[/yellow]")
    # TODO: Implement evaluation
    raise typer.Exit(code=1)


@app.command()
def forecast(
    action: str = typer.Argument("run", help="Action: run"),
):
    """Run forecasting pipeline."""
    console.print("[yellow]Running forecasting...[/yellow]")
    # TODO: Implement forecasting
    raise typer.Exit(code=1)


@app.command()
def db(
    action: str = typer.Argument("migrate", help="Action: migrate"),
):
    """Database operations."""
    console.print(f"[yellow]Running db {action}...[/yellow]")
    # TODO: Implement DB operations
    raise typer.Exit(code=1)


@app.command()
def splits(
    action: str = typer.Argument("make", help="Action: make, verify, stats"),
    train_ratio: float = typer.Option(0.8, "--train-ratio", help="Train split ratio"),
    dev_ratio: float = typer.Option(0.1, "--dev-ratio", help="Dev split ratio"),
    test_ratio: float = typer.Option(0.1, "--test-ratio", help="Test split ratio"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Create train/dev/test splits."""
    if action == "make":
        console.print("[yellow]Creating splits...[/yellow]")
        splits = make_splits(
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        console.print("Splits created:")
        console.print(
            f"  Trials - Train: {len(splits['trials']['train'])}, Dev: {len(splits['trials']['dev'])}, Test: {len(splits['trials']['test'])}"
        )
        console.print(
            f"  PubMed - Train: {len(splits['pubmed']['train'])}, Dev: {len(splits['pubmed']['dev'])}, Test: {len(splits['pubmed']['test'])}"
        )
        console.print(
            f"  Labels - Train: {len(splits['labels']['train'])}, Dev: {len(splits['labels']['dev'])}, Test: {len(splits['labels']['test'])}"
        )

    elif action == "verify":
        console.print("[yellow]Verifying no leakage...[/yellow]")
        ok = verify_no_leakage()
        if ok:
            console.print("[green]No leakage detected[/green]")
        else:
            console.print("[red]Leakage detected![/red]")
            raise typer.Exit(code=1)

    elif action == "stats":
        console.print("[yellow]Split statistics...[/yellow]")
        stats = get_split_stats()
        console.print_json(data=stats)

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(code=1)


@app.command()
def sample(
    action: str = typer.Argument("create", help="Action: create, export"),
    trials: int = typer.Option(500, "--trials", help="Number of sample trials"),
    pubmed: int = typer.Option(2000, "--pubmed", help="Number of sample PubMed articles"),
    labels: int = typer.Option(200, "--labels", help="Number of sample drug labels"),
    output_dir: str = typer.Option("data/sample", "--output-dir", help="Output directory"),
    format: str = typer.Option("json", "--format", help="Export format: json, jsonl"),
):
    """Create or export sample dataset for demos/CI."""
    if action == "create":
        create_sample(trials=trials, pubmed=pubmed, labels=labels, output_dir=output_dir)
    elif action == "export":
        export_sample(output_dir=output_dir, format=format)
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
