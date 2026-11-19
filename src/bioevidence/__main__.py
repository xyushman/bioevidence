"""Main CLI entry point for BioEvidence."""

import typer
from rich.console import Console

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
):
    """Run data quality checks."""
    console.print("[yellow]Running quality checks...[/yellow]")
    # TODO: Implement quality checks
    raise typer.Exit(code=1)


@app.command()
def chunk(
    action: str = typer.Argument("build", help="Action: build"),
):
    """Build chunks from documents."""
    console.print("[yellow]Building chunks...[/yellow]")
    # TODO: Implement chunking
    raise typer.Exit(code=1)


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
    action: str = typer.Argument("make", help="Action: make"),
):
    """Create train/dev/test splits."""
    console.print("[yellow]Creating splits...[/yellow]")
    # TODO: Implement splits
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
