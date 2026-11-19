"""Main CLI entry point for BioEvidence."""

import typer
from rich.console import Console

from bioevidence.config import get_config

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
):
    """Ingest data from a source."""
    console.print(f"[yellow]Ingesting from {source}...[/yellow]")
    # TODO: Implement ingestion
    raise typer.Exit(code=1)


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
