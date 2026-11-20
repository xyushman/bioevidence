"""Main CLI entry point for BioEvidence."""

from pathlib import Path

import typer
from rich.console import Console

from bioevidence.agent import BioEvidenceAgent
from bioevidence.chunking import build_chunks, get_split_stats, make_splits, verify_no_leakage
from bioevidence.chunking.splits import SplitConfig
from bioevidence.config import get_config
from bioevidence.embedding import build_embeddings
from bioevidence.extraction import prepare_extraction_dataset
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
    max_studies: int = typer.Option(None, "--max-studies", help="Max studies for CT.gov"),
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
            max_pages=max_pages,
            max_studies=max_studies,
            rate_limit=rate_limit,
            start_date_min=f"{start_year}-01-01",
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
    max_studies: int = typer.Option(None, "--max-studies", help="Max studies to fetch"),
    rate_limit: float = typer.Option(3.0, "--rate-limit", help="Requests per second"),
    start_year: int = typer.Option(2005, "--start-year", help="Minimum start year"),
):
    """Ingest from ClinicalTrials.gov only."""
    console.print("[yellow]Ingesting from ClinicalTrials.gov...[/yellow]")
    stats = ingest_ctgov_fn(
        max_pages=max_pages,
        max_studies=max_studies,
        rate_limit=rate_limit,
        start_date_min=f"{start_year}-01-01",
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
    stats = ingest_pubmed_fn(
        max_articles=max_articles, rate_limit=rate_limit, start_year=start_year
    )
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
    stats = ingest_openfda_fn(max_labels=max_labels, rate_limit=rate_limit, use_bulk=use_bulk)
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
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size"),
    finetuned: bool = typer.Option(False, "--finetuned", help="Use fine-tuned column"),
):
    """Build embeddings for chunks."""
    console.print("[yellow]Building embeddings...[/yellow]")
    stats = build_embeddings(model=model, batch_size=batch_size, use_finetuned=finetuned)
    console.print(f"Embeddings built: {stats}")


@app.command()
def train(
    component: str = typer.Argument(..., help="Component to train: embed, rerank, lora"),
    epochs: int = typer.Option(3, "--epochs", help="Number of epochs"),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size"),
    output_dir: str = typer.Option(None, "--output-dir", help="Output directory"),
):
    """Train a model component."""
    console.print(f"[yellow]Training {component}...[/yellow]")

    if component == "embed":
        from scripts.train_embeddings import TrainConfig, train_embeddings

        config = TrainConfig(num_epochs=epochs, batch_size=batch_size)
        if output_dir:
            config.output_dir = output_dir
        train_embeddings(config)
    elif component == "rerank":
        from scripts.train_reranker import RerankTrainConfig, train_reranker

        config = RerankTrainConfig(num_epochs=epochs, batch_size=batch_size)
        if output_dir:
            config.output_dir = output_dir
        train_reranker(config)
    elif component == "lora":
        from scripts.train_lora import LoRATrainConfig, train_lora

        config = LoRATrainConfig(num_epochs=epochs, batch_size=batch_size)
        if output_dir:
            config.output_dir = output_dir
        train_lora(config)
    else:
        console.print(f"[red]Unknown component: {component}[/red]")
        raise typer.Exit(code=1)

    console.print("[green]Training complete[/green]")


@app.command()
def eval(
    mode: str = typer.Argument("run", help="Mode: run, ci"),
    systems: str = typer.Option(
        "A0,A1a,A1b,A2,A3,A4,A4b,A5", "--systems", help="Comma-separated systems"
    ),
    retrieval_golden: str = typer.Option(
        None, "--retrieval-golden", help="Retrieval golden set path"
    ),
    qa_golden: str = typer.Option(None, "--qa-golden", help="QA golden set path"),
    output_dir: str = typer.Option(None, "--output-dir", help="Output directory"),
    baseline: str = typer.Option(None, "--baseline", help="Baseline path for regression"),
):
    """Run evaluation."""
    if mode == "ci":
        console.print("[yellow]Running eval CI gate...[/yellow]")
        from scripts.eval_ci import run_eval_ci

        system_list = [s.strip() for s in systems.split(",")]
        results = run_eval_ci(
            systems=system_list,
            retrieval_golden=Path(retrieval_golden) if retrieval_golden else None,
            qa_golden=Path(qa_golden) if qa_golden else None,
            output_dir=Path(output_dir) if output_dir else None,
            baseline_path=Path(baseline) if baseline else None,
        )

        all_passed = True
        for name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            console.print(f"  {name}: {status}")
            if result.failures:
                for f in result.failures:
                    console.print(f"    - {f}")
            if not result.passed:
                all_passed = False

        if not all_passed:
            console.print("\n[red]Eval CI gate FAILED[/red]")
            raise typer.Exit(code=1)
        else:
            console.print("\n[green]All eval gates PASSED[/green]")
    else:
        console.print(f"[yellow]Running evaluation ({mode})...[/yellow]")
        # TODO: Implement full evaluation
        raise typer.Exit(code=1)


@app.command()
def agent(
    question: str = typer.Argument(..., help="Question to ask the agent"),
    system: str = typer.Option("A5", "--system", help="Retrieval system: A0-A5"),
):
    """Ask a question to the BioEvidence agent."""
    console.print(f"[yellow]Asking agent (system={system})...[/yellow]")
    agent = BioEvidenceAgent(system=system)
    response = agent.answer(question)

    console.print("\n[bold]Answer:[/bold]")
    console.print(response.answer)

    if response.citations:
        console.print(f"\n[dim]Citations: {response.citations}[/dim]")
    if response.tools_used:
        console.print(f"[dim]Tools used: {response.tools_used}[/dim]")
    if response.sql_queries:
        console.print(f"[dim]SQL: {response.sql_queries}[/dim]")


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
    if action != "migrate":
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(code=1)

    console.print("[yellow]Applying database migrations...[/yellow]")
    from alembic import command
    from alembic.config import Config as AlembicConfig

    alembic_config = AlembicConfig("alembic.ini")
    command.upgrade(alembic_config, "head")
    console.print("[green]Database schema is up to date[/green]")


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
            config=SplitConfig(
                train_ratio=train_ratio,
                dev_ratio=dev_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
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


@app.command(name="eval-retrieval")
def eval_retrieval(
    system: str = typer.Argument(..., help="System to evaluate: A0, A1a, A1b, A2, A3, A4, A4b, A5"),
    golden_file: str = typer.Option(
        "data/goldens/retrieval_golden.jsonl", "--golden", help="Golden set file"
    ),
    output: str = typer.Option("results/retrieval", "--output", help="Output directory"),
    k: int = typer.Option(5, "--k", help="Top-k for retrieval"),
):
    """Evaluate a retrieval system against golden set."""
    console.print(f"[yellow]Evaluating retrieval system {system}...[/yellow]")

    # Load golden set
    from bioevidence.eval import load_golden_set

    queries = load_golden_set(Path(golden_file))
    console.print(f"Loaded {len(queries)} golden queries")

    # Get retriever
    from bioevidence.retrieval.pipeline import get_retriever

    retriever = get_retriever(system)

    # Evaluate
    from bioevidence.eval import evaluate_retrieval

    result = evaluate_retrieval(queries, retriever, ks=[1, 5, 10, 20])

    # Save results
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    import json

    result_file = output_path / f"{system}_results.json"
    result_file.write_text(
        json.dumps(
            {
                "system": system,
                "aggregate": {
                    "recall_at_1": result["aggregate"].recall_at_1,
                    "recall_at_5": result["aggregate"].recall_at_5,
                    "recall_at_10": result["aggregate"].recall_at_10,
                    "recall_at_20": result["aggregate"].recall_at_20,
                    "mrr_at_10": result["aggregate"].mrr_at_10,
                    "ndcg_at_10": result["aggregate"].ndcg_at_10,
                    "hit_at_5": result["aggregate"].hit_at_5,
                },
                "cis": result["cis"],
                "n_queries": result["n_queries"],
            },
            indent=2,
        )
    )

    console.print(f"Results saved to {result_file}")
    console.print(f"Recall@5: {result['aggregate'].recall_at_5:.4f}")
    console.print(f"MRR@10: {result['aggregate'].mrr_at_10:.4f}")
    console.print(f"nDCG@10: {result['aggregate'].ndcg_at_10:.4f}")


@app.command()
def extract(
    action: str = typer.Argument("prepare", help="Action: prepare, evaluate, run"),
    nct_id: str = typer.Option(None, "--nct-id", help="NCT ID for single trial extraction"),
    method: str = typer.Option("lora", "--method", help="Extraction method: lora, constrained"),
    model_path: str = typer.Option(None, "--model-path", help="LoRA model path"),
    test_file: str = typer.Option(
        "data/processed/extraction_test.jsonl", "--test-file", help="Test file for evaluation"
    ),
    output: str = typer.Option("results/extraction", "--output", help="Output directory"),
):
    """Trial field extraction."""
    if action == "prepare":
        console.print("[yellow]Preparing extraction dataset...[/yellow]")
        train, val, test = prepare_extraction_dataset()
        console.print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    elif action == "evaluate":
        console.print("[yellow]Evaluating extraction...[/yellow]")
        console.print("[red]Evaluation requires trained model - not implemented yet[/red]")
        raise typer.Exit(code=1)

    elif action == "run":
        if not nct_id:
            console.print("[red]--nct-id required for run action[/red]")
            raise typer.Exit(code=1)

        console.print(f"[yellow]Extracting fields for {nct_id}...[/yellow]")
        from bioevidence.db.models import Trial, TrialCondition, TrialIntervention, TrialOutcome
        from bioevidence.db.session import get_session

        with get_session() as session:
            trial = session.get(Trial, nct_id)
            if not trial:
                console.print(f"[red]Trial {nct_id} not found[/red]")
                raise typer.Exit(code=1)

            from bioevidence.scripts.prepare_extraction_data import build_trial_text

            trial_data = {
                "trial": trial,
                "conditions": session.query(TrialCondition)
                .filter(TrialCondition.nct_id == nct_id)
                .all(),
                "interventions": session.query(TrialIntervention)
                .filter(TrialIntervention.nct_id == nct_id)
                .all(),
                "outcomes": session.query(TrialOutcome).filter(TrialOutcome.nct_id == nct_id).all(),
            }
            trial_text = build_trial_text(trial_data)

        from bioevidence.extraction import extract_trial_fields

        result = extract_trial_fields(trial_text, method=method, model_path=model_path)

        console.print("\n[bold]Extracted Fields:[/bold]")
        console.print_json(data=result)

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
    seed: int = typer.Option(42, "--seed", help="Random seed for deterministic samples"),
):
    """Create or export sample dataset for demos/CI."""
    if action == "create":
        create_sample(
            trials=trials,
            pubmed=pubmed,
            labels=labels,
            output_dir=output_dir,
            seed=seed,
        )
    elif action == "export":
        export_sample(output_dir=output_dir, format=format)
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
