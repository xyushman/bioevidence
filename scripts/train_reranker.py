"""Reranker fine-tuning for BioEvidence."""

import json
import structlog
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader

from bioevidence.config import get_config
from bioevidence.scripts.generate_training_pairs import load_pairs, Pair

logger = structlog.get_logger(__name__)


@dataclass
class RerankTrainConfig:
    """Configuration for reranker fine-tuning."""
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    output_dir: str = "models/minilm-reranker-ft"
    max_seq_length: int = 512
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    num_epochs: int = 2
    warmup_ratio: float = 0.1
    fp16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2


def load_rerank_data(
    train_path: Path,
    dev_path: Path | None = None,
) -> tuple[list[InputExample], list[InputExample] | None]:
    """Load training pairs for cross-encoder (query + passage -> label)."""
    train_pairs = load_pairs(train_path)

    train_examples = []
    for pair in train_pairs:
        # Positive
        train_examples.append(InputExample(texts=[pair.query, pair.positive], label=1))

        # Hard negatives
        for neg in pair.hard_negatives:
            train_examples.append(InputExample(texts=[pair.query, neg], label=0))

    logger.info("loaded_rerank_train", count=len(train_examples))

    dev_examples = None
    if dev_path and dev_path.exists():
        dev_pairs = load_pairs(dev_path)
        dev_examples = []
        for pair in dev_pairs:
            dev_examples.append(InputExample(texts=[pair.query, pair.positive], label=1))
            for neg in pair.hard_negatives:
                dev_examples.append(InputExample(texts=[pair.query, neg], label=0))
        logger.info("loaded_rerank_dev", count=len(dev_examples))

    return train_examples, dev_examples


def train_reranker(
    config: RerankTrainConfig | None = None,
    train_path: Path | None = None,
    dev_path: Path | None = None,
    output_dir: Path | None = None,
) -> CrossEncoder:
    """Fine-tune cross-encoder reranker."""
    config = config or RerankTrainConfig()

    bio_config = get_config()
    data_dir = Path(bio_config.llm.cache_dir).parent.parent / "processed"
    train_path = train_path or data_dir / "rerank_pairs.jsonl"
    dev_path = dev_path or data_dir / "rerank_pairs_dev.jsonl"
    output_dir = output_dir or Path(config.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_examples, dev_examples = load_rerank_data(train_path, dev_path)

    # Create model
    model = CrossEncoder(
        config.base_model,
        num_labels=1,
        max_length=config.max_seq_length,
        automodel_args={"torch_dtype": torch.float16} if config.fp16 else {},
    )

    # Loss
    train_loss = losses.BCEWithLogitsLoss()

    # Data loader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.batch_size,
        drop_last=True,
    )

    # Evaluator
    evaluator = None
    if dev_examples:
        # Need queries, passages, and labels for CE evaluator
        dev_queries = [ex.texts[0] for ex in dev_examples]
        dev_passages = [ex.texts[1] for ex in dev_examples]
        dev_labels = [float(ex.label) for ex in dev_examples]

        evaluator = CEBinaryClassificationEvaluator(
            dev_queries,
            dev_passages,
            dev_labels,
            name="dev",
            show_progress_bar=True,
        )

    # Training
    warmup_steps = int(len(train_dataloader) * config.num_epochs * config.warmup_ratio)

    model.fit(
        train_dataloader=train_dataloader,
        loss_fct=train_loss,
        evaluator=evaluator,
        epochs=config.num_epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        optimizer_params={"lr": config.learning_rate, "eps": 1e-8, "weight_decay": 0.01},
        use_amp=config.fp16,
        show_progress_bar=True,
        save_best_model=True,
    )

    # Save final model
    model.save(str(output_dir))
    logger.info("reranker_saved", path=str(output_dir))

    return model


def main():
    """Main entry point for reranker fine-tuning."""
    config = RerankTrainConfig()

    try:
        import mlflow
        mlflow.set_experiment("bioevidence-rerank")
        mlflow.start_run()
        mlflow.log_params(config.__dict__)
    except ImportError:
        pass

    train_reranker(config)

    try:
        import mlflow
        mlflow.end_run()
    except ImportError:
        pass


if __name__ == "__main__":
    main()