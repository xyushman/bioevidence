"""Embedding fine-tuning for BioEvidence."""

import json
import logging
import os
import structlog
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

from bioevidence.config import get_config
from bioevidence.scripts.generate_training_pairs import load_pairs, Pair

logger = structlog.get_logger(__name__)


@dataclass
class TrainConfig:
    """Configuration for embedding fine-tuning."""
    base_model: str = "BAAI/bge-small-en-v1.5"
    output_dir: str = "models/bge-small-ft"
    max_seq_length: int = 512
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    fp16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    metric_for_best_model: str = "eval_recall_at_5"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True


def load_training_data(train_path: Path, dev_path: Path | None = None) -> tuple[list[InputExample], list[InputExample] | None]:
    """Load training pairs and convert to InputExamples."""
    train_pairs = load_pairs(train_path)

    train_examples = []
    for pair in train_pairs:
        # Positive pair
        train_examples.append(InputExample(texts=[pair.query, pair.positive], label=1.0))

        # Hard negative pairs
        for neg in pair.hard_negatives:
            train_examples.append(InputExample(texts=[pair.query, neg], label=0.0))

    logger.info("loaded_train_examples", count=len(train_examples))

    dev_examples = None
    if dev_path and dev_path.exists():
        dev_pairs = load_pairs(dev_path)
        dev_examples = []
        for pair in dev_pairs:
            dev_examples.append(InputExample(texts=[pair.query, pair.positive], label=1.0))
            for neg in pair.hard_negatives:
                dev_examples.append(InputExample(texts=[pair.query, neg], label=0.0))
        logger.info("loaded_dev_examples", count=len(dev_examples))

    return train_examples, dev_examples


def create_model(config: TrainConfig) -> SentenceTransformer:
    """Create sentence transformer model with custom pooling."""
    # Load base model
    word_embedding_model = models.Transformer(
        config.base_model,
        max_seq_length=config.max_seq_length,
    )

    # Mean pooling
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


def train_embeddings(
    config: TrainConfig | None = None,
    train_path: Path | None = None,
    dev_path: Path | None = None,
    output_dir: Path | None = None,
) -> SentenceTransformer:
    """Fine-tune embeddings using MultipleNegativesRankingLoss."""
    config = config or TrainConfig()

    # Setup paths
    bio_config = get_config()
    data_dir = Path(bio_config.llm.cache_dir).parent.parent / "processed"
    train_path = train_path or data_dir / "synth_pairs.jsonl"
    dev_path = dev_path or data_dir / "synth_pairs_dev.jsonl"
    output_dir = output_dir or Path(config.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_examples, dev_examples = load_training_data(train_path, dev_path)

    # Create model
    model = create_model(config)
    model.max_seq_length = config.max_seq_length

    # Loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)

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
        # For IR evaluator, we need queries, corpus, and relevant_docs
        # Simplified: just use the dev examples for validation
        evaluator = InformationRetrievalEvaluator(
            queries={str(i): ex.texts[0] for i, ex in enumerate(dev_examples[:100])},
            corpus={str(i): ex.texts[1] for i, ex in enumerate(dev_examples[:100])},
            relevant_docs={str(i): {str(i)} for i in range(min(100, len(dev_examples)))},
            name="dev",
            show_progress_bar=True,
        )

    # Training arguments
    warmup_steps = int(len(train_dataloader) * config.num_epochs * config.warmup_ratio)

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config.num_epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        optimizer_params={"lr": config.learning_rate, "eps": 1e-8, "weight_decay": 0.01},
        use_amp=config.fp16,
        show_progress_bar=True,
        checkpoint_path=str(output_dir / "checkpoints") if config.save_steps > 0 else None,
        checkpoint_save_steps=config.save_steps,
        checkpoint_save_total_limit=config.save_total_limit,
    )

    # Save final model
    model.save(str(output_dir))
    logger.info("model_saved", path=str(output_dir))

    return model


def main():
    """Main entry point for embedding fine-tuning."""
    config = TrainConfig()

    # Enable MLflow logging if available
    try:
        import mlflow
        mlflow.set_experiment("bioevidence-embed")
        mlflow.start_run()
        mlflow.log_params(config.__dict__)
    except ImportError:
        pass

    train_embeddings(config)

    try:
        import mlflow
        mlflow.end_run()
    except ImportError:
        pass


if __name__ == "__main__":
    main()