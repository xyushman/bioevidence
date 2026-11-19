"""Train/dev/test splits for BioEvidence."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from bioevidence.config import get_config
from bioevidence.db.models import Chunk, DocType, SplitType
from bioevidence.db.session import get_session

logger = structlog.get_logger(__name__)


@dataclass
class SplitConfig:
    """Configuration for splits."""

    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


def make_splits(
    config: SplitConfig | None = None,
    splits_path: Path | None = None,
) -> dict[str, Any]:
    """Create train/dev/test splits by document ID (not chunk)."""
    config = config or SplitConfig()
    config_path = get_config()

    if splits_path is None:
        splits_path = Path(config_path.llm.cache_dir).parent.parent / "splits" / "doc_splits.json"

    # Set seed for reproducibility
    random.seed(config.seed)

    with get_session() as session:
        # Get all unique document IDs by type
        trial_ids = (
            session.query(Chunk.doc_id).filter(Chunk.doc_type == DocType.TRIAL).distinct().all()
        )
        trial_ids = [t[0] for t in trial_ids]

        pubmed_ids = (
            session.query(Chunk.doc_id).filter(Chunk.doc_type == DocType.PUBMED).distinct().all()
        )
        pubmed_ids = [p[0] for p in pubmed_ids]

        label_ids = (
            session.query(Chunk.doc_id).filter(Chunk.doc_type == DocType.LABEL).distinct().all()
        )
        label_ids = [lbl[0] for lbl in label_ids]

        logger.info(
            "split_doc_counts", trials=len(trial_ids), pubmed=len(pubmed_ids), labels=len(label_ids)
        )

        # Shuffle and split each type
        def split_ids(doc_ids: list[str]) -> dict[str, list[str]]:
            random.shuffle(doc_ids)
            n = len(doc_ids)
            n_train = int(n * config.train_ratio)
            n_dev = int(n * config.dev_ratio)

            return {
                "train": doc_ids[:n_train],
                "dev": doc_ids[n_train : n_train + n_dev],
                "test": doc_ids[n_train + n_dev :],
            }

        trial_splits = split_ids(trial_ids)
        pubmed_splits = split_ids(pubmed_ids)
        label_splits = split_ids(label_ids)

        # Update chunks with split assignment
        for split_name, ids in trial_splits.items():
            split_enum = SplitType(split_name)
            session.query(Chunk).filter(
                Chunk.doc_type == DocType.TRIAL,
                Chunk.doc_id.in_(ids),
            ).update({Chunk.split: split_enum}, synchronize_session=False)

        for split_name, ids in pubmed_splits.items():
            split_enum = SplitType(split_name)
            session.query(Chunk).filter(
                Chunk.doc_type == DocType.PUBMED,
                Chunk.doc_id.in_(ids),
            ).update({Chunk.split: split_enum}, synchronize_session=False)

        for split_name, ids in label_splits.items():
            split_enum = SplitType(split_name)
            session.query(Chunk).filter(
                Chunk.doc_type == DocType.LABEL,
                Chunk.doc_id.in_(ids),
            ).update({Chunk.split: split_enum}, synchronize_session=False)

        session.commit()

        # Save splits to file
        splits_data = {
            "seed": config.seed,
            "ratios": {
                "train": config.train_ratio,
                "dev": config.dev_ratio,
                "test": config.test_ratio,
            },
            "trials": trial_splits,
            "pubmed": pubmed_splits,
            "labels": label_splits,
        }

        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps(splits_data, indent=2))

        logger.info("splits_saved", path=str(splits_path))

        return splits_data


def verify_no_leakage(splits_path: Path | None = None) -> bool:
    """Verify no leakage between splits."""
    config_path = get_config()

    if splits_path is None:
        splits_path = Path(config_path.llm.cache_dir).parent.parent / "splits" / "doc_splits.json"

    if not splits_path.exists():
        logger.error("splits_file_not_found", path=str(splits_path))
        return False

    with open(splits_path) as f:
        splits = json.load(f)

    # Check that train/dev/test are disjoint for each doc type
    for doc_type in ["trials", "pubmed", "labels"]:
        train_set = set(splits[doc_type]["train"])
        dev_set = set(splits[doc_type]["dev"])
        test_set = set(splits[doc_type]["test"])

        train_dev = train_set & dev_set
        train_test = train_set & test_set
        dev_test = dev_set & test_set

        if train_dev or train_test or dev_test:
            logger.error(
                "split_leakage_detected",
                doc_type=doc_type,
                train_dev=len(train_dev),
                train_test=len(train_test),
                dev_test=len(dev_test),
            )
            return False

    logger.info("no_leakage_verified")
    return True


def get_split_stats(splits_path: Path | None = None) -> dict[str, Any]:
    """Get statistics about splits."""
    config_path = get_config()

    if splits_path is None:
        splits_path = Path(config_path.llm.cache_dir).parent.parent / "splits" / "doc_splits.json"

    if not splits_path.exists():
        return {}

    with open(splits_path) as f:
        splits = json.load(f)

    stats = {"seed": splits["seed"], "ratios": splits["ratios"]}

    for doc_type in ["trials", "pubmed", "labels"]:
        stats[doc_type] = {
            "train": len(splits[doc_type]["train"]),
            "dev": len(splits[doc_type]["dev"]),
            "test": len(splits[doc_type]["test"]),
            "total": len(splits[doc_type]["train"])
            + len(splits[doc_type]["dev"])
            + len(splits[doc_type]["test"]),
        }

    return stats
