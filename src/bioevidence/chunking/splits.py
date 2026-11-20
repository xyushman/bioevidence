"""Deterministic train/dev/test splits keyed by document ID."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import select

from bioevidence.config import get_config
from bioevidence.db.models import Chunk, DocType, SplitType
from bioevidence.db.session import get_session

logger = structlog.get_logger(__name__)
DEFAULT_SPLITS_PATH = Path("data/splits/doc_splits.json")
_DOC_TYPES = ("trials", "pubmed", "labels")
_SPLITS = ("train", "dev", "test")
# Keep each UPDATE below common PostgreSQL parameter limits when splitting large corpora.
_SPLIT_UPDATE_BATCH_SIZE = 1_000


@dataclass
class SplitConfig:
    """Configuration for splits."""

    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    def validate(self) -> None:
        """Validate ratios before assigning documents."""
        if any(ratio < 0 for ratio in (self.train_ratio, self.dev_ratio, self.test_ratio)):
            msg = "Split ratios must be non-negative"
            raise ValueError(msg)
        if abs(self.train_ratio + self.dev_ratio + self.test_ratio - 1.0) > 1e-6:
            msg = "Split ratios must sum to 1.0"
            raise ValueError(msg)


def resolve_splits_path(splits_path: Path | None = None) -> Path:
    """Return an explicit path or the repository-standard split file."""
    if splits_path is not None:
        return splits_path
    config = get_config()
    configured = Path(config.llm.cache_dir).parent.parent / "splits" / "doc_splits.json"
    return configured if configured.exists() else DEFAULT_SPLITS_PATH


def _batched(values: list[str], size: int = _SPLIT_UPDATE_BATCH_SIZE) -> Iterator[list[str]]:
    """Yield bounded slices of document IDs for safe parameterized updates."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def split_doc_ids(doc_ids: list[str], config: SplitConfig, salt: str) -> dict[str, list[str]]:
    """Shuffle and split document IDs deterministically."""
    config.validate()
    rng = random.Random(f"{config.seed}:{salt}")
    ordered = sorted(set(doc_ids))
    rng.shuffle(ordered)
    n_train = int(len(ordered) * config.train_ratio)
    n_dev = int(len(ordered) * config.dev_ratio)
    return {
        "train": ordered[:n_train],
        "dev": ordered[n_train : n_train + n_dev],
        "test": ordered[n_train + n_dev :],
    }


def make_splits(
    config: SplitConfig | None = None,
    splits_path: Path | None = None,
) -> dict[str, Any]:
    """Create and persist train/dev/test splits by document ID (not chunk)."""
    config = config or SplitConfig()
    config.validate()
    path = resolve_splits_path(splits_path)

    doc_ids: dict[str, list[str]] = {}
    with get_session() as session:
        for name, doc_type in zip(
            _DOC_TYPES, (DocType.TRIAL, DocType.PUBMED, DocType.LABEL), strict=True
        ):
            doc_ids[name] = list(
                session.execute(
                    select(Chunk.doc_id)
                    .where(Chunk.doc_type == doc_type)
                    .distinct()
                    .order_by(Chunk.doc_id)
                )
                .scalars()
                .all()
            )

        logger.info(
            "split_doc_counts",
            trials=len(doc_ids["trials"]),
            pubmed=len(doc_ids["pubmed"]),
            labels=len(doc_ids["labels"]),
        )

        for name, doc_type in zip(
            _DOC_TYPES, (DocType.TRIAL, DocType.PUBMED, DocType.LABEL), strict=True
        ):
            assignment = split_doc_ids(doc_ids[name], config, salt=name)
            for split_name, ids in assignment.items():
                for batch in _batched(ids):
                    session.query(Chunk).filter(
                        Chunk.doc_type == doc_type,
                        Chunk.doc_id.in_(batch),
                    ).update({Chunk.split: SplitType(split_name)}, synchronize_session=False)
        session.commit()

    splits_data: dict[str, Any] = {
        "seed": config.seed,
        "ratios": {
            "train": config.train_ratio,
            "dev": config.dev_ratio,
            "test": config.test_ratio,
        },
    }
    for name in _DOC_TYPES:
        splits_data[name] = split_doc_ids(doc_ids[name], config, salt=name)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(splits_data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    logger.info("splits_saved", path=str(path))
    return splits_data


def _validate_split_file(splits: dict[str, Any]) -> bool:
    """Ensure the persisted split file assigns each document to one split."""
    for doc_type in _DOC_TYPES:
        groups = splits.get(doc_type)
        if not isinstance(groups, dict):
            return False
        sets = {name: set(groups.get(name, [])) for name in _SPLITS}
        if any(sets[left] & sets[right] for left in _SPLITS for right in _SPLITS if left < right):
            return False
    return True


def verify_no_db_leakage() -> bool:
    """Verify each database document ID appears in exactly one split."""
    assigned: dict[tuple[DocType, str], str] = {}
    with get_session() as session:
        rows = session.execute(select(Chunk.doc_type, Chunk.doc_id, Chunk.split).distinct()).all()
    for doc_type, doc_id, split in rows:
        key = (doc_type, str(doc_id))
        name = split.value if hasattr(split, "value") else str(split)
        if key in assigned and assigned[key] != name:
            logger.error("db_split_leakage_detected", doc_type=doc_type, doc_id=doc_id)
            return False
        assigned[key] = name
    logger.info("db_no_leakage_verified", documents=len(assigned))
    return True


def verify_no_leakage(splits_path: Path | None = None, check_db: bool = True) -> bool:
    """Verify no train/dev/test leakage in the split file and database."""
    path = resolve_splits_path(splits_path)
    if not path.exists():
        logger.error("splits_file_not_found", path=str(path))
        return False
    try:
        splits = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("splits_file_unreadable", path=str(path))
        return False
    if not isinstance(splits, dict) or not _validate_split_file(splits):
        logger.error("split_leakage_detected", path=str(path))
        return False
    if check_db and not verify_no_db_leakage():
        return False
    logger.info("no_leakage_verified", path=str(path))
    return True


def get_split_stats(splits_path: Path | None = None) -> dict[str, Any]:
    """Get statistics about persisted splits."""
    path = resolve_splits_path(splits_path)
    if not path.exists():
        return {}
    try:
        splits = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(splits, dict):
        return {}

    stats: dict[str, Any] = {"seed": splits.get("seed"), "ratios": splits.get("ratios", {})}
    for doc_type in _DOC_TYPES:
        groups = splits.get(doc_type, {})
        stats[doc_type] = {name: len(groups.get(name, [])) for name in _SPLITS}
        stats[doc_type]["total"] = sum(stats[doc_type].values())
    return stats
