"""Tests for deterministic document splits and leakage checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioevidence.chunking import SplitConfig, split_doc_ids, verify_no_leakage
from bioevidence.chunking.splits import _batched


def test_split_updates_are_batched_for_large_corpora() -> None:
    batches = list(_batched([f"doc{index}" for index in range(2_501)], size=1_000))

    assert [len(batch) for batch in batches] == [1_000, 1_000, 501]
    assert [doc for batch in batches for doc in batch][-1] == "doc2500"


def test_split_doc_ids_is_deterministic_and_disjoint() -> None:
    ids = [f"NCT{index:08d}" for index in range(100)]
    config = SplitConfig(seed=42)

    first = split_doc_ids(ids, config, salt="trials")
    second = split_doc_ids(ids, config, salt="trials")
    other = split_doc_ids(ids, config, salt="pubmed")

    assert first == second
    assert first != other
    assert sum(len(part) for part in first.values()) == 100
    assert set(first["train"]).isdisjoint(first["dev"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["dev"]).isdisjoint(first["test"])


def test_split_config_rejects_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitConfig(train_ratio=0.9, dev_ratio=0.2, test_ratio=0.1).validate()


def test_verify_no_leakage_reads_committed_file(tmp_path: Path) -> None:
    valid = {
        "seed": 42,
        "ratios": {"train": 0.8, "dev": 0.1, "test": 0.1},
        "trials": {"train": ["NCT1"], "dev": ["NCT2"], "test": ["NCT3"]},
        "pubmed": {"train": [], "dev": [], "test": []},
        "labels": {"train": [], "dev": [], "test": []},
    }
    path = tmp_path / "doc_splits.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert verify_no_leakage(path, check_db=False)

    valid["trials"]["test"] = ["NCT2"]
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert not verify_no_leakage(path, check_db=False)
