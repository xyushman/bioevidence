"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_trial_data():
    """Sample trial data for testing."""
    return {
        "nct_id": "NCT01234567",
        "title": "Test Trial",
        "brief_summary": "A test trial for testing.",
        "detailed_description": "Detailed description of the test trial.",
        "phase": "Phase 3",
        "overall_status": "Completed",
        "start_date": "2020-01-01",
        "completion_date": "2022-12-31",
    }


@pytest.fixture
def sample_pubmed_data():
    """Sample PubMed data for testing."""
    return {
        "pmid": "12345678",
        "title": "Test Article",
        "abstract": "This is a test abstract.",
        "journal": "Test Journal",
        "year": 2023,
        "mesh_terms": ["Neoplasms", "Humans"],
    }


@pytest.fixture
def sample_label_data():
    """Sample drug label data for testing."""
    return {
        "set_id": "test-set-id",
        "brand_names": ["TestBrand"],
        "generic_names": ["testgeneric"],
        "sections": {
            "indications_and_usage": "For testing purposes.",
            "contraindications": "Do not use if allergic.",
        },
        "effective_date": "2023-01-01",
    }
