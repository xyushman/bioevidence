"""Add intervention links and fetch metadata to drug labels."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add columns used to prove intervention linkage and provenance."""
    op.add_column(
        "drug_labels",
        sa.Column("matched_interventions", sa.ARRAY(sa.Text()), nullable=True),
    )
    op.add_column(
        "drug_labels",
        sa.Column("fetched_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_drug_labels_fetched_at", "drug_labels", ["fetched_at"])


def downgrade() -> None:
    """Remove the added columns."""
    op.drop_index("ix_drug_labels_fetched_at", table_name="drug_labels")
    op.drop_column("drug_labels", "fetched_at")
    op.drop_column("drug_labels", "matched_interventions")
