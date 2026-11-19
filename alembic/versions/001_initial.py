"""Initial migration: create all tables."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSON, TSVECTOR
from pgvector.sqlalchemy import Vector

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create enums
    op.execute("CREATE TYPE doctype AS ENUM ('trial', 'pubmed', 'label')")
    op.execute("CREATE TYPE outcomekind AS ENUM ('primary', 'secondary')")
    op.execute("CREATE TYPE splittype AS ENUM ('train', 'dev', 'test')")

    # Trials table
    op.create_table(
        'trials',
        sa.Column('nct_id', sa.String(20), primary_key=True),
        sa.Column('title', sa.Text, nullable=False),
        sa.Column('brief_summary', sa.Text),
        sa.Column('detailed_description', sa.Text),
        sa.Column('eligibility_text', sa.Text),
        sa.Column('phase', sa.String(50)),
        sa.Column('overall_status', sa.String(50)),
        sa.Column('study_type', sa.String(50)),
        sa.Column('allocation', sa.String(50)),
        sa.Column('masking', sa.String(100)),
        sa.Column('enrollment', sa.Integer),
        sa.Column('enrollment_type', sa.String(50)),
        sa.Column('start_date', sa.Date),
        sa.Column('completion_date', sa.Date),
        sa.Column('sponsor', sa.Text),
        sa.Column('sponsor_class', sa.String(50)),
        sa.Column('min_age_years', sa.Integer),
        sa.Column('max_age_years', sa.Integer),
        sa.Column('sex', sa.String(20)),
        sa.Column('countries', ARRAY(sa.Text)),
        sa.Column('num_arms', sa.Integer),
        sa.Column('source_url', sa.Text, nullable=False),
        sa.Column('fetched_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('date_imputed', sa.Boolean, server_default=sa.false()),
    )

    # Trial conditions
    op.create_table(
        'trial_conditions',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('nct_id', sa.String(20), sa.ForeignKey('trials.nct_id', ondelete='CASCADE'), nullable=False),
        sa.Column('condition', sa.Text, nullable=False),
        sa.Column('condition_mesh', sa.Text),
    )

    # Trial interventions
    op.create_table(
        'trial_interventions',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('nct_id', sa.String(20), sa.ForeignKey('trials.nct_id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.Text, nullable=False),
        sa.Column('type', sa.String(50)),
        sa.Column('description', sa.Text),
    )

    # Trial outcomes
    op.create_table(
        'trial_outcomes',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('nct_id', sa.String(20), sa.ForeignKey('trials.nct_id', ondelete='CASCADE'), nullable=False),
        sa.Column('kind', sa.Enum('primary', 'secondary', name='outcomekind'), nullable=False),
        sa.Column('measure', sa.Text, nullable=False),
        sa.Column('time_frame', sa.Text),
        sa.Column('description', sa.Text),
    )

    # PubMed articles
    op.create_table(
        'pubmed_articles',
        sa.Column('pmid', sa.String(20), primary_key=True),
        sa.Column('title', sa.Text, nullable=False),
        sa.Column('abstract', sa.Text),
        sa.Column('journal', sa.Text),
        sa.Column('year', sa.Integer),
        sa.Column('mesh_terms', ARRAY(sa.Text)),
        sa.Column('source_url', sa.Text, nullable=False),
        sa.Column('fetched_at', sa.DateTime, server_default=sa.func.now()),
    )

    # Drug labels
    op.create_table(
        'drug_labels',
        sa.Column('set_id', sa.String(100), primary_key=True),
        sa.Column('brand_names', ARRAY(sa.Text)),
        sa.Column('generic_names', ARRAY(sa.Text)),
        sa.Column('sections', JSON, server_default='{}'),
        sa.Column('effective_date', sa.Date),
        sa.Column('source_url', sa.Text, nullable=False),
    )

    # Chunks
    op.create_table(
        'chunks',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('doc_type', sa.Enum('trial', 'pubmed', 'label', name='doctype'), nullable=False),
        sa.Column('doc_id', sa.Text, nullable=False),
        sa.Column('section', sa.Text),
        sa.Column('text', sa.Text, nullable=False),
        sa.Column('token_count', sa.Integer, nullable=False),
        sa.Column('metadata', JSON, server_default='{}'),
        sa.Column('embedding', Vector(384)),
        sa.Column('embedding_ft', Vector(384)),
        sa.Column('tsv', TSVECTOR),
        sa.Column('split', sa.Enum('train', 'dev', 'test', name='splittype'), nullable=False, server_default='train'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

    # Indexes
    op.create_index('ix_trials_phase', 'trials', ['phase'])
    op.create_index('ix_trials_start_date', 'trials', ['start_date'])
    op.create_index('ix_trials_overall_status', 'trials', ['overall_status'])
    op.create_index('ix_trial_conditions_nct_id', 'trial_conditions', ['nct_id'])
    op.create_index('ix_trial_conditions_mesh', 'trial_conditions', ['condition_mesh'])
    op.create_index('ix_trial_interventions_nct_id', 'trial_interventions', ['nct_id'])
    op.create_index('ix_trial_interventions_name', 'trial_interventions', ['name'])
    op.create_index('ix_trial_outcomes_nct_id', 'trial_outcomes', ['nct_id'])
    op.create_index('ix_trial_outcomes_kind', 'trial_outcomes', ['kind'])
    op.create_index('ix_pubmed_year', 'pubmed_articles', ['year'])
    op.create_index('ix_drug_labels_effective_date', 'drug_labels', ['effective_date'])
    op.create_index('ix_chunks_doc_type', 'chunks', ['doc_type'])
    op.create_index('ix_chunks_doc_id', 'chunks', ['doc_id'])
    op.create_index('ix_chunks_split', 'chunks', ['split'])

    # GIN index for tsvector
    op.execute('CREATE INDEX ix_chunks_tsv ON chunks USING GIN (tsv)')

    # HNSW indexes for embeddings (created after data is loaded)
    # op.execute('CREATE INDEX ix_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)')
    # op.execute('CREATE INDEX ix_chunks_embedding_ft ON chunks USING hnsw (embedding_ft vector_cosine_ops)')


def downgrade():
    op.drop_table('chunks')
    op.drop_table('drug_labels')
    op.drop_table('pubmed_articles')
    op.drop_table('trial_outcomes')
    op.drop_table('trial_interventions')
    op.drop_table('trial_conditions')
    op.drop_table('trials')
    op.execute("DROP TYPE splittype")
    op.execute("DROP TYPE outcomekind")
    op.execute("DROP TYPE doctype")