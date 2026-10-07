"""Append-only local collection review history."""

import sqlalchemy as sa
from alembic import op

revision = "0003_collection_reviews"
down_revision = "0002_reference_checks"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "collection_reviews",
        sa.Column("review_id", sa.String(100), primary_key=True),
        sa.Column(
            "appearance_id",
            sa.String(160),
            sa.ForeignKey("appearances.appearance_id"),
            nullable=False,
        ),
        sa.Column("payload", sa.Text(), nullable=False),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_bin",
        mariadb_engine="InnoDB",
        mariadb_charset="utf8mb4",
        mariadb_collate="utf8mb4_bin",
    )


def downgrade():
    op.drop_table("collection_reviews")
