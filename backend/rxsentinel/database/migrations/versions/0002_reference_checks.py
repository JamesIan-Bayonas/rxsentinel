"""Persist current-label evidence separately from identity review."""

import sqlalchemy as sa
from alembic import op

revision = "0002_reference_checks"
down_revision = "0001_catalog"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "reference_checks",
        sa.Column("check_id", sa.String(160), primary_key=True),
        sa.Column(
            "appearance_id",
            sa.String(160),
            sa.ForeignKey("appearances.appearance_id"),
            nullable=False,
        ),
        sa.Column(
            "product_id", sa.String(160), sa.ForeignKey("products.product_id"), nullable=False
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
    op.drop_table("reference_checks")
