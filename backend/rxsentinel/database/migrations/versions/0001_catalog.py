"""Catalog, normalized ingredients, source snapshots, appearances and image assets.

Revision ID: 0001_catalog
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_catalog"
down_revision = None
branch_labels = None
depends_on = None

OPTIONS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_bin",
    "mariadb_engine": "InnoDB",
    "mariadb_charset": "utf8mb4",
    "mariadb_collate": "utf8mb4_bin",
}


def upgrade():
    op.create_table(
        "products",
        sa.Column("product_id", sa.String(160), primary_key=True),
        sa.Column("payload", sa.Text(), nullable=False),
        **OPTIONS,
    )
    op.create_table(
        "ingredients",
        sa.Column("rxcui", sa.String(20), primary_key=True),
        sa.Column("normalized_name", sa.String(255), nullable=False),
        **OPTIONS,
    )
    op.create_table(
        "product_ingredients",
        sa.Column(
            "product_id", sa.String(160), sa.ForeignKey("products.product_id"), primary_key=True
        ),
        sa.Column("ordinal", sa.Integer(), primary_key=True),
        sa.Column("rxcui", sa.String(20), sa.ForeignKey("ingredients.rxcui")),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("strength", sa.Text()),
        **OPTIONS,
    )
    op.create_table(
        "source_snapshots",
        sa.Column("snapshot_id", sa.String(100), primary_key=True),
        sa.Column("manifest", sa.Text(), nullable=False),
        **OPTIONS,
    )
    op.create_table(
        "appearances",
        sa.Column("appearance_id", sa.String(160), primary_key=True),
        sa.Column("payload", sa.Text(), nullable=False),
        **OPTIONS,
    )
    op.create_table(
        "appearance_products",
        sa.Column(
            "appearance_id",
            sa.String(160),
            sa.ForeignKey("appearances.appearance_id"),
            primary_key=True,
        ),
        sa.Column(
            "product_id", sa.String(160), sa.ForeignKey("products.product_id"), primary_key=True
        ),
        **OPTIONS,
    )
    op.create_table(
        "image_assets",
        sa.Column("asset_id", sa.String(160), primary_key=True),
        sa.Column(
            "appearance_id",
            sa.String(160),
            sa.ForeignKey("appearances.appearance_id"),
            nullable=False,
        ),
        sa.Column("payload", sa.Text(), nullable=False),
        **OPTIONS,
    )


def downgrade():
    # Available to Alembic tooling, never run by the application at startup.
    for table in (
        "image_assets",
        "appearance_products",
        "appearances",
        "source_snapshots",
        "product_ingredients",
        "ingredients",
        "products",
    ):
        op.drop_table(table)
