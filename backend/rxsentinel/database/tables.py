from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

metadata = MetaData(
    naming_convention={
        "pk": "pk_%(table_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    }
)
OPTIONS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_bin",
    "mariadb_engine": "InnoDB",
    "mariadb_charset": "utf8mb4",
    "mariadb_collate": "utf8mb4_bin",
}

products = Table(
    "products",
    metadata,
    Column("product_id", String(160), primary_key=True),
    Column("payload", Text, nullable=False),
    **OPTIONS,
)
ingredients = Table(
    "ingredients",
    metadata,
    Column("rxcui", String(20), primary_key=True),
    Column("normalized_name", String(255), nullable=False),
    **OPTIONS,
)
product_ingredients = Table(
    "product_ingredients",
    metadata,
    Column("product_id", String(160), ForeignKey("products.product_id"), primary_key=True),
    Column("ordinal", Integer, primary_key=True),
    Column("rxcui", String(20), ForeignKey("ingredients.rxcui"), nullable=True),
    Column("source_name", Text, nullable=False),
    Column("strength", Text, nullable=True),
    **OPTIONS,
)
source_snapshots = Table(
    "source_snapshots",
    metadata,
    Column("snapshot_id", String(100), primary_key=True),
    Column("manifest", Text, nullable=False),
    **OPTIONS,
)
appearances = Table(
    "appearances",
    metadata,
    Column("appearance_id", String(160), primary_key=True),
    Column("payload", Text, nullable=False),
    **OPTIONS,
)
appearance_products = Table(
    "appearance_products",
    metadata,
    Column("appearance_id", String(160), ForeignKey("appearances.appearance_id"), primary_key=True),
    Column("product_id", String(160), ForeignKey("products.product_id"), primary_key=True),
    **OPTIONS,
)
image_assets = Table(
    "image_assets",
    metadata,
    Column("asset_id", String(160), primary_key=True),
    Column("appearance_id", String(160), ForeignKey("appearances.appearance_id"), nullable=False),
    Column("payload", Text, nullable=False),
    **OPTIONS,
)

reference_checks = Table(
    "reference_checks",
    metadata,
    Column("check_id", String(160), primary_key=True),
    Column("appearance_id", String(160), ForeignKey("appearances.appearance_id"), nullable=False),
    Column("product_id", String(160), ForeignKey("products.product_id"), nullable=False),
    Column("payload", Text, nullable=False),
    **OPTIONS,
)

collection_reviews = Table(
    "collection_reviews",
    metadata,
    Column("review_id", String(100), primary_key=True),
    Column("appearance_id", String(160), ForeignKey("appearances.appearance_id"), nullable=False),
    Column("payload", Text, nullable=False),
    **OPTIONS,
)
