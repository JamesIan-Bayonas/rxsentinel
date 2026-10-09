import json

from sqlalchemy import delete, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from rxsentinel.database.config import engine_for
from rxsentinel.database.tables import (
    appearance_products,
    appearances,
    collection_reviews,
    image_assets,
    ingredients,
    product_ingredients,
    products,
    reference_checks,
    source_snapshots,
)
from rxsentinel.schemas import Product


def upsert(connection, table, values: dict, key: str):
    if connection.dialect.name in ("mysql", "mariadb"):
        statement = mysql_insert(table).values(**values)
        statement = statement.on_duplicate_key_update(
            **{field: statement.inserted[field] for field in values if field != key}
        )
    else:
        statement = sqlite_insert(table).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[key],
            set_={field: statement.excluded[field] for field in values if field != key},
        )
    connection.execute(statement)


class RelationalStore:
    def __init__(self, url: str):
        self.engine = engine_for(url)

    def import_products(self, records: list[Product], manifest: dict | None = None):
        # Validate all records before starting writes; all DML is one transaction.
        if not records or len({p.product_id for p in records}) != len(records):
            raise ValueError("An import requires nonempty, unique products")
        with self.engine.begin() as connection:
            for record in records:
                upsert(
                    connection,
                    products,
                    {"product_id": record.product_id, "payload": record.model_dump_json()},
                    "product_id",
                )
                connection.execute(
                    delete(product_ingredients).where(
                        product_ingredients.c.product_id == record.product_id
                    )
                )
                for ordinal, ingredient in enumerate(record.ingredients):
                    if ingredient.rxcui:
                        upsert(
                            connection,
                            ingredients,
                            {
                                "rxcui": ingredient.rxcui,
                                "normalized_name": ingredient.normalized_name,
                            },
                            "rxcui",
                        )
                    connection.execute(
                        product_ingredients.insert().values(
                            product_id=record.product_id,
                            ordinal=ordinal,
                            rxcui=ingredient.rxcui,
                            source_name=ingredient.source_name,
                            strength=ingredient.strength,
                        )
                    )
            if manifest is not None:
                upsert(
                    connection,
                    source_snapshots,
                    {
                        "snapshot_id": manifest["snapshot_id"],
                        "manifest": json.dumps(manifest, ensure_ascii=False),
                    },
                    "snapshot_id",
                )

    def all(self) -> list[Product]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(products.c.payload).order_by(products.c.product_id))
            return [Product.model_validate_json(row[0]) for row in rows]

    def put_manifest(self, manifest: dict):
        with self.engine.begin() as connection:
            upsert(
                connection,
                source_snapshots,
                {
                    "snapshot_id": manifest["snapshot_id"],
                    "manifest": json.dumps(manifest, ensure_ascii=False),
                },
                "snapshot_id",
            )

    def get(self, product_id: str) -> Product | None:
        with self.engine.connect() as connection:
            payload = connection.scalar(
                select(products.c.payload).where(products.c.product_id == product_id)
            )
        return Product.model_validate_json(payload) if payload is not None else None

    def put_appearance(self, appearance):
        self.import_appearances([appearance])

    def import_appearances(self, records, manifest: dict | None = None):
        if len({record.appearance_id for record in records}) != len(records):
            raise ValueError("Appearance IDs must be unique in one import")
        with self.engine.begin() as connection:
            for appearance in records:
                # FK constraints enforce that every linked product exists.
                upsert(
                    connection,
                    appearances,
                    {
                        "appearance_id": appearance.appearance_id,
                        "payload": appearance.model_dump_json(),
                    },
                    "appearance_id",
                )
                connection.execute(
                    delete(appearance_products).where(
                        appearance_products.c.appearance_id == appearance.appearance_id
                    )
                )
                for product_id in appearance.product_ids:
                    connection.execute(
                        appearance_products.insert().values(
                            appearance_id=appearance.appearance_id, product_id=product_id
                        )
                    )
            if manifest is not None:
                upsert(
                    connection,
                    source_snapshots,
                    {
                        "snapshot_id": manifest["snapshot_id"],
                        "manifest": json.dumps(manifest, ensure_ascii=False),
                    },
                    "snapshot_id",
                )

    def put_asset(self, asset):
        with self.engine.begin() as connection:
            upsert(
                connection,
                image_assets,
                {
                    "asset_id": asset.asset_id,
                    "appearance_id": asset.appearance_id,
                    "payload": asset.model_dump_json(),
                },
                "asset_id",
            )

    def appearances(self):
        from rxsentinel.schemas import Appearance

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(appearances.c.payload).order_by(appearances.c.appearance_id)
            )
            return [Appearance.model_validate_json(row[0]) for row in rows]

    def assets(self):
        from rxsentinel.schemas import ImageAsset

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(image_assets.c.payload).order_by(image_assets.c.asset_id)
            )
            return [ImageAsset.model_validate_json(row[0]) for row in rows]

    def import_reference_checks(self, records, assets, manifest):
        if len({r.check_id for r in records}) != len(records):
            raise ValueError("Reference check IDs must be unique")
        with self.engine.begin() as connection:
            for record in records:
                upsert(
                    connection,
                    reference_checks,
                    {
                        "check_id": record.check_id,
                        "appearance_id": record.appearance_id,
                        "product_id": record.product_id,
                        "payload": record.model_dump_json(),
                    },
                    "check_id",
                )
            for asset in assets:
                # Refreshes must not replace a review someone has already recorded.
                existing = connection.scalar(
                    select(image_assets.c.asset_id).where(image_assets.c.asset_id == asset.asset_id)
                )
                if existing is None:
                    connection.execute(
                        image_assets.insert().values(
                            asset_id=asset.asset_id,
                            appearance_id=asset.appearance_id,
                            payload=asset.model_dump_json(),
                        )
                    )
            upsert(
                connection,
                source_snapshots,
                {
                    "snapshot_id": manifest["snapshot_id"],
                    "manifest": json.dumps(manifest),
                },
                "snapshot_id",
            )

    def reference_checks(self):
        from rxsentinel.schemas import ReferenceCheck

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(reference_checks.c.payload).order_by(reference_checks.c.check_id)
            )
            return [ReferenceCheck.model_validate_json(row[0]) for row in rows]

    def collection_reviews(self):
        from rxsentinel.schemas import CollectionReview

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(collection_reviews.c.payload).order_by(collection_reviews.c.review_id)
            )
            return [CollectionReview.model_validate_json(row[0]) for row in rows]

    def collection_transaction(self):
        # All intake/review writers share this row lock, including partition checks.
        # SQLite test transactions reserve the writer before reads.
        from sqlalchemy import text

        connection = self.engine.connect()
        try:
            if self.engine.dialect.name == "sqlite":
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                connection.begin()
                connection.execute(text("SELECT version_num FROM alembic_version FOR UPDATE"))
        except Exception:
            connection.close()
            raise
        return connection
