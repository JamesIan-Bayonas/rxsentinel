import sqlite3
from pathlib import Path

from rxsentinel.schemas import Product


class Catalog:
    """Persist validated product snapshots; no network access during an audit."""

    def __init__(self, path: Path | str):
        self.path = path if isinstance(path, Path) else None
        self.store = None
        if isinstance(path, str):
            from rxsentinel.database.store import RelationalStore

            self.store = RelationalStore(path)

    @property
    def backend(self):
        return self.store.engine.dialect.name if self.store else "sqlite"

    def import_products(self, products: list[Product], manifest: dict | None = None) -> None:
        if self.store:
            self.store.import_products(products, manifest)
            return
        if not products or len({p.product_id for p in products}) != len(products):
            raise ValueError("An import requires nonempty, unique products")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS products "
                "(product_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO products(product_id, payload) VALUES (?, ?) "
                "ON CONFLICT(product_id) DO UPDATE SET payload=excluded.payload",
                [(p.product_id, p.model_dump_json()) for p in products],
            )

    def all(self) -> list[Product]:
        if self.store:
            return self.store.all()
        if not self.path.exists():
            return []
        # Read-only connection prevents an API request from creating a blank database.
        with sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True) as conn:
            rows = conn.execute("SELECT payload FROM products ORDER BY product_id").fetchall()
        return [Product.model_validate_json(row[0]) for row in rows]

    def get(self, product_id: str) -> Product | None:
        if self.store:
            return self.store.get(product_id)
        if not self.path.exists():
            return None
        with sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT payload FROM products WHERE product_id = ?", (product_id,)
            ).fetchone()
        return Product.model_validate_json(row[0]) if row else None


def configured_catalog(data_dir: Path, url: str | None = None) -> Catalog:
    from rxsentinel.database.config import database_url

    selected = url if url is not None else database_url()
    return Catalog(selected or data_dir / "catalog.sqlite")
