import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path

import pymysql
from sqlalchemy import inspect, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

from rxsentinel.catalog import Catalog, configured_catalog
from rxsentinel.database.config import database_url, engine_for
from rxsentinel.database.migrate import upgrade_schema
from rxsentinel.database.tables import product_ingredients
from rxsentinel.safety import SafetyEngine
from rxsentinel.schemas import AuditRequest, MedicationEntry, RuleSet


def bootstrap_xampp(port: int = 3306, resume_empty: bool = False) -> dict:
    """Create only the local project database/account; never reset an existing account."""
    config_file = Path(".env")
    if config_file.exists():
        raise ValueError(".env already exists; configure it and use upgrade instead of bootstrap")
    password = secrets.token_urlsafe(36)
    with (
        pymysql.connect(
            host="127.0.0.1",
            port=port,
            user="root",
            password=os.environ.get("RXSENTINEL_DB_ADMIN_PASSWORD", ""),
            connect_timeout=10,
            autocommit=True,
        ) as admin,
        admin.cursor() as cursor,
    ):
        cursor.execute(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'rxsentinel'"
        )
        exists = bool(cursor.fetchone())
        if exists:
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'rxsentinel'"
            )
            if not resume_empty or cursor.fetchone()[0]:
                raise ValueError(
                    "rxsentinel already exists; use --resume-empty only for an "
                    "empty database left by an interrupted bootstrap"
                )
        # XAMPP MariaDB 10.4 stores accounts in global_priv; mysql.user is a view.
        cursor.execute("SELECT User FROM mysql.global_priv WHERE User = 'rxsentinel_app'")
        if cursor.fetchone():
            raise ValueError("rxsentinel_app already exists; no existing account was changed")
        if not exists:
            cursor.execute("CREATE DATABASE rxsentinel CHARACTER SET utf8mb4 COLLATE utf8mb4_bin")
        cursor.execute("CREATE USER 'rxsentinel_app'@'localhost' IDENTIFIED BY %s", (password,))
        cursor.execute(
            "GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX, REFERENCES "
            "ON rxsentinel.* TO 'rxsentinel_app'@'localhost'"
        )
    url = URL.create(
        "mariadb+pymysql",
        username="rxsentinel_app",
        password=password,
        host="127.0.0.1",
        port=port,
        database="rxsentinel",
        query={"charset": "utf8mb4"},
    )
    # Exclusive creation; credentials are never returned or printed.
    with config_file.open("x", encoding="utf-8") as output:
        output.write("RXSENTINEL_DATABASE_URL=" + url.render_as_string(hide_password=False) + "\n")
        output.write("RXSENTINEL_DATA_DIR=data\n")
    return {"database": "rxsentinel", "account": "rxsentinel_app", "config": ".env"}


def migrate_sqlite(source: Path, target_url: str, rules_path: Path) -> dict:
    legacy = Catalog(source)
    records = legacy.all()
    if not records:
        raise ValueError("The source SQLite catalog is empty; no migration was performed")
    manifests = []
    for snapshot_id in sorted({p.snapshot_id for p in records}):
        snapshot_dir = source.parent / "snapshots" / snapshot_id
        if (snapshot_dir / "manifest.json").exists():
            manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
            digest = hashlib.sha256((snapshot_dir / "raw.json").read_bytes()).hexdigest()
            if digest != manifest["raw_sha256"] or manifest["snapshot_id"] != snapshot_id:
                raise ValueError("Source snapshot verification failed; target was not changed")
            manifests.append(manifest)
    target = Catalog(target_url)
    target.import_products(records, manifests[0] if manifests else None)
    for manifest in manifests[1:]:
        target.store.put_manifest(manifest)
    migrated = {record.product_id: record for record in target.all()}
    if any(migrated.get(record.product_id) != record for record in records):
        raise ValueError("Migration verification failed: product records differ")
    # Compare every pair, not just a single demonstration example.
    rules = RuleSet.model_validate_json(rules_path.read_text(encoding="utf-8"))
    before = SafetyEngine(legacy, rules)
    after = SafetyEngine(target, rules)
    from itertools import combinations

    requests = [
        AuditRequest(
            medications=[
                MedicationEntry(
                    entry_id=f"entry-{index}", product_id=record.product_id, identity_confirmed=True
                )
                for index, record in enumerate(pair)
            ]
        )
        for pair in combinations(records, 2)
    ]
    for request in requests:
        if before.audit(request) != after.audit(request):
            raise ValueError("Migration verification failed: medication audit differs")
    result = {
        "products_verified": len(records),
        "pairwise_audits_verified": len(requests),
        "source_retained": str(source),
        "target_backend": target.backend,
        "source_manifests_verified": len(manifests),
    }
    (source.parent / "database-migration.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    target.store.engine.dispose()
    return result


def main():
    parser = argparse.ArgumentParser(description="Manage RxSentinel's project database")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    subcommands = parser.add_subparsers(dest="command", required=True)
    bootstrap = subcommands.add_parser("bootstrap-xampp")
    bootstrap.add_argument("--port", type=int, default=3306)
    bootstrap.add_argument("--resume-empty", action="store_true")
    subcommands.add_parser("upgrade")
    subcommands.add_parser("status")
    migration = subcommands.add_parser("migrate-sqlite")
    migration.add_argument("--source", type=Path, default=Path("data/catalog.sqlite"))
    args = parser.parse_args()
    try:
        if args.command == "bootstrap-xampp":
            result = bootstrap_xampp(args.port, args.resume_empty)
        else:
            url = database_url()
            if not url:
                raise ValueError("Set RXSENTINEL_DATABASE_URL in .env or the environment first")
            if args.command == "upgrade":
                engine = engine_for(url)
                upgrade_schema(engine)
                result = {"schema": "upgraded", "backend": engine.dialect.name}
                engine.dispose()
            elif args.command == "migrate-sqlite":
                result = migrate_sqlite(
                    args.source, url, args.data_dir / "rules" / "interactions.json"
                )
            else:
                catalog = configured_catalog(args.data_dir)
                with catalog.store.engine.connect() as connection:
                    version = connection.scalar(text("SELECT version_num FROM alembic_version"))
                    ingredient_rows = len(connection.execute(select(product_ingredients)).all())
                parsed = make_url(url)
                result = {
                    "backend": catalog.backend,
                    "database": parsed.database,
                    "schema_version": version,
                    "products": len(catalog.all()),
                    "ingredient_links": ingredient_rows,
                    "tables": inspect(catalog.store.engine).get_table_names(),
                }
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    except (pymysql.MySQLError, SQLAlchemyError) as error:
        original = getattr(error, "orig", error)
        code = (
            original.args[0] if original.args and isinstance(original.args[0], int) else "unknown"
        )
        parser.exit(
            1,
            f"Database operation failed (code {code}). Check server, account permissions, and "
            "schema configuration; connection credentials are not displayed.\n",
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
