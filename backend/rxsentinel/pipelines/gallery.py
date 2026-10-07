"""Export or validate a versioned reference image gallery."""

import argparse
import json
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from rxsentinel.catalog import Catalog, configured_catalog
from rxsentinel.gallery import export_gallery, validate_gallery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--data-dir", type=Path, default=Path("data"))
    export.add_argument(
        "--synthetic-directory",
        type=Path,
        help="Use an isolated matching-demo SQLite catalog, ignoring production config",
    )
    check = commands.add_parser("validate")
    check.add_argument("--directory", type=Path, required=True)
    check.add_argument("--check-source", action="store_true")
    check.add_argument("--data-dir", type=Path, default=Path("data"))
    check.add_argument("--synthetic-directory", type=Path)
    args = parser.parse_args()
    catalog = None
    try:
        if args.command == "export" or args.check_source:
            if args.synthetic_directory:
                data_dir = args.synthetic_directory.resolve()
                path = data_dir / "catalog.sqlite"
                if not path.is_file():
                    raise ValueError("Synthetic demo catalog is missing")
                catalog = Catalog(f"sqlite+pysqlite:///{path.as_posix()}")
                if any(not p.product_id.startswith("synthetic:") for p in catalog.all()):
                    raise ValueError("Synthetic catalog contains non-synthetic product records")
            else:
                data_dir = args.data_dir
                catalog = configured_catalog(data_dir)
        if args.command == "export":
            result = export_gallery(
                catalog,
                data_dir,
                args.output_dir,
                scope="synthetic" if args.synthetic_directory else "reviewed-collection",
            )
        else:
            result = validate_gallery(
                args.directory, catalog=catalog, data_dir=data_dir if catalog else None
            )
        print(json.dumps(result, indent=2))
        return 0
    except SQLAlchemyError:
        parser.exit(1, "Configured catalog unavailable or schema not initialized\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Gallery failed: {error}\n")
    finally:
        if catalog and catalog.store:
            catalog.store.engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
