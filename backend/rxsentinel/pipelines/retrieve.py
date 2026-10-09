"""Build and query a reviewed image gallery with fixed experimental visual features."""

import argparse
import html
import json
import tempfile
from pathlib import Path

import httpx
from sqlalchemy.exc import SQLAlchemyError

from rxsentinel.catalog import Catalog, configured_catalog
from rxsentinel.gallery import validate_gallery
from rxsentinel.pipelines.photo import bounded_read
from rxsentinel.retrieval import build_vectors, retrieve_photo, validate_vectors
from rxsentinel.vision.embedding import DEFAULT_WEIGHTS, ResNet18Encoder, download_weights


def write_report(report, output):
    output = output.resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new report directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rxsentinel-visual-report-", dir=output.parent
    ) as temp:
        stage = Path(temp) / "report"
        stage.mkdir()
        (stage / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        rows = "".join(
            f"<tr><td>{html.escape(c.appearance_id)}</td>"
            f"<td>{html.escape(', '.join(p.generic_name for p in c.products))}</td>"
            f"<td>{c.cosine_similarity:.6f}</td>"
            f"<td>{html.escape(c.best_reference.asset_id)}</td></tr>"
            for c in report.candidates
        )
        page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RxSentinel experimental visual retrieval</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}}
td,th{{padding:.6rem;border-bottom:1px solid #ccc;text-align:left}}
code{{overflow-wrap:anywhere}}</style>
<h1>Experimental visual retrieval: {report.status}</h1>
<p>Scope: {report.scope}. {html.escape(report.reason.replace("_", " "))}.</p>
<p>No medication identity is confirmed. Similarity scores are not confidence percentages.</p>
<p>Total candidates: {report.total_candidate_count}; displayed: {len(report.candidates)};
tied first candidates: {report.tied_first_candidate_count}.</p>
<p>Source reviews were checked at execution time. Recheck before another use.</p>
<table><tr><th>Appearance</th><th>Catalog name</th><th>Cosine similarity</th>
<th>Best reference image ID</th></tr>{rows}</table>
<p><a href="report.json">Full provenance, processing warnings and model versions</a></p></html>"""
        (stage / "report.html").write_text(page, encoding="utf-8")
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download-weights", help="Explicitly download pinned official weights"
    )
    download.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    for name in ("build", "validate", "query"):
        command = commands.add_parser(name)
        command.add_argument("--gallery-dir", type=Path, required=True)
        command.add_argument("--index-dir", type=Path, required=True)
        command.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
        command.add_argument("--data-dir", type=Path, default=Path("data"))
        command.add_argument(
            "--synthetic-directory",
            type=Path,
            help="Isolated synthetic catalog; never opens production configuration",
        )
        if name == "query":
            command.add_argument("--photo", type=Path, required=True)
            command.add_argument("--limit", type=int, default=3)
            command.add_argument(
                "--minimum-similarity",
                type=float,
                help="Optional experimental cutoff; not calibrated identity acceptance",
            )
            command.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    catalog = None
    try:
        if args.command == "download-weights":
            print(json.dumps(download_weights(args.weights), indent=2))
            return 0
        if args.synthetic_directory:
            data_dir = args.synthetic_directory.resolve()
            path = data_dir / "catalog.sqlite"
            if not path.is_file():
                raise ValueError("Synthetic catalog missing")
            catalog = Catalog(f"sqlite+pysqlite:///{path.as_posix()}")
            if any(not p.product_id.startswith("synthetic:") for p in catalog.all()):
                raise ValueError("Synthetic catalog contains non-synthetic products")
        else:
            data_dir = args.data_dir
            catalog = configured_catalog(data_dir)
        summary = validate_gallery(args.gallery_dir, catalog=catalog, data_dir=data_dir)
        if summary["scope"] == "synthetic" and not args.synthetic_directory:
            raise ValueError("Synthetic galleries require an isolated --synthetic-directory")
        if summary["scope"] != "synthetic" and args.synthetic_directory:
            raise ValueError("Reviewed production galleries cannot use a synthetic catalog")
        # Validate the gallery before importing the optional model stack or loading weights.
        encoder = ResNet18Encoder(args.weights)
        if args.command == "build":
            result = build_vectors(
                args.gallery_dir, args.index_dir, encoder, catalog=catalog, data_dir=data_dir
            )
            print(
                json.dumps(
                    {
                        "index_id": result.index_id,
                        "gallery_id": result.gallery_id,
                        "reference_image_count": len(result.rows),
                        "dimension": result.encoder.dimension,
                        "scope": summary["scope"],
                        "photo_identification_available": False,
                    },
                    indent=2,
                )
            )
        elif args.command == "validate":
            manifest, _, _ = validate_vectors(
                args.index_dir, args.gallery_dir, encoder, catalog=catalog, data_dir=data_dir
            )
            print(
                json.dumps(
                    {"index_id": manifest.index_id, "current_source_verified": True}, indent=2
                )
            )
        else:
            result = retrieve_photo(
                bounded_read(args.photo),
                args.index_dir,
                args.gallery_dir,
                encoder,
                catalog=catalog,
                data_dir=data_dir,
                limit=args.limit,
                minimum_similarity=args.minimum_similarity,
            )
            if args.output_dir:
                write_report(result, args.output_dir)
            print(result.model_dump_json(indent=2))
        return 0
    except SQLAlchemyError:
        parser.exit(1, "Configured catalog unavailable or schema not initialized\n")
    except (ValueError, OSError, RuntimeError, httpx.HTTPError) as error:
        parser.exit(1, f"Visual retrieval failed: {error}\n")
    finally:
        if catalog and catalog.store:
            catalog.store.engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
