"""Benchmark manual matching against a frozen gallery and explicit known/unknown labels."""

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw
from sqlalchemy.exc import SQLAlchemyError

from rxsentinel.catalog import Catalog, configured_catalog
from rxsentinel.collection_models import ReviewInput
from rxsentinel.database.migrate import upgrade_schema
from rxsentinel.matching import snapshot
from rxsentinel.matching_evaluation import MatchingBenchmark, matcher_code_sha256, run_benchmark
from rxsentinel.pipelines.collection import intake, review, statuses
from rxsentinel.pipelines.photo import bounded_read
from rxsentinel.schemas import EvidenceSource, Ingredient, Product


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def picture(path, label, number):
    image = Image.new("RGB", (360 + number, 240), (40 + number, 55, 70))
    drawing = ImageDraw.Draw(image)
    drawing.ellipse((100, 70, 250, 160), fill=(180, 170, 140))
    drawing.text((15, 15), "SYNTHETIC GEOMETRY ONLY", fill="white")
    drawing.text((15, 200), label, fill="white")
    image.save(path)


def make_demo(output, *, picture_factory=picture):
    output = output.resolve()
    if output.exists():
        raise ValueError("Demo directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rxsentinel-match-demo-", dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        catalog = Catalog(f"sqlite+pysqlite:///{(stage / 'catalog.sqlite').as_posix()}")
        try:
            upgrade_schema(catalog.store.engine)
            now = datetime.now(UTC)
            source = EvidenceSource(
                name="Synthetic benchmark only; no real medications",
                url="https://example.org/synthetic-benchmark",
                retrieved_at=now,
            )
            products = [
                Product(
                    product_id=f"synthetic:{name}",
                    product_ndc=f"SYNTHETIC-{name}",
                    generic_name=f"Synthetic fixture {name}; not a medication",
                    dosage_form="SYNTHETIC",
                    ingredients=[
                        Ingredient(
                            source_name="Synthetic ingredient",
                            rxcui="1",
                            normalized_name="Synthetic ingredient",
                            normalization_source=source,
                        )
                    ],
                    source=source,
                    snapshot_id="synthetic-only",
                )
                for name in ("alpha", "beta")
            ]
            catalog.import_products(products)
            (stage / "packaging.txt").write_text(
                "Synthetic identity context only; no real product."
            )
            (stage / "ownership.txt").write_text(
                "Synthetic drawing ownership fixture; not clinical data."
            )
            for group, name in enumerate(("alpha", "beta")):
                photos = []
                for n, (partition, side) in enumerate(
                    [
                        ("reference", "front"),
                        ("reference", "back"),
                        ("validation", "both"),
                        ("validation", "both"),
                        ("validation", "both"),
                    ]
                ):
                    filename = f"{name}-{n}.png"
                    picture_factory(
                        stage / filename,
                        f"A1 / {'111' if name == 'alpha' else '222'}",
                        group * 10 + n,
                    )
                    photos.append(
                        {
                            "file": filename,
                            "side": side,
                            "partition": partition,
                            "capture_session": f"synthetic-{name}-{partition}",
                            "photo_origin": "own_photo",
                            "photographer": "Synthetic generator",
                            "captured_at": now.isoformat(),
                        }
                    )
                manifest = {
                    "appearance_id": f"local:synthetic-{name}",
                    "product_id": f"synthetic:{name}",
                    "collection_version": "synthetic-only",
                    "imprint_front": "A1",
                    "imprint_back": "111" if name == "alpha" else "222",
                    "color": "synthetic-tan",
                    "shape": "synthetic-oval",
                    "manufacturer": "Synthetic",
                    "photos": photos,
                    "evidence": [
                        {
                            "evidence_id": "context",
                            "purpose": "packaging",
                            "description": "Synthetic context",
                            "file": "packaging.txt",
                        },
                        {
                            "evidence_id": "reuse",
                            "purpose": "ownership",
                            "description": "Synthetic ownership",
                            "file": "ownership.txt",
                        },
                    ],
                }
                path = stage / f"intake-{name}.json"
                path.write_text(json.dumps(manifest))
                intake(catalog.store, stage, path)
                state = next(
                    s
                    for s in statuses(catalog.store, stage)
                    if s["appearance_id"] == manifest["appearance_id"]
                )
                review(
                    catalog.store,
                    stage,
                    ReviewInput(
                        appearance_id=manifest["appearance_id"],
                        expected_fingerprint=state["fingerprint"],
                        reviewer="Synthetic workflow generator",
                        identity_decision="approve",
                        reuse_decision="approve",
                        basis="Synthetic demonstration; no real identity or permission approval",
                        identity_evidence_ids=["context"],
                        reuse_evidence_ids=["reuse"],
                        reuse_asset_ids=[a["asset_id"] for a in state["assets"]],
                        valid_until=now + timedelta(days=30),
                    ),
                )
            cases = []
            assets = catalog.store.assets()
            for name, number, case_id, query, condition in [
                ("alpha", 2, "known-alpha", {"imprint_front": "A1"}, "single-view-ambiguity"),
                (
                    "beta",
                    2,
                    "known-beta",
                    {"imprint_front": "A1", "imprint_back": "222"},
                    "two-views",
                ),
                (
                    "alpha",
                    3,
                    "known-unreadable",
                    {"imprint_front": "A?"},
                    "unreadable-transcription",
                ),
                (
                    "alpha",
                    4,
                    "known-misread",
                    {"imprint_front": "A1", "imprint_back": "222"},
                    "transcription-error",
                ),
            ]:
                sha = file_hash(stage / f"{name}-{number}.png")
                asset = next(a for a in assets if a.sha256 == sha)
                cases.append(
                    {
                        "case_id": case_id,
                        "kind": "known",
                        "expected_appearance_id": asset.appearance_id,
                        "query": query,
                        "asset_id": asset.asset_id,
                        "input_sha256": sha,
                        "capture_session": asset.capture_session,
                        "conditions": [condition],
                    }
                )
            evidence = [
                {
                    "evidence_id": "context",
                    "purpose": "packaging",
                    "description": "Synthetic open-set context",
                    "local_path": "packaging.txt",
                    "sha256": file_hash(stage / "packaging.txt"),
                },
                {
                    "evidence_id": "reuse",
                    "purpose": "ownership",
                    "description": "Synthetic rights fixture",
                    "local_path": "ownership.txt",
                    "sha256": file_hash(stage / "ownership.txt"),
                },
            ]
            for n, (case_id, query) in enumerate(
                [
                    ("unknown-unmatched", {"imprint_front": "Z9"}),
                    ("unknown-collision", {"imprint_front": "A1", "imprint_back": "111"}),
                ]
            ):
                filename = f"{case_id}.png"
                label = "Z9" if n == 0 else "A1 / 111"
                picture_factory(stage / filename, f"{label}: OUTSIDE SYNTHETIC GALLERY", 30 + n)
                cases.append(
                    {
                        "case_id": case_id,
                        "kind": "unknown",
                        "query": query,
                        "image": filename,
                        "input_sha256": file_hash(stage / filename),
                        "capture_session": "synthetic-open-set-validation",
                        "conditions": [case_id],
                        "evidence": evidence,
                    }
                )
            spec = MatchingBenchmark(
                dataset_id="Synthetic manual matching benchmark; no real medication photos",
                scope="synthetic",
                partition="validation",
                synthetic_catalog="catalog.sqlite",
                reference_snapshot_sha256=snapshot(catalog, stage)[3],
                matcher_code_sha256=matcher_code_sha256(),
                annotator="Synthetic fixture generator",
                annotation_basis="Synthetic transcription errors and an imprint collision",
                observations_transcribed_from_query=True,
                cases=cases,
            )
            (stage / "manifest.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        finally:
            catalog.store.engine.dispose()
        stage.rename(output)
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Create an isolated synthetic gallery and benchmark")
    demo.add_argument("--directory", type=Path, required=True)
    run = commands.add_parser("run", help="Evaluate a frozen known/unknown manifest")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--data-dir", type=Path)
    protocol = commands.add_parser(
        "protocol", help="Show current matcher and catalog fingerprint pins"
    )
    protocol.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    catalog = None
    try:
        if args.command == "demo":
            print(
                json.dumps(
                    {"manifest": str(make_demo(args.directory)), "scope": "synthetic"}, indent=2
                )
            )
            return 0
        if args.command == "protocol":
            from rxsentinel.matching import MATCHER_VERSION, MatchQuery, match_observations

            catalog = configured_catalog(args.data_dir)
            result = match_observations(catalog, args.data_dir, MatchQuery())
            print(
                json.dumps(
                    {
                        "matcher_version": MATCHER_VERSION,
                        "matcher_code_sha256": matcher_code_sha256(),
                        "reference_snapshot_sha256": result.reference_snapshot_sha256,
                        "eligible_appearance_count": result.eligible_appearance_count,
                    },
                    indent=2,
                )
            )
            return 0
        spec = MatchingBenchmark.model_validate_json(bounded_read(args.manifest))
        if spec.scope == "synthetic":
            root = args.manifest.parent.resolve()
            path = (root / spec.synthetic_catalog).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Synthetic catalog must exist inside the manifest directory")
            if args.data_dir and args.data_dir.resolve() != root:
                raise ValueError("Synthetic data directory must be the manifest directory")
            data_dir = root
            catalog = Catalog(f"sqlite+pysqlite:///{path.as_posix()}")
        else:
            data_dir = args.data_dir or Path("data")
            catalog = configured_catalog(data_dir)
        result = run_benchmark(args.manifest, args.output_dir, catalog, data_dir)
        print(
            json.dumps(
                {
                    "scope": result["scope"],
                    "metrics": result["metrics"],
                    "report": str((args.output_dir / "report.html").resolve()),
                    "m4_exit_gate_satisfied": False,
                },
                indent=2,
            )
        )
        return 0
    except SQLAlchemyError:
        parser.exit(1, "Configured catalog unavailable or schema not initialized\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")
    finally:
        if catalog and catalog.store:
            catalog.store.engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
