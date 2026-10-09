"""Evaluate frozen visual retrieval datasets or create an isolated synthetic demonstration."""

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw
from sqlalchemy.exc import SQLAlchemyError

from rxsentinel.catalog import Catalog, configured_catalog
from rxsentinel.gallery import export_gallery, validate_gallery
from rxsentinel.matching_evaluation import MatchingBenchmark
from rxsentinel.pipelines.evaluate_matching import make_demo
from rxsentinel.pipelines.photo import bounded_read
from rxsentinel.retrieval import build_vectors, validate_vectors
from rxsentinel.vision.embedding import DEFAULT_WEIGHTS, ResNet18Encoder
from rxsentinel.visual_evaluation import VisualBenchmark, evaluation_code_sha256, run_benchmark


def picture(path, label, number):
    """Synthetic geometric cases deliberately include known crop misses and unknown collisions."""
    image = Image.new("RGB", (400 + number, 300), (40, 55, 70))
    drawing = ImageDraw.Draw(image)
    drawing.text((15, 15), "SYNTHETIC GEOMETRY ONLY", fill="white")
    drawing.text((15, 260), label, fill="white")
    if number in (3, 30):
        image.save(path)  # Known missing object and unknown empty scene; kept in denominators.
        return
    if path.name.startswith("beta"):
        drawing.rounded_rectangle((110, 90, 230, 180), radius=12, fill=(90, 160, 220))
    else:
        drawing.ellipse((100, 90, 240, 190), fill=(180, 170, 140))
    if number == 4:
        drawing.ellipse((275, 100, 335, 180), fill=(180, 170, 140))
    image.save(path)


def make_visual_demo(output, weights=DEFAULT_WEIGHTS):
    output = output.resolve()
    if output.exists():
        raise ValueError("Demo directory already exists")
    encoder = ResNet18Encoder(weights)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rxsentinel-visual-demo-", dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        old_path = make_demo(stage, picture_factory=picture)
        manual = MatchingBenchmark.model_validate_json(bounded_read(old_path))
        old_path.rename(stage / "manual-manifest.json")
        catalog = Catalog(f"sqlite+pysqlite:///{(stage / 'catalog.sqlite').as_posix()}")
        try:
            gallery = export_gallery(catalog, stage, stage / "gallery", scope="synthetic")
            index = build_vectors(
                stage / "gallery", stage / "vectors", encoder, catalog=catalog, data_dir=stage
            )
            cases = []
            for case in manual.cases:
                data = case.model_dump(mode="json", exclude={"query"})
                data["conditions"] = [
                    "synthetic-geometry",
                    {
                        "known-alpha": "single-object",
                        "known-beta": "single-object",
                        "known-unreadable": "missing-object",
                        "known-misread": "multiple-objects",
                        "unknown-unmatched": "empty-scene",
                        "unknown-collision": "visual-collision",
                    }[case.case_id],
                ]
                if case.kind == "unknown":
                    data.update(
                        reuse_reviewer="Synthetic fixture generator",
                        reuse_basis="Generated geometry; no real patient or medication data",
                    )
                cases.append(data)
            spec = VisualBenchmark(
                dataset_id="Synthetic visual retrieval benchmark; no real medication photographs",
                scope="synthetic",
                partition="validation",
                synthetic_catalog="catalog.sqlite",
                gallery_id=gallery["gallery_id"],
                index_id=index.index_id,
                evaluator_code_sha256=evaluation_code_sha256(),
                annotator="Synthetic fixture generator",
                annotation_basis="Generated labels, missing/multiple objects and visual collision",
                validation_cutoffs=[0.8, 0.95, 1.0],
                cases=cases,
            )
            path = stage / "manifest.json"
            path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        finally:
            catalog.store.engine.dispose()
        stage.rename(output)
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Create synthetic data, gallery, vectors and protocol")
    demo.add_argument("--directory", type=Path, required=True)
    demo.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    protocol = commands.add_parser("protocol", help="Inspect verified gallery/index protocol pins")
    protocol.add_argument("--data-dir", type=Path, default=Path("data"))
    protocol.add_argument("--synthetic-directory", type=Path)
    run = commands.add_parser("run", help="Measure known/unknown visual retrieval")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--data-dir", type=Path)
    for command in (protocol, run):
        command.add_argument("--gallery-dir", type=Path, required=True)
        command.add_argument("--index-dir", type=Path, required=True)
        command.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    catalog = None
    try:
        if args.command == "demo":
            print(
                json.dumps(
                    {
                        "manifest": str(make_visual_demo(args.directory, args.weights)),
                        "scope": "synthetic",
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "run":
            spec = VisualBenchmark.model_validate_json(bounded_read(args.manifest))
            synthetic = args.manifest.parent.resolve() if spec.scope == "synthetic" else None
            catalog_path = spec.synthetic_catalog
        else:
            synthetic = args.synthetic_directory.resolve() if args.synthetic_directory else None
            catalog_path = "catalog.sqlite"
        if synthetic:
            path = (synthetic / catalog_path).resolve()
            if not path.is_relative_to(synthetic) or not path.is_file():
                raise ValueError("Synthetic catalog must exist inside its dataset directory")
            if args.command == "run" and args.data_dir and args.data_dir.resolve() != synthetic:
                raise ValueError("Synthetic data directory must be the manifest directory")
            data_dir = synthetic
            catalog = Catalog(f"sqlite+pysqlite:///{path.as_posix()}")
            if any(not p.product_id.startswith("synthetic:") for p in catalog.all()):
                raise ValueError("Synthetic catalog contains non-synthetic products")
        else:
            data_dir = args.data_dir or Path("data")
            catalog = configured_catalog(data_dir)
        checked = validate_gallery(args.gallery_dir, catalog=catalog, data_dir=data_dir)
        if (checked["scope"] == "synthetic") != bool(synthetic):
            raise ValueError("Gallery scope does not match source configuration")
        encoder = ResNet18Encoder(args.weights)
        if args.command == "protocol":
            index, gallery, _ = validate_vectors(
                args.index_dir, args.gallery_dir, encoder, catalog=catalog, data_dir=data_dir
            )
            print(
                json.dumps(
                    {
                        "version": "m4-visual-evaluation-1",
                        "gallery_id": gallery.gallery_id,
                        "index_id": index.index_id,
                        "evaluator_code_sha256": evaluation_code_sha256(),
                        "encoder": index.encoder.model_dump(mode="json"),
                        "scope": gallery.scope,
                        "current_source_verified": True,
                    },
                    indent=2,
                )
            )
        else:
            result = run_benchmark(
                args.manifest,
                args.output_dir,
                catalog,
                data_dir,
                args.gallery_dir,
                args.index_dir,
                encoder,
            )
            print(
                json.dumps(
                    {
                        "scope": result["scope"],
                        "metrics": result["metrics"],
                        "report": str((args.output_dir / "report.html").resolve()),
                        "thresholds_calibrated": False,
                        "m4_exit_gate_satisfied": False,
                    },
                    indent=2,
                )
            )
        return 0
    except SQLAlchemyError:
        parser.exit(1, "Configured catalog unavailable or schema not initialized\n")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Visual evaluation failed: {error}\n")
    finally:
        if catalog and catalog.store:
            catalog.store.engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
