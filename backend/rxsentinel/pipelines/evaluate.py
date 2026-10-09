"""Evaluate a fixed contour configuration against explicit positive/negative annotations."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sqlalchemy.exc import SQLAlchemyError

from rxsentinel.catalog import configured_catalog
from rxsentinel.pipelines.photo import bounded_read, synthetic_demo
from rxsentinel.vision.evaluation import EvaluationManifest, run_evaluation


def make_demo(directory: Path):
    if directory.exists():
        raise ValueError("Demo directory already exists")
    directory.mkdir(parents=True)
    source, annotation = synthetic_demo()
    normal = np.array(Image.open(io.BytesIO(source)))
    truth = np.array(Image.open(io.BytesIO(annotation)))
    empty = np.full_like(normal, (35, 45, 50))
    low = empty.copy()
    low[truth > 0] = (37, 47, 52)
    distractor = empty.copy()
    cv2.circle(distractor, (400, 300), 90, (170, 200, 180), -1)
    cases = []
    for case_id, rgb, mask, condition in [
        ("positive-clear", normal, truth, "plain-background"),
        ("positive-low-contrast", low, truth, "low-contrast"),
        ("negative-empty", empty, np.zeros_like(truth), "empty-scene"),
        ("negative-distractor", distractor, np.zeros_like(truth), "unrelated-object"),
    ]:
        image_name, mask_name = f"{case_id}.png", f"{case_id}-mask.png"
        Image.fromarray(rgb).save(directory / image_name)
        Image.fromarray(mask).save(directory / mask_name)
        cases.append(
            {
                "case_id": case_id,
                "kind": "positive" if case_id.startswith("positive") else "negative",
                "image": image_name,
                "annotation": mask_name,
                "input_sha256": hashlib.sha256((directory / image_name).read_bytes()).hexdigest(),
                "annotation_sha256": hashlib.sha256(
                    (directory / mask_name).read_bytes()
                ).hexdigest(),
                "capture_session": f"synthetic-{case_id}",
                "conditions": [condition],
            }
        )
    manifest = EvaluationManifest(
        dataset_id="Synthetic geometry only; no real medication photos",
        scope="synthetic",
        partition="validation",
        annotator="Synthetic fixture generator",
        annotation_basis="Known drawn masks; distractor is an unrelated synthetic shape",
        cases=cases,
    )
    path = directory / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Write explicit synthetic positive/negative examples")
    demo.add_argument("--directory", type=Path, required=True)
    run = commands.add_parser("run", help="Evaluate an existing fixed dataset manifest")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    catalog = None
    try:
        if args.command == "demo":
            print(
                json.dumps(
                    {"manifest": str(make_demo(args.directory).resolve()), "scope": "synthetic"},
                    indent=2,
                )
            )
            return 0
        manifest = EvaluationManifest.model_validate_json(bounded_read(args.manifest))
        if manifest.scope == "reviewed-collection":
            catalog = configured_catalog(args.data_dir)
        result = run_evaluation(
            args.manifest,
            args.output_dir,
            store=catalog.store if catalog else None,
            data_dir=args.data_dir,
        )
        print(
            json.dumps(
                {
                    "scope": result["scope"],
                    "metrics": result["metrics"],
                    "report": str((args.output_dir / "report.html").resolve()),
                    "m3_exit_gate_satisfied": False,
                },
                indent=2,
            )
        )
        return 0
    except (ValueError, OSError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    except SQLAlchemyError:
        parser.exit(1, "Evaluation failed: configured catalog unavailable\n")
    finally:
        if catalog and catalog.store:
            catalog.store.engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
