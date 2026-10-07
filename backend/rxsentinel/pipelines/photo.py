"""Offline M3 CLI. Processing never changes catalog data or reference eligibility."""

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rxsentinel.pipelines.reference_files import MAX_IMAGE_BYTES
from rxsentinel.vision.report import publish_report
from rxsentinel.vision.single import ProcessingConfig, evaluate_mask, process_photo


def bounded_read(path: Path) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(MAX_IMAGE_BYTES + 1)
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Image byte limit exceeded")
    return content


def synthetic_demo() -> tuple[bytes, bytes]:
    """Geometric fixture, explicitly unrelated to any drug or catalog product."""
    rgb = np.full((600, 800, 3), (35, 45, 50), dtype=np.uint8)
    truth = np.zeros((600, 800), dtype=np.uint8)
    cv2.ellipse(truth, (400, 300), (160, 95), -15, 0, 360, 255, -1)
    rgb[truth > 0] = (205, 170, 140)
    cv2.line(rgb, (370, 255), (430, 345), (125, 90, 65), 3)
    result = []
    for pixels in (rgb, truth):
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        result.append(buffer.getvalue())
    return result[0], result[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input", type=Path, help="One opaque JPEG/PNG photo on a plain background"
    )
    inputs.add_argument("--demo", action="store_true", help="Synthetic geometry; no real pill data")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--config", type=Path, help="ProcessingConfig JSON overrides")
    parser.add_argument("--annotation", type=Path, help="Oriented, binary positive-object mask PNG")
    args = parser.parse_args()
    if args.demo and args.annotation:
        parser.error("The demo supplies its own synthetic annotation")
    try:
        config = (
            ProcessingConfig.model_validate_json(bounded_read(args.config)) if args.config else None
        )
        if args.demo:
            source, annotation = synthetic_demo()
        else:
            source = bounded_read(args.input)
            annotation = bounded_read(args.annotation) if args.annotation else None
        report, artifacts = process_photo(source, config)
        if annotation:
            evaluate_mask(
                report,
                artifacts,
                annotation,
                scope="synthetic" if args.demo else "unverified-local",
            )
            with Image.open(io.BytesIO(annotation)) as image:
                artifacts["annotation"] = np.array(image.convert("L"))
        if args.demo:
            report.warnings.insert(
                0, "SYNTHETIC GEOMETRY DEMO: not a real pill or clinical evaluation"
            )
        path = publish_report(args.output_dir, report, artifacts, source)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Processing failed: {error}\n")
    print(
        json.dumps(
            {
                "status": report.status,
                "report": str(path),
                "rejection_reasons": report.rejection_reasons,
                "medication_identification_available": False,
            },
            indent=2,
        )
    )
    return 0 if report.status == "processed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
