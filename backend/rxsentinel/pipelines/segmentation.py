"""Offline experimental segmentation and fixed development comparison."""

import argparse
import json
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from rxsentinel.catalog import configured_catalog
from rxsentinel.pipelines.photo import bounded_read
from rxsentinel.vision.evaluation import EvaluationManifest
from rxsentinel.vision.report import publish_report
from rxsentinel.vision.segmentation import SegmentationConfig, process_segmented_photo
from rxsentinel.vision.segmentation_comparison import compare_segmentation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    process = actions.add_parser("process", help="Predict from one photograph, without annotations")
    process.add_argument("--input", type=Path, required=True)
    compare = actions.add_parser("compare", help="Compare fixed methods on validation cases")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--data-dir", type=Path, default=Path("data"))
    for command in (process, compare):
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--config", type=Path, help="Fixed SegmentationConfig JSON")
    args = parser.parse_args()
    catalog = None
    try:
        config = (
            SegmentationConfig.model_validate_json(bounded_read(args.config))
            if args.config
            else None
        )
        if args.action == "process":
            content = bounded_read(args.input)
            report, artifacts = process_segmented_photo(content, config=config)
            publish_report(args.output_dir, report, artifacts, content)
            result = {"status": report.status, "rejection_reasons": report.rejection_reasons}
        else:
            manifest = EvaluationManifest.model_validate_json(bounded_read(args.manifest))
            if manifest.scope == "reviewed-collection":
                catalog = configured_catalog(args.data_dir)
            report = compare_segmentation(
                args.manifest,
                args.output_dir,
                config=config,
                store=catalog.store if catalog else None,
                data_dir=args.data_dir,
            )
            result = {"scope": report["scope"], "methods": report["methods"]}
        result["report"] = str((args.output_dir / "report.html").resolve())
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Segmentation failed: {error}\n")
    except SQLAlchemyError:
        parser.exit(1, "Segmentation failed: configured catalog unavailable\n")
    finally:
        if catalog and catalog.store:
            catalog.store.engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
