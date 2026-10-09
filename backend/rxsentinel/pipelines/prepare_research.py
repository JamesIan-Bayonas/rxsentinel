"""Prepare downloaded MEDISEG photos for annotation review and M3 development evaluation."""

import argparse
import json
from pathlib import Path

from rxsentinel.research_collection import export_annotation_review, prepare_mediseg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser(
        "prepare", help="Inspect originals and publish a pilot review package"
    )
    prepare.add_argument("--directory", type=Path, default=Path("data/external/mediseg-v2"))
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--per-class", type=int, default=3)
    review = actions.add_parser(
        "export-reviewed", help="Export approved annotations, not identities"
    )
    review.add_argument("--package", type=Path, required=True)
    review.add_argument("--review", type=Path, required=True)
    review.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            result = prepare_mediseg(
                args.directory,
                args.output_dir,
                per_class=args.per_class,
                progress=lambda message: print(message, flush=True),
            )
        else:
            result = export_annotation_review(args.package, args.review, args.output_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Research preparation failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
