import argparse
import csv
import json
from pathlib import Path

from rxsentinel.catalog import Catalog


def generate_readiness(data_dir: Path) -> dict:
    products = Catalog(data_dir / "catalog.sqlite").all()
    rows = []
    for product in products:
        gaps = []
        if any(not ingredient.rxcui for ingredient in product.ingredients):
            gaps.append("ingredient_mapping")
        for field in ("imprint", "shape", "color"):
            if not getattr(product, field):
                gaps.append(field)
        if not product.reference_images:
            gaps.append("reference_images")
        if not product.image_reuse_verified:
            gaps.append("image_reuse_terms")
        rows.append(
            {
                "product_id": product.product_id,
                "name": product.brand_name or product.generic_name,
                "snapshot_id": product.snapshot_id,
                "ingredients_normalized": all(i.rxcui for i in product.ingredients),
                "reference_images": len(product.reference_images),
                "appearance_metadata_ready": not gaps,
                "missing_requirements": ";".join(gaps),
            }
        )
    summary = {
        "product_count": len(products),
        "fully_normalized_product_count": sum(r["ingredients_normalized"] for r in rows),
        "appearance_metadata_ready_count": sum(r["appearance_metadata_ready"] for r in rows),
        "visual_identification_ready": False,
        "next_gate": "Verify image reuse, link actual pill appearances, and collect independent "
        "evaluation photos before training or claiming visual coverage.",
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "product_id", "name", "snapshot_id", "ingredients_normalized", "reference_images",
        "appearance_metadata_ready", "missing_requirements",
    ]
    with (data_dir / "readiness.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (data_dir / "readiness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Report reference-catalog feasibility gaps")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(generate_readiness(args.data_dir), indent=2))


if __name__ == "__main__":
    main()
