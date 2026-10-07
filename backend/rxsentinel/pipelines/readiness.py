import argparse
import csv
import json
from pathlib import Path

from rxsentinel.catalog import configured_catalog
from rxsentinel.collection_support import collection_status, file_valid


def asset_file_valid(data_dir, asset):
    return file_valid(data_dir, asset, "assets", image=True)


def generate_readiness(data_dir: Path) -> dict:
    catalog = configured_catalog(data_dir)
    products = catalog.all()
    appearance_records = catalog.store.appearances() if catalog.store else []
    asset_records = catalog.store.assets() if catalog.store else []
    checks = catalog.store.reference_checks() if catalog.store else []
    valid_files = {a.asset_id: asset_file_valid(data_dir, a) for a in asset_records}
    collection_rows = [
        collection_status(
            data_dir,
            a,
            [asset for asset in asset_records if asset.appearance_id == a.appearance_id],
            [product for product in products if product.product_id in a.product_ids],
            asset_records,
        )
        for a in appearance_records
    ]
    rows = []
    for product in products:
        linked = [a for a in appearance_records if product.product_id in a.product_ids]
        statuses = [
            status
            for status in collection_rows
            if status["appearance_id"] in {a.appearance_id for a in linked}
        ]
        gaps = sorted({gap for status in statuses for gap in status["missing_requirements"]})
        if not statuses:
            gaps.append("current_reviewed_appearance_link")
        ready = any(status["reference_ready"] for status in statuses)
        rows.append(
            {
                "product_id": product.product_id,
                "name": product.brand_name or product.generic_name,
                "snapshot_id": product.snapshot_id,
                "ingredients_normalized": all(i.rxcui for i in product.ingredients),
                "reference_images": sum(
                    a.partition == "reference"
                    for a in asset_records
                    if a.appearance_id in {appearance.appearance_id for appearance in linked}
                ),
                "archived_appearance_candidates": sum(
                    a.appearance_id.startswith("pillbox:") for a in linked
                ),
                "verified_identity_links": sum(s["reference_ready"] for s in statuses),
                "appearance_metadata_ready": ready,
                "missing_requirements": "" if ready else ";".join(gaps),
            }
        )
    eligible_appearances = [s for s in collection_rows if s["reference_ready"]]
    evaluated = [
        s for s in eligible_appearances if {"validation", "test"} <= set(s["evaluation_partitions"])
    ]
    summary = {
        "product_count": len(products),
        "database_backend": catalog.backend,
        "fully_normalized_product_count": sum(r["ingredients_normalized"] for r in rows),
        "appearance_metadata_ready_count": sum(r["appearance_metadata_ready"] for r in rows),
        "visual_identification_ready": False,
        "appearance_candidate_count": len(appearance_records),
        "verified_appearance_links": sum(
            a.identity_link_status == "verified" and not a.stale for a in appearance_records
        ),
        "registered_image_asset_count": len(asset_records),
        "valid_image_file_count": sum(valid_files.values()),
        "inspection_only_image_count": sum(
            a.intended_use == "inspection_only" for a in asset_records
        ),
        "current_label_check_count": len(checks),
        "local_collection_count": sum(
            a.appearance_id.startswith("local:") for a in appearance_records
        ),
        "eligible_reference_appearance_count": len(eligible_appearances),
        "appearances_with_validation_and_test_photos": len(evaluated),
        "reference_data_gate_ready": len(evaluated) >= 20,
        "collection_requirements": collection_rows,
        "next_gate": "Verify image reuse, link actual pill appearances, and collect independent "
        "evaluation photos before training or claiming visual coverage.",
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "product_id",
        "name",
        "snapshot_id",
        "ingredients_normalized",
        "reference_images",
        "archived_appearance_candidates",
        "verified_identity_links",
        "appearance_metadata_ready",
        "missing_requirements",
    ]
    with (data_dir / "readiness.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (data_dir / "readiness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if catalog.store:
        catalog.store.engine.dispose()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Report reference-catalog feasibility gaps")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(generate_readiness(args.data_dir), indent=2))


if __name__ == "__main__":
    main()
