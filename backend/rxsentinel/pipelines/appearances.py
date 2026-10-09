import argparse
import asyncio
import csv
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from rxsentinel.catalog import configured_catalog
from rxsentinel.pipelines.ingest import get_json
from rxsentinel.schemas import Appearance, EvidenceSource, Product

DATASET_URL = "https://datadiscovery.nlm.nih.gov/api/views/crzr-uvwg.json"
RECORDS_URL = "https://datadiscovery.nlm.nih.gov/resource/crzr-uvwg.json"


def appearance_candidate(record: dict, product: Product, source: EvidenceSource) -> Appearance:
    if record.get("product_code") != product.product_ndc:
        raise ValueError("Archived product code must exactly match the catalog product")
    return Appearance(
        appearance_id=f"pillbox:{record['id']}",
        product_ids=[product.product_id],
        imprint_unassigned=record.get("pillbox_imprint") or record.get("splimprint"),
        shape=record.get("pillbox_shape_text") or record.get("splshape_text"),
        color=record.get("pillbox_color_text") or record.get("splcolor_text"),
        score_marks=record.get("pillbox_score") or record.get("splscore"),
        source=source,
        source_version=str(record.get("updated_at") or record.get("effective_time") or "unknown"),
        identity_link_status="unverified",
        stale=True,
        notes=[
            "Archived product-code match; current appearance and formulation are not verified.",
            "Imprint text is not assigned to a side; blank values are not verified blank imprints.",
            f"Archived image flag: {record.get('has_image', 'unknown')}; "
            f"image identifier: {record.get('splimage') or 'missing'}; "
            f"origin: {record.get('image_source') or 'unknown'}.",
        ],
    )


async def discover(data_dir: Path) -> dict:
    catalog = configured_catalog(data_dir)
    if not catalog.store:
        raise ValueError("Configure the versioned relational database before appearance discovery")
    products = catalog.all()
    if not products:
        raise ValueError("Import product metadata before appearance discovery")
    by_code = {product.product_ndc: product for product in products}
    if any(not re.fullmatch(r"\d{4,5}-\d{3,4}", code) for code in by_code):
        raise ValueError("Catalog contains unsupported product-code formats")
    if len(by_code) > 100:
        raise ValueError("This feasibility command supports at most 100 product codes per run")
    codes = ",".join(f"'{code}'" for code in sorted(by_code))
    url = str(
        httpx.URL(
            RECORDS_URL,
            params={
                "$where": f"product_code in ({codes})",
                "$limit": "1000",
            },
        )
    )
    retrieved_at = datetime.now(UTC)
    snapshot_id = retrieved_at.strftime("pillbox-discovery-%Y%m%dT%H%M%S%fZ")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        metadata = await get_json(client, DATASET_URL)
        response = await client.get(url)
        response.raise_for_status()
        records = response.json()
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise ValueError("Expected an array of archived appearance records")
        if len(records) >= 1000:
            raise ValueError("Archived response may be truncated; use a smaller catalog sample")
    source = EvidenceSource(
        name="NLM Pillbox archived appearance metadata", url=url, retrieved_at=retrieved_at
    )
    candidates = []
    checklist = []
    for record in records:
        product = by_code.get(record.get("product_code"))
        if product is None:
            raise ValueError("Provider returned a product outside the requested exact codes")
        candidate = appearance_candidate(record, product, source)
        candidates.append(candidate)
        checklist.append(
            {
                "appearance_id": candidate.appearance_id,
                "product_id": product.product_id,
                "name": product.brand_name or product.generic_name,
                "archived_has_image": str(record.get("has_image", "")).casefold() == "true",
                "image_identifier": record.get("splimage") or "",
                "image_origin": record.get("image_source") or "",
                "identity_link_verified": False,
                "image_reuse_verified": False,
                "reference_image_downloaded": False,
                "archived": True,
            }
        )
    raw = json.dumps({"dataset_metadata": metadata, "records": records}, indent=2).encode("utf-8")
    manifest = {
        "snapshot_id": snapshot_id,
        "retrieved_at": retrieved_at.isoformat(),
        "query_url": url,
        "dataset_url": DATASET_URL,
        "dataset_name": metadata.get("name"),
        "catalog_products": len(products),
        "archived_appearance_candidates": len(candidates),
        "products_with_archived_matches": len({r["product_id"] for r in checklist}),
        "candidates_with_image_flags": sum(r["archived_has_image"] for r in checklist),
        "reference_images_verified": 0,
        "visual_identification_ready": False,
        "dataset_license_metadata": metadata.get("license"),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "remaining_requirements": [
            "Verify current appearance and formulation against the archived product link.",
            "Inspect actual image files and their specific origins and reuse terms.",
            "Collect independent query photos and capture-session partitions.",
        ],
    }
    directory = data_dir / "source-discovery" / snapshot_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "raw.json").write_bytes(raw)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    catalog.store.import_appearances(candidates, manifest)
    (data_dir / "appearance-candidates.json").write_text(
        json.dumps([candidate.model_dump(mode="json") for candidate in candidates], indent=2),
        encoding="utf-8",
    )
    (data_dir / "image-source-feasibility.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    fields = list(checklist[0]) if checklist else ["appearance_id", "product_id"]
    with (data_dir / "image-source-feasibility.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checklist)
    catalog.store.engine.dispose()
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Discover unverified archived appearance links")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(asyncio.run(discover(args.data_dir)), indent=2))


if __name__ == "__main__":
    main()
