"""Exercise the running local API with catalog-only demonstration inputs."""

import argparse
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description="Check the running local API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expect-backend", choices=["mariadb", "mysql", "sqlite"])
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=10) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["catalog_ready"]
        assert health.json()["photo_identification_available"] is False
        backend = health.json()["database_backend"]
        if args.expect_backend:
            assert backend == args.expect_backend
        response = client.get("/api/v1/catalog/products", params={"limit": 100})
        response.raise_for_status()
        products = response.json()

        def matches(rxcui):
            return [
                p
                for p in products
                if len(p["ingredients"]) == 1 and p["ingredients"][0]["rxcui"] == rxcui
            ]

        aspirin = matches("1191")[0]
        warfarin = matches("11289")[0]
        metformin = matches("6809")[:2]
        assert len(metformin) == 2
        chosen = [aspirin, warfarin, *metformin]
        request = {
            "medications": [
                {
                    "entry_id": f"demo-{i}",
                    "product_id": p["product_id"],
                    "identity_confirmed": True,
                }
                for i, p in enumerate(chosen)
            ]
        }
        response = client.post("/api/v1/medication-audits", json=request)
        response.raise_for_status()
        report = response.json()
        assert {finding["kind"] for finding in report["findings"]} == {
            "possible_duplicate_ingredient",
            "documented_interaction",
        }
        assert not report["excluded_entries"]
        assert report["interaction_coverage"] == "limited_curated_rules"
        assert client.get("/openapi.json").status_code == 200
        appearance_response = client.get("/api/v1/catalog/appearances", params={"limit": 100})
        appearance_response.raise_for_status()
        appearances = appearance_response.json()
        asset_response = client.get("/api/v1/catalog/image-assets", params={"limit": 100})
        asset_response.raise_for_status()
        assets = asset_response.json()
        check_response = client.get("/api/v1/catalog/reference-checks", params={"limit": 100})
        check_response.raise_for_status()
        checks = check_response.json()
        collection_response = client.get("/api/v1/catalog/collections")
        collection_response.raise_for_status()
        collections = collection_response.json()
        assert client.get("/api/v1/catalog/collection-reviews").status_code == 200
    artifact = {
        "purpose": "Catalog-only API demonstration; not a real patient's regimen. "
        "Identity confirmation flags are simulated for this test.",
        "health": health.json(),
        "request": request,
        "response": report,
    }
    Path("data/audit-example.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "catalog_products": len(products),
                "database_backend": backend,
                "appearance_records": len(appearances),
                "registered_image_assets": len(assets),
                "reference_checks": len(checks),
                "local_photo_collections": len(collections),
                "audit_findings": len(report["findings"]),
                "finding_types": sorted({f["kind"] for f in report["findings"]}),
                "artifact": "data/audit-example.json",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
