import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from rxsentinel.catalog import configured_catalog
from rxsentinel.schemas import EvidenceSource, Ingredient, Product

DEFAULT_QUERIES = (
    'generic_name:"metformin" AND dosage_form:"TABLET" AND route:"ORAL"',
    'generic_name:"aspirin" AND dosage_form:"TABLET" AND route:"ORAL"',
    'generic_name:"warfarin" AND dosage_form:"TABLET" AND route:"ORAL"',
)


async def get_json(client: httpx.AsyncClient, url: str, params: dict | None = None) -> dict:
    """Bounded retries for transient errors. Invalid responses fail the import."""
    for attempt in range(4):
        try:
            response = await client.get(url, params=params)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                await asyncio.sleep(2**attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object from the reference provider")
            return payload
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == 3:
                raise
            await asyncio.sleep(2**attempt)
    raise RuntimeError("Reference provider retry limit exhausted")


async def resolve_ingredient(client: httpx.AsyncClient, name: str, retrieved_at: datetime):
    """Exact-name lookup only; ambiguous/failed mappings remain unresolved."""
    url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
    lookup = await get_json(client, url, {"name": name, "search": "0"})
    ids = lookup.get("idGroup", {}).get("rxnormId", []) or []
    if len(ids) != 1:
        return {"rxcui": None, "normalized_name": None, "normalization_source": None}, {
            "lookup": lookup,
            "reason": "exact_lookup_not_unique",
        }
    related_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{ids[0]}/related.json"
    related = await get_json(client, related_url, {"tty": "IN"})
    concepts = [
        concept
        for group in related.get("relatedGroup", {}).get("conceptGroup", []) or []
        if group.get("tty") == "IN"
        for concept in group.get("conceptProperties", []) or []
    ]
    if len(concepts) != 1:
        return {"rxcui": None, "normalized_name": None, "normalization_source": None}, {
            "lookup": lookup,
            "related": related,
            "reason": "ingredient_relation_not_unique",
        }
    concept = concepts[0]
    return {
        "rxcui": concept["rxcui"],
        "normalized_name": concept["name"],
        "normalization_source": EvidenceSource(
            name="NLM RxNorm exact lookup and IN relationship",
            url=str(httpx.URL(related_url, params={"tty": "IN"})),
            retrieved_at=retrieved_at,
            section=f"Exact source name: {name}; matched concept: {ids[0]}",
        ),
    }, {"lookup": lookup, "related": related}


async def ingest(data_dir: Path, queries: list[str], limit: int) -> dict:
    retrieved_at = datetime.now(UTC)
    snapshot_id = retrieved_at.strftime("openfda-rxnorm-%Y%m%dT%H%M%S%fZ")
    raw_responses: list[dict] = []
    products: dict[str, Product] = {}
    mappings: dict[str, dict] = {}
    raw_mappings: dict[str, dict] = {}
    async with httpx.AsyncClient(
        timeout=30,
        follow_redirects=True,
        headers={"User-Agent": "RxSentinel-research-prototype/0.1"},
    ) as client:
        rxnorm_version = await get_json(client, "https://rxnav.nlm.nih.gov/REST/version.json")
        for query in queries:
            url = str(
                httpx.URL(
                    "https://api.fda.gov/drug/ndc.json",
                    params={"search": query, "limit": limit},
                )
            )
            payload = await get_json(client, url)
            raw_responses.append({"url": url, "payload": payload})
            for record in payload["results"]:
                ingredients: list[Ingredient] = []
                for item in record.get("active_ingredients", []):
                    name = item["name"]
                    cache_key = name.casefold().strip()
                    if cache_key not in mappings:
                        mapping, raw = await resolve_ingredient(client, name, retrieved_at)
                        mappings[cache_key] = mapping
                        raw_mappings[cache_key] = raw
                    ingredients.append(
                        Ingredient(
                            source_name=name,
                            strength=item.get("strength"),
                            **mappings[cache_key],
                        )
                    )
                ndc = record["product_ndc"]
                product = Product(
                    product_id=f"ndc:{ndc}",
                    product_ndc=ndc,
                    brand_name=record.get("brand_name"),
                    generic_name=record["generic_name"],
                    dosage_form=record["dosage_form"],
                    manufacturers=record.get("openfda", {}).get("manufacturer_name", []),
                    ingredients=ingredients,
                    source=EvidenceSource(
                        name="openFDA NDC Directory",
                        url=url,
                        retrieved_at=retrieved_at,
                    ),
                    snapshot_id=snapshot_id,
                )
                products[product.product_id] = product

    if not products:
        raise ValueError("The import did not produce any validated products")
    raw_text = json.dumps(
        {"openfda": raw_responses, "rxnorm": raw_mappings, "rxnorm_version": rxnorm_version},
        indent=2,
        ensure_ascii=False,
    )
    manifest = {
        "snapshot_id": snapshot_id,
        "retrieved_at": retrieved_at.isoformat(),
        "product_count": len(products),
        "fully_normalized_product_count": sum(
            all(i.rxcui for i in p.ingredients) for p in products.values()
        ),
        "reference_image_count": 0,
        "visual_identification_ready": False,
        "queries": queries,
        "limit_per_query": limit,
        "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "rxnorm_version": rxnorm_version,
        "notes": [
            "Metadata feasibility sample; not an exhaustive or representative product catalog.",
            "NDC presence does not establish FDA approval or validate listing accuracy.",
            "Appearance characteristics and images are not supplied by this ingestion source.",
            "RxNorm IN mapping is ingredient-level only; formulation differences remain relevant.",
        ],
    }
    # All network fetches and validation finish before the catalog is mutated.
    snapshot_dir = data_dir / "snapshots" / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    # Write exact hashed bytes; Windows text-mode newline conversion changes checksums.
    (snapshot_dir / "raw.json").write_bytes(raw_text.encode("utf-8"))
    (snapshot_dir / "products.json").write_text(
        json.dumps([p.model_dump(mode="json") for p in products.values()], indent=2),
        encoding="utf-8",
    )
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    configured_catalog(data_dir).import_products(list(products.values()), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Import a small, traceable metadata sample")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit-per-query", type=int, default=10)
    parser.add_argument("--query", action="append", help="Repeat to replace the default queries")
    args = parser.parse_args()
    if not 1 <= args.limit_per_query <= 100:
        parser.error("--limit-per-query must be between 1 and 100")
    manifest = asyncio.run(
        ingest(args.data_dir, args.query or list(DEFAULT_QUERIES), args.limit_per_query)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
