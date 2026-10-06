from datetime import UTC, datetime

import httpx
import pytest
from rxsentinel.pipelines.ingest import get_json, resolve_ingredient
from rxsentinel.pipelines.readiness import generate_readiness


@pytest.mark.anyio
async def test_exact_lookup_maps_precise_ingredient_to_base():
    def handler(request):
        if request.url.path == "/REST/rxcui.json":
            assert request.url.params["search"] == "0"
            return httpx.Response(200, json={"idGroup": {"rxnormId": ["999"]}})
        assert request.url.params["tty"] == "IN"
        return httpx.Response(
            200,
            json={
                "relatedGroup": {
                    "conceptGroup": [
                        {"tty": "IN", "conceptProperties": [{"rxcui": "6809", "name": "metformin"}]}
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        mapping, raw = await resolve_ingredient(
            client, "metformin hydrochloride", datetime.now(UTC)
        )
    assert mapping["rxcui"] == "6809"
    assert "related" in raw


@pytest.mark.anyio
async def test_ambiguous_lookup_is_not_guessed():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"idGroup": {"rxnormId": ["1", "2"]}})
        )
    ) as client:
        mapping, _ = await resolve_ingredient(client, "ambiguous", datetime.now(UTC))
    assert mapping["rxcui"] is None


@pytest.mark.anyio
async def test_nontransient_failure_is_not_hidden():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(403, json={"error": "forbidden"}))
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await get_json(client, "https://example.org")


def test_metadata_only_catalog_cannot_pass_visual_readiness(catalog):
    summary = generate_readiness(catalog.path.parent)
    assert summary["product_count"] == 6
    assert summary["appearance_metadata_ready_count"] == 0
    assert summary["visual_identification_ready"] is False
    assert "reference_images" in (catalog.path.parent / "readiness.csv").read_text()


@pytest.mark.anyio
async def test_transient_errors_have_bounded_retries(monkeypatch):
    calls = []

    async def no_sleep(_):
        return None

    monkeypatch.setattr("rxsentinel.pipelines.ingest.asyncio.sleep", no_sleep)

    def handler(_):
        calls.append(True)
        return httpx.Response(429, json={"error": "rate limit"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await get_json(client, "https://example.org")
    assert len(calls) == 4


@pytest.mark.anyio
async def test_import_snapshot_integrity_and_deduplication(tmp_path, monkeypatch):
    import hashlib
    import json

    from rxsentinel.catalog import Catalog
    from rxsentinel.pipelines.ingest import ingest

    async def fake_get(_, url, params=None):
        if url.endswith("version.json"):
            return {"version": "test-only"}
        return {
            "results": [
                {
                    "product_ndc": "TEST-NOT-REAL",
                    "generic_name": "synthetic unresolved product",
                    "dosage_form": "TEST TABLET",
                    "active_ingredients": [{"name": "synthetic ingredient", "strength": "test"}],
                }
            ]
        }

    async def fake_resolve(*_):
        return {"rxcui": None, "normalized_name": None, "normalization_source": None}, {
            "reason": "test-only"
        }

    monkeypatch.setattr("rxsentinel.pipelines.ingest.get_json", fake_get)
    monkeypatch.setattr("rxsentinel.pipelines.ingest.resolve_ingredient", fake_resolve)
    manifest = await ingest(tmp_path, ["test-query-a", "test-query-b"], 1)
    assert manifest["product_count"] == 1
    assert manifest["fully_normalized_product_count"] == 0
    snapshot = tmp_path / "snapshots" / manifest["snapshot_id"]
    raw = (snapshot / "raw.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["raw_sha256"]
    assert json.loads((snapshot / "manifest.json").read_text())["rxnorm_version"] == {
        "version": "test-only"
    }
    assert len(Catalog(tmp_path / "catalog.sqlite").all()) == 1


@pytest.mark.anyio
async def test_failed_import_does_not_mutate_catalog(tmp_path, monkeypatch):
    from rxsentinel.pipelines.ingest import ingest

    async def fail(*_):
        raise httpx.ConnectError("simulated provider outage")

    monkeypatch.setattr("rxsentinel.pipelines.ingest.get_json", fail)
    with pytest.raises(httpx.ConnectError):
        await ingest(tmp_path, ["test-query"], 1)
    assert not (tmp_path / "catalog.sqlite").exists()
    assert not (tmp_path / "snapshots").exists()
