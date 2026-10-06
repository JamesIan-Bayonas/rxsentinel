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
        return httpx.Response(200, json={"relatedGroup": {"conceptGroup": [{
            "tty": "IN", "conceptProperties": [{"rxcui": "6809", "name": "metformin"}]
        }]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        mapping, raw = await resolve_ingredient(client, "metformin hydrochloride", datetime.now(UTC))
    assert mapping["rxcui"] == "6809"
    assert "related" in raw


@pytest.mark.anyio
async def test_ambiguous_lookup_is_not_guessed():
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"idGroup": {"rxnormId": ["1", "2"]}})
    )) as client:
        mapping, _ = await resolve_ingredient(client, "ambiguous", datetime.now(UTC))
    assert mapping["rxcui"] is None


@pytest.mark.anyio
async def test_nontransient_failure_is_not_hidden():
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(403, json={"error": "forbidden"})
    )) as client, pytest.raises(httpx.HTTPStatusError):
        await get_json(client, "https://example.org")


def test_metadata_only_catalog_cannot_pass_visual_readiness(catalog):
    summary = generate_readiness(catalog.path.parent)
    assert summary["product_count"] == 6
    assert summary["appearance_metadata_ready_count"] == 0
    assert summary["visual_identification_ready"] is False
    assert "reference_images" in (catalog.path.parent / "readiness.csv").read_text()
