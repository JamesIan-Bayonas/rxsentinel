import hashlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from rxsentinel.api import create_app
from rxsentinel.catalog import Catalog, configured_catalog
from rxsentinel.database.config import database_url
from rxsentinel.database.migrate import upgrade_schema
from rxsentinel.database.tables import ingredients, product_ingredients
from rxsentinel.pipelines.appearances import appearance_candidate
from rxsentinel.safety import SafetyEngine
from rxsentinel.schemas import Appearance, AuditRequest, ImageAsset, MedicationEntry
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError, OperationalError


@pytest.fixture
def relational(tmp_path):
    catalog = Catalog(f"sqlite+pysqlite:///{(tmp_path / 'relational.sqlite').as_posix()}")
    upgrade_schema(catalog.store.engine)
    yield catalog
    catalog.store.engine.dispose()


def test_upgrade_is_repeatable_and_has_foreign_keys(relational):
    upgrade_schema(relational.store.engine)
    inspector = inspect(relational.store.engine)
    assert set(inspector.get_table_names()) == {
        "alembic_version",
        "products",
        "ingredients",
        "product_ingredients",
        "source_snapshots",
        "appearances",
        "appearance_products",
        "image_assets",
        "reference_checks",
        "collection_reviews",
    }
    assert len(inspector.get_foreign_keys("appearance_products")) == 2


def test_normalized_import_retains_source_payload_and_replaces_ingredient_rows(
    relational, product_factory
):
    first = product_factory("a", ["1191", "6809"])
    relational.import_products([first])
    assert relational.get("a") == first
    second = product_factory("a", ["6809"], unresolved=True)
    relational.import_products([second])
    relational.import_products([second])
    assert relational.all() == [second]
    with relational.store.engine.connect() as connection:
        rows = connection.execute(select(product_ingredients)).all()
        assert [row.rxcui for row in rows] == ["6809", None]
        assert len(connection.execute(select(ingredients)).all()) == 2


def test_relational_and_legacy_audits_are_identical(relational, catalog, rules):
    relational.import_products(catalog.all())
    request = AuditRequest(
        medications=[
            MedicationEntry(entry_id=str(i), product_id=p, identity_confirmed=True)
            for i, p in enumerate(["aspirin", "warfarin", "metformin-a", "metformin-b"])
        ]
    )
    assert SafetyEngine(relational, rules).audit(request) == SafetyEngine(catalog, rules).audit(
        request
    )


def test_existing_unversioned_sqlite_is_not_modified(catalog):
    engine_catalog = Catalog(f"sqlite+pysqlite:///{catalog.path.as_posix()}")
    before = catalog.all()
    with pytest.raises(ValueError, match="unversioned"):
        upgrade_schema(engine_catalog.store.engine)
    assert catalog.all() == before
    assert "alembic_version" not in inspect(engine_catalog.store.engine).get_table_names()
    engine_catalog.store.engine.dispose()


def test_environment_overrides_file_without_silent_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("RXSENTINEL_DATABASE_URL=sqlite+pysqlite:///:memory:\n")
    monkeypatch.delenv("RXSENTINEL_DATABASE_URL")
    assert database_url() == "sqlite+pysqlite:///:memory:"
    monkeypatch.setenv("RXSENTINEL_DATABASE_URL", "")
    assert database_url() is None
    assert configured_catalog(tmp_path).backend == "sqlite"


def test_database_failure_returns_503_without_credentials(catalog, monkeypatch, rules):
    from pathlib import Path

    rules_path = Path(__file__).resolve().parents[1] / "data/rules/interactions.json"

    def fail(_):
        raise OperationalError("SQL", {"password": "secret-test-value"}, Exception("unavailable"))

    monkeypatch.setattr(Catalog, "all", fail)
    response = TestClient(create_app(catalog.path, rules_path)).get("/health")
    assert response.status_code == 503
    assert "secret-test-value" not in response.text


def test_archived_match_is_unverified_and_imprint_side_is_not_guessed(product_factory):
    product = product_factory("a", ["1191"]).model_copy(update={"product_ndc": "12345-678"})
    record = {"id": "7", "product_code": "12345-678", "splimprint": "A;B", "has_image": "True"}
    candidate = appearance_candidate(record, product, product.source)
    assert candidate.identity_link_status == "unverified"
    assert candidate.stale
    assert candidate.imprint_front is None and candidate.imprint_back is None
    assert candidate.imprint_unassigned == "A;B"
    assert not candidate.blank_imprint_verified
    with pytest.raises(ValueError, match="exactly match"):
        appearance_candidate({**record, "product_code": "99999-999"}, product, product.source)


def test_appearance_batch_rolls_back_if_any_product_link_is_missing(relational, product_factory):
    product = product_factory("a", ["1191"])
    relational.import_products([product])
    valid = Appearance(
        appearance_id="one", product_ids=["a"], source=product.source, source_version="test-only"
    )
    invalid = valid.model_copy(update={"appearance_id": "two", "product_ids": ["missing"]})
    with pytest.raises(IntegrityError):
        relational.store.import_appearances([valid, invalid])
    assert relational.store.appearances() == []


def test_verified_identity_and_image_reuse_require_review_evidence(product_factory):
    product = product_factory("a", ["1191"])
    with pytest.raises(ValidationError, match="reviewer"):
        Appearance(
            appearance_id="one",
            product_ids=["a"],
            source=product.source,
            source_version="test-only",
            identity_link_status="verified",
        )
    with pytest.raises(ValidationError, match="Permitted reuse"):
        ImageAsset(
            asset_id="image",
            appearance_id="one",
            local_path="assets/test.png",
            side="unknown",
            sha256="a" * 64,
            source=product.source,
            reuse_status="permitted",
        )
    with pytest.raises(ValidationError, match="capture session"):
        ImageAsset(
            asset_id="image",
            appearance_id="one",
            local_path="assets/test.png",
            side="unknown",
            sha256="a" * 64,
            source=product.source,
            partition="test",
        )


def test_assets_require_an_existing_appearance(relational, product_factory):
    product = product_factory("a", ["1191"])
    asset = ImageAsset(
        asset_id="image",
        appearance_id="missing",
        local_path="assets/test.png",
        side="unknown",
        sha256="a" * 64,
        source=product.source,
    )
    with pytest.raises(IntegrityError):
        relational.store.put_asset(asset)
    assert relational.store.assets() == []


def test_source_snapshot_saved_with_product_import(relational, product_factory):
    from rxsentinel.database.tables import source_snapshots

    manifest = {"snapshot_id": "test-only", "raw_sha256": "a" * 64}
    relational.import_products([product_factory("a", ["1191"])], manifest)
    with relational.store.engine.connect() as connection:
        saved = connection.scalar(select(source_snapshots.c.manifest))
    assert json.loads(saved) == manifest


@pytest.mark.anyio
async def test_discovery_persists_traceable_candidates_without_inventing_assets(
    relational, product_factory, tmp_path, monkeypatch
):
    from rxsentinel.pipelines.appearances import discover

    product = product_factory("a", ["1191"]).model_copy(update={"product_ndc": "12345-678"})
    relational.import_products([product])
    record = {
        "id": "7",
        "product_code": "12345-678",
        "splimprint": "A;B",
        "has_image": "True",
        "image_source": "NLM",
        "splimage": "sample-only",
    }
    original_client = httpx.AsyncClient

    def handler(request):
        if request.url.path.startswith("/api/views/"):
            return httpx.Response(200, json={"name": "Archived test dataset", "license": None})
        assert "12345-678" in request.url.params["$where"]
        return httpx.Response(200, json=[record])

    monkeypatch.setattr("rxsentinel.pipelines.appearances.configured_catalog", lambda _: relational)
    monkeypatch.setattr(
        "rxsentinel.pipelines.appearances.httpx.AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = await discover(tmp_path)
    assert result["archived_appearance_candidates"] == 1
    assert result["candidates_with_image_flags"] == 1
    assert not result["visual_identification_ready"]
    assert result["reference_images_verified"] == 0
    candidate = relational.store.appearances()[0]
    assert candidate.identity_link_status == "unverified" and candidate.stale
    assert relational.store.assets() == []
    raw = (tmp_path / "source-discovery" / result["snapshot_id"] / "raw.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result["raw_sha256"]
