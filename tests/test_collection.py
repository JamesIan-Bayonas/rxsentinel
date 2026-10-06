import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from rxsentinel.api import create_app
from rxsentinel.catalog import Catalog
from rxsentinel.collection_models import IntakeManifest, PhotoInput, ReviewInput
from rxsentinel.database.migrate import upgrade_schema
from rxsentinel.pipelines.collection import (
    amend,
    intake,
    make_template,
    review,
    select_collection,
    statuses,
)
from rxsentinel.pipelines.readiness import generate_readiness


@pytest.fixture
def collection(tmp_path, product_factory):
    catalog = Catalog(f"sqlite+pysqlite:///{(tmp_path / 'catalog.sqlite').as_posix()}")
    upgrade_schema(catalog.store.engine)
    product = product_factory("synthetic-photo-test", ["1191"])
    catalog.import_products([product])
    now = datetime.now(UTC)
    photos = []
    for number, (partition, side) in enumerate(
        [
            ("reference", "front"),
            ("reference", "back"),
            ("test", "front"),
        ]
    ):
        filename = f"{partition}-{side}.png"
        # Distinct synthetic fixtures; these are not drug images or identity evidence.
        Image.new("RGB", (40 + number, 30), (50 + number, 80, 120)).save(tmp_path / filename)
        photos.append(
            {
                "file": filename,
                "side": side,
                "partition": partition,
                "capture_session": f"synthetic-{partition}",
                "photo_origin": "own_photo",
                "photographer": "Synthetic fixture author",
                "captured_at": now.isoformat(),
            }
        )
    (tmp_path / "packaging.txt").write_text("Synthetic packaging fixture; not a real product.")
    (tmp_path / "ownership.txt").write_text("Synthetic test permission; no real clinical photos.")
    manifest = {
        "appearance_id": "local:synthetic-001",
        "product_id": product.product_id,
        "collection_version": "synthetic-only",
        "imprint_front": "TEST",
        "imprint_back": "123",
        "color": "synthetic-blue",
        "shape": "synthetic-round",
        "manufacturer": "Synthetic fixture",
        "photos": photos,
        "evidence": [
            {
                "evidence_id": "pack",
                "purpose": "packaging",
                "description": "Synthetic evidence",
                "file": "packaging.txt",
            },
            {
                "evidence_id": "owner",
                "purpose": "ownership",
                "description": "Synthetic permission",
                "file": "ownership.txt",
            },
        ],
    }
    path = tmp_path / "intake.json"
    path.write_text(json.dumps(manifest))
    yield catalog.store, tmp_path, path, manifest, now
    catalog.store.engine.dispose()


def imported(collection):
    store, directory, path, _, _ = collection
    intake(store, directory, path)
    return store, directory


def approval(collection, identity="approve", reuse="approve"):
    store, directory, _, manifest, now = collection
    state = statuses(store, directory)[0]
    return ReviewInput(
        appearance_id=manifest["appearance_id"],
        expected_fingerprint=state["fingerprint"],
        reviewer="Synthetic test reviewer",
        identity_decision=identity,
        reuse_decision=reuse,
        basis="Synthetic workflow test only; no real identity or permission assertion",
        identity_evidence_ids=["pack"],
        reuse_evidence_ids=["owner"],
        reuse_asset_ids=[a["asset_id"] for a in state["assets"]],
        valid_until=now + timedelta(days=90),
    )


def test_excessive_pixel_headers_are_rejected_before_intake(collection, monkeypatch):
    store, directory, path, _, _ = collection
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(ValueError, match="pixel limit"):
        intake(store, directory, path)
    assert store.appearances() == store.assets() == []


def test_intake_keeps_originals_and_starts_unreviewed(collection):
    store, directory, path, spec, _ = collection
    result = intake(store, directory, path)
    assert result["photos_imported"] == 3
    appearance = store.appearances()[0]
    assert appearance.identity_link_status == "unverified" and appearance.stale
    assert len(appearance.submitted_evidence) == 2
    assert all(
        a.reuse_status == "unverified" and a.intended_use == "inspection_only"
        for a in store.assets()
    )
    for asset, photo in zip(
        sorted(store.assets(), key=lambda a: a.width), spec["photos"], strict=True
    ):
        assert (directory / asset.local_path).read_bytes() == (
            directory / photo["file"]
        ).read_bytes()
        assert "identity context" in asset.source.name
    assert not statuses(store, directory)[0]["reference_ready"]


def test_approval_enables_references_and_independent_queries_with_immutable_history(collection):
    store, directory = imported(collection)
    event = review(store, directory, approval(collection))
    state = statuses(store, directory)[0]
    assert state["reference_ready"]
    assert state["evaluation_partitions"] == ["test"]
    assert all(a["eligible"] for a in state["assets"])
    assert store.collection_reviews() == [event]
    assert event.previous_fingerprint != event.resulting_fingerprint == state["fingerprint"]
    assert {a.intended_use for a in store.assets()} == {"research_reference", "research_evaluation"}


def test_identity_and_reuse_are_separate_decisions(collection):
    store, directory = imported(collection)
    review(store, directory, approval(collection, reuse="pending"))
    assert store.appearances()[0].identity_link_status == "verified"
    assert all(a.reuse_status == "unverified" for a in store.assets())
    assert not statuses(store, directory)[0]["reference_ready"]
    review(store, directory, approval(collection, identity="pending"))
    assert statuses(store, directory)[0]["reference_ready"]


@pytest.mark.parametrize("decision", ["identity", "reuse"])
def test_rejection_revokes_eligibility_and_preserves_prior_approval(collection, decision):
    store, directory = imported(collection)
    first = review(store, directory, approval(collection))
    request = approval(
        collection,
        identity="reject" if decision == "identity" else "pending",
        reuse="reject" if decision == "reuse" else "pending",
    )
    review(store, directory, request)
    assert not statuses(store, directory)[0]["reference_ready"]
    history = store.collection_reviews()
    assert len(history) == 2
    assert first in history
    assert first.appearance.identity_link_status == "verified"


def test_stale_fingerprint_cannot_overwrite_a_new_review(collection):
    store, directory = imported(collection)
    old_request = approval(collection)
    review(store, directory, old_request)
    current = statuses(store, directory)[0]
    with pytest.raises(ValueError, match="changed"):
        review(store, directory, old_request)
    assert statuses(store, directory)[0] == current
    assert len(store.collection_reviews()) == 1


@pytest.mark.parametrize("evidence_name", ["pack", "owner"])
def test_evidence_tampering_blocks_approval_without_partial_updates(collection, evidence_name):
    store, directory = imported(collection)
    request = approval(collection)
    evidence = next(
        e for e in store.appearances()[0].submitted_evidence if e.evidence_id == evidence_name
    )
    (directory / evidence.local_path).write_text("Changed evidence")
    with pytest.raises(ValueError, match="checksums"):
        review(store, directory, request)
    assert store.appearances()[0].identity_link_status == "unverified"
    assert store.collection_reviews() == []
    assert all(a.reuse_status == "unverified" for a in store.assets())


@pytest.mark.parametrize("target", ["photo", "packaging", "ownership"])
def test_tampering_after_approval_removes_readiness(collection, target):
    store, directory = imported(collection)
    review(store, directory, approval(collection))
    appearance = store.appearances()[0]
    record = (
        store.assets()[0]
        if target == "photo"
        else appearance.identity_evidence[0]
        if target == "packaging"
        else store.assets()[0].reuse_evidence[0]
    )
    (directory / record.local_path).write_bytes(b"changed")
    assert not statuses(store, directory)[0]["reference_ready"]


def test_changed_product_snapshot_invalidates_identity_binding(collection):
    store, directory = imported(collection)
    review(store, directory, approval(collection))
    original = store.all()[0]
    store.import_products(
        [original.model_copy(update={"snapshot_id": "different-source-snapshot"})]
    )
    state = statuses(store, directory)[0]
    assert not state["reference_ready"]
    assert "product_snapshot_changed" in state["missing_requirements"]


def test_expired_review_is_not_eligible(collection):
    store, directory = imported(collection)
    _, _, _, _, now = collection
    request = approval(collection).model_copy(update={"valid_until": now + timedelta(seconds=1)})
    review(store, directory, request, now=now)
    from rxsentinel.collection_support import collection_status

    appearance = store.appearances()[0]
    appearance.valid_until = now - timedelta(seconds=1)
    state = collection_status(directory, appearance, store.assets(), store.all(), store.assets())
    assert not state["reference_ready"]
    assert "identity_review_expired_or_missing" in state["missing_requirements"]


def test_capture_sessions_cannot_span_partitions_and_failed_intake_has_no_rows(collection):
    store, directory, path, spec, _ = collection
    spec["photos"][-1]["capture_session"] = spec["photos"][0]["capture_session"]
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="partitions"):
        intake(store, directory, path)
    assert store.appearances() == store.assets() == []


def test_duplicate_files_cannot_be_relabelled_as_independent_queries(collection):
    store, directory = imported(collection)
    spec = collection[3]
    photo = dict(spec["photos"][0], partition="validation", capture_session="different-session")
    batch = {
        "appearance_id": spec["appearance_id"],
        "expected_fingerprint": statuses(store, directory)[0]["fingerprint"],
        "photos": [photo],
    }
    path = directory / "add.json"
    path.write_text(json.dumps(batch))
    with pytest.raises(ValueError, match="duplicate"):
        intake(store, directory, path, add=True)
    assert len(store.assets()) == 3


def test_added_query_photos_require_identity_rereview(collection):
    store, directory = imported(collection)
    first = review(store, directory, approval(collection))
    spec = collection[3]
    Image.new("RGB", (50, 40), "red").save(directory / "validation.png")
    photo = dict(
        spec["photos"][0],
        file="validation.png",
        partition="validation",
        capture_session="synthetic-validation",
    )
    batch = {
        "appearance_id": spec["appearance_id"],
        "expected_fingerprint": statuses(store, directory)[0]["fingerprint"],
        "photos": [photo],
    }
    path = directory / "add.json"
    path.write_text(json.dumps(batch))
    intake(store, directory, path, add=True)
    assert len(store.assets()) == 4
    assert not statuses(store, directory)[0]["reference_ready"]
    assert store.collection_reviews() == [first]
    review(store, directory, approval(collection))
    state = statuses(store, directory)[0]
    assert state["reference_ready"]
    assert state["evaluation_partitions"] == ["test", "validation"]


def test_external_photos_need_license_evidence(collection):
    store, directory, path, spec, _ = collection
    spec["photos"][0]["photo_origin"] = "external"
    path.write_text(json.dumps(spec))
    intake(store, directory, path)
    with pytest.raises(ValueError, match="license evidence"):
        review(store, directory, approval(collection))
    assert store.collection_reviews() == []
    assert store.appearances()[0].identity_link_status == "unverified"


def test_existing_collection_and_missing_product_cannot_be_silently_overwritten(collection):
    store, directory = imported(collection)
    path = collection[2]
    with pytest.raises(ValueError, match="already exists"):
        intake(store, directory, path)
    spec = dict(collection[3], appearance_id="local:other", product_id="missing")
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Product ID"):
        intake(store, directory, path)
    assert len(store.appearances()) == 1


def test_template_has_no_imported_photos_or_invented_review(collection):
    store, directory, _, spec, _ = collection
    output = directory / "starter"
    make_template(store, spec["product_id"], output, "local:starter")
    template = IntakeManifest.model_validate_json((output / "intake.json").read_bytes())
    assert template.imprint_front is template.color is None
    assert store.appearances() == store.assets() == []
    with pytest.raises(ValidationError):
        ReviewInput.model_validate_json((output / "review.example.json").read_bytes())
    with pytest.raises(ValueError, match="Missing"):
        intake(store, directory, output / "intake.json")


def test_collection_status_and_review_history_api(collection, monkeypatch):
    store, directory = imported(collection)
    review(store, directory, approval(collection))
    monkeypatch.setenv("RXSENTINEL_DATABASE_URL", str(store.engine.url))
    monkeypatch.setenv("RXSENTINEL_DATA_DIR", str(directory))
    rules_file = Path(__file__).resolve().parents[1] / "data/rules/interactions.json"
    with TestClient(create_app(rules_path=rules_file)) as client:
        response = client.get("/api/v1/catalog/collections")
        assert response.status_code == 200 and response.json()[0]["reference_ready"]
        history = client.get("/api/v1/catalog/collection-reviews").json()
        assert history[0]["reviewer"] == "Synthetic test reviewer"
        assert client.get("/api/v1/catalog/collections?appearance_id=missing").status_code == 404
        assert client.get("/api/v1/catalog/collection-reviews?limit=101").status_code == 422
        assert client.get("/health").json()["photo_identification_available"] is False


def test_readiness_reports_reference_and_evaluation_gates_without_claiming_identification(
    collection,
    monkeypatch,
):
    store, directory = imported(collection)
    review(store, directory, approval(collection))
    monkeypatch.setattr(
        "rxsentinel.pipelines.readiness.configured_catalog",
        lambda _: Catalog(str(store.engine.url)),
    )
    result = generate_readiness(directory)
    assert result["eligible_reference_appearance_count"] == 1
    assert result["appearance_metadata_ready_count"] == 1
    assert result["appearances_with_validation_and_test_photos"] == 0
    assert not result["reference_data_gate_ready"]
    assert not result["visual_identification_ready"]


def test_capture_and_review_times_must_be_valid(collection):
    photo = collection[3]["photos"][0]
    for value in (datetime.now(), datetime.now(UTC) + timedelta(days=1)):
        with pytest.raises(ValidationError):
            PhotoInput.model_validate(dict(photo, captured_at=value))
    store, directory = imported(collection)
    request = approval(collection).model_copy(
        update={"valid_until": collection[4] - timedelta(days=1)}
    )
    with pytest.raises(ValueError, match="expiry"):
        review(store, directory, request)
    assert store.collection_reviews() == []


def test_visual_inspection_has_local_photos_evidence_and_pending_draft(collection):
    from rxsentinel.collection_report import write_inspection

    store, directory = imported(collection)
    with store.engine.connect() as connection:
        appearance, assets, products, all_assets = select_collection(
            connection, "local:synthetic-001"
        )
    appearance.manufacturer = "<script>untrusted</script>"
    output = write_inspection(directory, appearance, assets, products, all_assets)
    markup = Path(output["inspection_report"]).read_text(encoding="utf-8")
    assert "&lt;script&gt;" in markup and "<script>untrusted</script>" not in markup
    assert "../assets/collections/" in markup and "../evidence/" in markup
    draft = json.loads(Path(output["pending_review_draft"]).read_text())
    assert draft["identity_decision"] == draft["reuse_decision"] == "pending"
    assert len(draft["reuse_asset_ids"]) == 3
    assert store.collection_reviews() == []
    with pytest.raises(ValidationError):
        ReviewInput.model_validate(draft)


def test_amendments_can_correct_characteristics_and_add_evidence_without_overwriting_history(
    collection,
):
    store, directory = imported(collection)
    first = review(store, directory, approval(collection))
    (directory / "license.txt").write_text("Synthetic additional license evidence")
    request = {
        "appearance_id": "local:synthetic-001",
        "expected_fingerprint": statuses(store, directory)[0]["fingerprint"],
        "reason": "Correct a synthetic characteristic after inspection",
        "changes": {"color": "synthetic-white"},
        "evidence": [
            {
                "evidence_id": "license",
                "purpose": "license",
                "file": "license.txt",
                "description": "Synthetic additional proof",
            }
        ],
    }
    path = directory / "amend.json"
    path.write_text(json.dumps(request))
    amend(store, directory, path)
    updated = store.appearances()[0]
    assert updated.color == "synthetic-white"
    assert updated.identity_link_status == "unverified"
    assert len(updated.submitted_evidence) == 3
    assert store.collection_reviews() == [first]
    assert not statuses(store, directory)[0]["reference_ready"]
    review(store, directory, approval(collection, reuse="pending"))
    assert statuses(store, directory)[0]["reference_ready"]
    with pytest.raises(ValueError, match="changed"):
        amend(store, directory, path)
