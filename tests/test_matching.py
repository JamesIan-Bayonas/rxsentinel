import json
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from rxsentinel.api import create_app
from rxsentinel.catalog import Catalog
from rxsentinel.matching import MatchQuery, imprint_matches, match_observations
from rxsentinel.pipelines.collection import intake, review, statuses
from rxsentinel.pipelines.match import main, write_report
from sqlalchemy.exc import OperationalError
from test_collection import approval, imported
from test_collection import collection as collection

RULES = Path(__file__).resolve().parents[1] / "data" / "rules" / "interactions.json"


def ready(collection, changes=None):
    store, directory, path, manifest, _ = collection
    if changes:
        manifest.update(changes)
        path.write_text(json.dumps(manifest))
    imported(collection)
    review(store, directory, approval(collection))
    return Catalog(str(store.engine.url)), directory


def test_metadata_only_catalog_returns_unknown_without_creating_database(tmp_path):
    path = tmp_path / "missing.sqlite"
    result = match_observations(Catalog(path), tmp_path, MatchQuery(imprint_front="TEST"))
    assert result.status == "unknown" and result.reason == "no_eligible_references"
    assert not result.candidates and not result.query_identity_confirmed
    assert not path.exists()


def test_reviewed_reference_match_has_traceable_evidence_without_confirming_query(collection):
    catalog, directory = ready(collection)
    result = match_observations(
        catalog,
        directory,
        MatchQuery(
            imprint_front=" test ",
            imprint_back="123",
            color="synthetic-blue",
            shape="synthetic-round",
        ),
    )
    assert result.status == "candidates" and result.total_candidate_count == 1
    candidate = result.candidates[0]
    assert candidate.evidence_points == 22
    assert candidate.products[0].product_id == "synthetic-photo-test"
    assert len(candidate.reference_images) == 2
    assert {a.side for a in candidate.reference_images} == {"front", "back"}
    assert all(a.reuse_reviewer and a.sha256 for a in candidate.reference_images)
    assert candidate.collection_fingerprint == statuses(catalog.store, directory)[0]["fingerprint"]
    assert candidate.requires_human_review and not result.query_identity_confirmed
    assert not result.photo_identification_available


def test_both_sides_can_be_swapped_but_not_replaced_or_repeated(collection):
    catalog, directory = ready(collection)
    swapped = match_observations(
        catalog, directory, MatchQuery(imprint_front="123", imprint_back="TEST")
    )
    assert swapped.status == "candidates"
    for back in ("WRONG", "TEST"):
        result = match_observations(
            catalog, directory, MatchQuery(imprint_front="TEST", imprint_back=back)
        )
        assert result.reason == "no_exact_imprint_match"


@pytest.mark.parametrize(
    "query,reason",
    [
        ({"color": "synthetic-blue", "shape": "synthetic-round"}, "imprint_required"),
        ({"imprint_front": "TES?"}, "unsupported_imprint"),
        ({"imprint_front": "TEST."}, "no_exact_imprint_match"),
        ({"imprint_front": "ТЕST"}, "unsupported_imprint"),
        ({"imprint_front": "1234"}, "no_exact_imprint_match"),
    ],
)
def test_abstention_preserves_unreadable_and_unmatched_queries(collection, query, reason):
    catalog, directory = ready(collection)
    result = match_observations(catalog, directory, MatchQuery(**query))
    assert result.status == "unknown" and result.reason == reason
    assert result.candidates == []


def test_lookalike_characters_are_not_corrected(collection):
    catalog, directory = ready(collection, {"imprint_front": "O1"})
    assert (
        match_observations(catalog, directory, MatchQuery(imprint_front="O1")).status
        == "candidates"
    )
    for imprint in ("01", "OI", "O 1"):
        assert (
            match_observations(catalog, directory, MatchQuery(imprint_front=imprint)).status
            == "unknown"
        )


def test_conflicting_optional_observations_are_visible_without_hard_filtering(collection):
    catalog, directory = ready(collection)
    result = match_observations(
        catalog, directory, MatchQuery(imprint_front="TEST", color="white", shape="oval")
    )
    candidate = result.candidates[0]
    assert candidate.evidence_points == 10
    assert candidate.comparisons == {"color": "mismatch", "shape": "mismatch"}
    assert "color_mismatch" in candidate.uncertainties
    assert "only_one_imprint_observed" in candidate.uncertainties


def test_blank_imprint_does_not_become_a_color_only_identity(collection):
    catalog, directory = ready(
        collection,
        {
            "imprint_front": None,
            "imprint_back": None,
            "blank_imprint_verified": True,
        },
    )
    result = match_observations(catalog, directory, MatchQuery(color="synthetic-blue"))
    assert result.reason == "imprint_required" and result.eligible_appearance_count == 1


def test_unassigned_imprint_only_supports_one_observed_side(collection):
    catalog, _ = ready(collection)
    appearance = catalog.store.appearances()[0].model_copy(
        update={
            "imprint_front": None,
            "imprint_back": None,
            "imprint_unassigned": "TEST",
        }
    )
    assert imprint_matches(MatchQuery(imprint_back="TEST"), appearance) == ["unassigned"]
    assert imprint_matches(MatchQuery(imprint_front="TEST", imprint_back="TEST"), appearance) == []


@pytest.mark.parametrize(
    "case", ["pending", "revoked", "expired", "file", "product", "reference_use"]
)
def test_ineligible_collections_cannot_become_candidates(collection, product_factory, case):
    store, directory, _, _, now = collection
    if case == "pending":
        imported(collection)
        catalog = Catalog(str(store.engine.url))
    else:
        catalog, directory = ready(collection)
    if case == "revoked":
        review(store, directory, approval(collection, identity="reject", reuse="pending"))
    elif case == "expired":
        appearance = store.appearances()[0]
        store.put_appearance(appearance.model_copy(update={"valid_until": now - timedelta(days=1)}))
    elif case == "file":
        asset = next(a for a in store.assets() if a.partition == "reference")
        (directory / asset.local_path).write_bytes(b"changed")
    elif case == "product":
        product = store.all()[0]
        store.import_products([product.model_copy(update={"snapshot_id": "changed-source"})])
    elif case == "reference_use":
        for asset in store.assets():
            store.put_asset(asset.model_copy(update={"intended_use": "research_evaluation"}))
    result = match_observations(catalog, directory, MatchQuery(imprint_front="TEST"))
    assert result.status == "unknown" and result.reason == "no_eligible_references"
    assert result.excluded_appearance_count == 1


def test_truncation_does_not_hide_ambiguity_and_tie_order_is_stable(collection):
    catalog, directory = ready(collection)
    store, _, _, manifest, now = collection
    second = json.loads(json.dumps(manifest))
    second["appearance_id"] = "local:synthetic-002"
    second["color"] = "different-color"
    for number, photo in enumerate(second["photos"]):
        photo["file"] = f"second-{number}.png"
        photo["capture_session"] = f"second-{photo['partition']}"
        Image.new("RGB", (50 + number, 30), (110 + number, 80, 120)).save(directory / photo["file"])
    path = directory / "second.json"
    path.write_text(json.dumps(second))
    intake(store, directory, path)
    state = next(
        s for s in statuses(store, directory) if s["appearance_id"] == second["appearance_id"]
    )
    decision = approval(collection).model_copy(
        update={
            "appearance_id": second["appearance_id"],
            "expected_fingerprint": state["fingerprint"],
            "reuse_asset_ids": [a["asset_id"] for a in state["assets"]],
        }
    )
    review(store, directory, decision)
    result = match_observations(catalog, directory, MatchQuery(imprint_front="TEST", limit=1))
    assert result.total_candidate_count == 2 and result.truncated and result.ambiguous
    assert result.reason == "ambiguous_matches"
    assert result.candidates[0].appearance_id == "local:synthetic-001"
    preferred = match_observations(
        catalog, directory, MatchQuery(imprint_front="TEST", color="different-color")
    )
    assert preferred.candidates[0].appearance_id == "local:synthetic-002"
    assert preferred.ambiguous


def test_reference_change_between_snapshots_abstains(collection, monkeypatch):
    from rxsentinel import matching

    catalog, directory = ready(collection)
    original = matching.snapshot
    calls = 0

    def changing(catalog, directory):
        nonlocal calls
        calls += 1
        if calls == 2:
            asset = next(a for a in catalog.store.assets() if a.partition == "reference")
            (directory / asset.local_path).write_bytes(b"changed")
        return original(catalog, directory)

    monkeypatch.setattr(matching, "snapshot", changing)
    result = match_observations(catalog, directory, MatchQuery(imprint_front="TEST"))
    assert result.reason == "references_changed_during_search" and not result.candidates


def test_api_candidates_validation_and_database_error_contract(collection, monkeypatch):
    catalog, directory = ready(collection)
    monkeypatch.setenv("RXSENTINEL_DATA_DIR", str(directory))
    monkeypatch.setattr("rxsentinel.api.configured_catalog", lambda _: catalog)
    client = TestClient(create_app(rules_path=RULES))
    result = client.post("/api/v1/appearance-matches", json={"imprint_front": "TEST"})
    assert result.status_code == 200 and result.json()["status"] == "candidates"
    for payload in (
        {"identity_confirmed": True},
        {"limit": "3"},
        {"limit": 100},
        {"imprint_front": ""},
    ):
        assert client.post("/api/v1/appearance-matches", json=payload).status_code == 422

    def unavailable():
        raise OperationalError("SQL", {"password": "secret-test-value"}, Exception("down"))

    monkeypatch.setattr(catalog.store, "collection_transaction", unavailable)
    response = client.post("/api/v1/appearance-matches", json={"imprint_front": "TEST"})
    assert response.status_code == 503 and "secret-test-value" not in response.text


def test_cli_and_immutable_escaped_report(collection, tmp_path, monkeypatch, capsys):
    catalog, directory = ready(collection)
    query = tmp_path / "query.json"
    query.write_text(json.dumps({"imprint_front": "TEST"}))
    output = tmp_path / "matching"
    monkeypatch.setattr("rxsentinel.pipelines.match.configured_catalog", lambda _: catalog)
    monkeypatch.setattr(
        "sys.argv",
        ["match", "--query", str(query), "--data-dir", str(directory), "--output-dir", str(output)],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "candidates"
    report = match_observations(catalog, directory, MatchQuery(imprint_front="TEST"))
    with pytest.raises(ValueError, match="already exists"):
        write_report(report, output)
    report.candidates[0].color = "<script>unsafe</script>"
    other = tmp_path / "escaped"
    write_report(report, other)
    assert "<script>" not in (other / "report.html").read_text()


def test_request_strictness():
    with pytest.raises(ValidationError):
        MatchQuery(imprint_front="TEST", identity_confirmed=True)
