import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta

import pytest
from rxsentinel.catalog import Catalog
from rxsentinel.gallery import ReferenceGallery, export_gallery, gallery_id, validate_gallery
from rxsentinel.pipelines.gallery import main
from test_matching_evaluation import benchmark as benchmark


def rehash(directory, mutate):
    path = directory / "manifest.json"
    spec = ReferenceGallery.model_validate_json(path.read_text())
    mutate(spec)
    spec.gallery_id = gallery_id(spec)
    path.write_text(spec.model_dump_json(indent=2))


def test_portable_export_preserves_bytes_provenance_and_excludes_queries(benchmark, tmp_path):
    _, catalog, data = benchmark
    output = tmp_path / "gallery"
    result = export_gallery(catalog, data, output, scope="synthetic")
    assert result["current_source_verified"]
    assert result["appearance_count"] == 2 and result["reference_image_count"] == 4
    assert not result["photo_identification_available"]
    spec = ReferenceGallery.model_validate_json((output / "manifest.json").read_text())
    for asset in spec.reference_assets:
        original = next(a for a in catalog.store.assets() if a.asset_id == asset.asset_id)
        assert (output / asset.local_path).read_bytes() == (data / original.local_path).read_bytes()
        assert asset.partition == "reference"
    assert all(not a.submitted_evidence for a in spec.appearances)
    for path, sha in spec.files.items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == sha
    assert spec.source_collection_fingerprints
    assert (output / "gallery.html").exists()
    moved = tmp_path / "moved-gallery"
    shutil.copytree(output, moved)
    offline = validate_gallery(moved)
    assert offline["gallery_id"] == result["gallery_id"]
    assert not offline["current_source_verified"]
    with pytest.raises(ValueError, match="already exists"):
        export_gallery(catalog, data, output, scope="synthetic")


def test_empty_or_unreviewed_catalog_cannot_export(benchmark, tmp_path):
    _, catalog, data = benchmark
    for appearance in catalog.store.appearances():
        catalog.store.put_appearance(appearance.model_copy(update={"stale": True}))
    with pytest.raises(ValueError, match="empty gallery"):
        export_gallery(catalog, data, tmp_path / "empty")
    assert not (tmp_path / "empty").exists()
    path = tmp_path / "missing.sqlite"
    with pytest.raises(ValueError, match="empty gallery"):
        export_gallery(Catalog(path), data, tmp_path / "legacy")
    assert not path.exists()


@pytest.mark.parametrize(
    "case", ["image", "evidence", "manifest", "expiry", "partition", "inventory", "path"]
)
def test_bundle_validation_rejects_tampering_even_with_rehashed_metadata(benchmark, tmp_path, case):
    _, catalog, data = benchmark
    output = tmp_path / "gallery"
    export_gallery(catalog, data, output, scope="synthetic")
    spec = ReferenceGallery.model_validate_json((output / "manifest.json").read_text())
    if case == "image":
        (output / spec.reference_assets[0].local_path).write_bytes(b"changed")
    elif case == "evidence":
        (output / spec.appearances[0].identity_evidence[0].local_path).write_bytes(b"changed")
    elif case == "manifest":
        raw = json.loads((output / "manifest.json").read_text())
        raw["products"][0]["generic_name"] = "changed"
        (output / "manifest.json").write_text(json.dumps(raw))
    elif case == "expiry":

        def expired(manifest):
            manifest.created_at = datetime.now(UTC) - timedelta(days=5)
            manifest.valid_until = datetime.now(UTC) - timedelta(days=1)

        rehash(output, expired)
    elif case == "partition":
        rehash(output, lambda m: setattr(m.reference_assets[0], "partition", "test"))
    elif case == "inventory":
        rehash(output, lambda m: m.files.update({"unexpected.txt": "0" * 64}))
    else:
        rehash(output, lambda m: setattr(m.reference_assets[0], "local_path", "../outside.png"))
    with pytest.raises(ValueError):
        validate_gallery(output)


def test_current_source_check_detects_revocation_but_offline_is_historical(benchmark, tmp_path):
    _, catalog, data = benchmark
    output = tmp_path / "gallery"
    export_gallery(catalog, data, output, scope="synthetic")
    appearance = catalog.store.appearances()[0]
    catalog.store.put_appearance(appearance.model_copy(update={"stale": True}))
    assert not validate_gallery(output)["current_source_verified"]
    with pytest.raises(ValueError, match="changed since"):
        validate_gallery(output, catalog=catalog, data_dir=data)


def test_recomputed_bundle_id_cannot_forge_current_source_metadata(benchmark, tmp_path):
    _, catalog, data = benchmark
    output = tmp_path / "gallery"
    export_gallery(catalog, data, output, scope="synthetic")
    rehash(output, lambda m: setattr(m.reference_assets[0], "reuse_basis", "Forged basis"))
    validate_gallery(output)  # Local checksums are not a cryptographic provenance signature.
    with pytest.raises(ValueError, match="reference metadata differs"):
        validate_gallery(output, catalog=catalog, data_dir=data)


def test_source_changed_during_copy_prevents_partial_publication(benchmark, tmp_path, monkeypatch):
    from rxsentinel import gallery

    _, catalog, data = benchmark
    original = gallery.bounded_read
    changed = False

    def mutate(path):
        nonlocal changed
        content = original(path)
        if path.is_relative_to(data / "assets") and not changed:
            changed = True
            path.write_bytes(b"changed during export")
        return content

    monkeypatch.setattr(gallery, "bounded_read", mutate)
    with pytest.raises(ValueError):
        export_gallery(catalog, data, tmp_path / "failed", scope="synthetic")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".rxsentinel-gallery-*"))


def test_gallery_budget_is_checked_before_publication(benchmark, tmp_path, monkeypatch):
    _, catalog, data = benchmark
    monkeypatch.setattr("rxsentinel.gallery.MAX_GALLERY_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        export_gallery(catalog, data, tmp_path / "oversized", scope="synthetic")
    assert not (tmp_path / "oversized").exists()


def test_cli_synthetic_and_offline_modes_ignore_production_config(
    benchmark, tmp_path, monkeypatch, capsys
):
    _, _, data = benchmark
    monkeypatch.setattr(
        "rxsentinel.pipelines.gallery.configured_catalog",
        lambda _: pytest.fail("Must not open production configuration"),
    )
    output = tmp_path / "gallery"
    monkeypatch.setattr(
        "sys.argv",
        ["gallery", "export", "--synthetic-directory", str(data), "--output-dir", str(output)],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["scope"] == "synthetic"
    monkeypatch.setattr("sys.argv", ["gallery", "validate", "--directory", str(output)])
    assert main() == 0
    assert not json.loads(capsys.readouterr().out)["current_source_verified"]
