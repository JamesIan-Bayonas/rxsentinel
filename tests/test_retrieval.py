import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw
from rxsentinel.gallery import export_gallery
from rxsentinel.pipelines.retrieve import main, write_report
from rxsentinel.retrieval import (
    VectorManifest,
    build_vectors,
    index_id,
    retrieve_photo,
    unit_vector,
    validate_vectors,
)
from rxsentinel.vision.embedding import (
    EncoderSpec,
    ResNet18Encoder,
    implementation_sha256,
    weights_bytes,
)
from test_matching_evaluation import benchmark as benchmark


class FixtureEncoder:
    """Test seam only; the production CLI always constructs the pinned pretrained model."""

    def __init__(self):
        self.spec = EncoderSpec(library_versions={}, implementation_sha256=implementation_sha256())
        self.vector = np.zeros(512, dtype=np.float32)
        self.vector[0] = 1

    def encode(self, image):
        assert image.shape == (224, 224, 3) and image.dtype == np.uint8
        return self.vector.copy()


def photo(*, blank=False, multiple=False):
    image = Image.new("RGB", (400, 300), (40, 55, 70))
    if not blank:
        drawing = ImageDraw.Draw(image)
        drawing.ellipse((100, 90, 240, 190), fill=(180, 170, 140))
        if multiple:
            drawing.ellipse((270, 100, 330, 180), fill=(180, 170, 140))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def vector_gallery(benchmark, tmp_path):
    _, catalog, data = benchmark
    gallery = tmp_path / "gallery"
    export_gallery(catalog, data, gallery, scope="synthetic")
    encoder = FixtureEncoder()
    directory = tmp_path / "vectors"
    manifest = build_vectors(gallery, directory, encoder, catalog=catalog, data_dir=data)
    return catalog, data, gallery, encoder, directory, manifest


def query(fixture, **kwargs):
    catalog, data, gallery, encoder, directory, _ = fixture
    content = kwargs.pop("content", photo())
    return retrieve_photo(
        content, directory, gallery, encoder, catalog=catalog, data_dir=data, **kwargs
    )


def rewrite_index(directory, *, mutation=None, vectors=None):
    manifest = VectorManifest.model_validate_json((directory / "manifest.json").read_text())
    if vectors is not None:
        buffer = io.BytesIO()
        np.save(buffer, vectors, allow_pickle=False)
        content = buffer.getvalue()
        (directory / "vectors.npy").write_bytes(content)
        manifest.vectors_sha256 = hashlib.sha256(content).hexdigest()
    if mutation:
        mutation(manifest)
    manifest.index_id = index_id(manifest)
    (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def test_build_and_query_are_grounded_unconfirmed_and_ties_survive_limit(vector_gallery, tmp_path):
    catalog, data, gallery, encoder, directory, manifest = vector_gallery
    assert len(manifest.rows) == 4
    loaded, spec, vectors = validate_vectors(
        directory, gallery, encoder, catalog=catalog, data_dir=data
    )
    assert loaded.index_id == manifest.index_id
    assert vectors.shape == (4, 512)
    assert all(r.processing.status == "processed" for r in loaded.rows)
    assert {r.asset_id for r in loaded.rows} == {a.asset_id for a in spec.reference_assets}
    result = query(vector_gallery, limit=1)
    assert result.status == "candidates" and result.scope == "synthetic"
    assert result.total_candidate_count == result.tied_first_candidate_count == 2
    assert len(result.candidates) == 1
    assert result.candidates[0].appearance_id == "local:synthetic-alpha"
    assert result.candidates[0].products[0].product_id == "synthetic:alpha"
    assert len(result.candidates[0].reference_images) == 2
    assert result.candidates[0].identity_reviewer
    assert result.current_source_verified
    assert not result.query_identity_confirmed and not result.thresholds_calibrated
    assert not result.photo_identification_available
    write_report(result, tmp_path / "report")
    saved = json.loads((tmp_path / "report/report.json").read_text())
    assert saved["index_id"] == manifest.index_id
    assert (tmp_path / "report/report.html").exists()
    with pytest.raises(ValueError, match="already exists"):
        write_report(result, tmp_path / "report")
    with pytest.raises(ValueError, match="already exists"):
        build_vectors(gallery, directory, encoder, catalog=catalog, data_dir=data)


def test_ranking_uses_best_side_and_returns_one_candidate_per_appearance(vector_gallery):
    _, _, _, _, directory, manifest = vector_gallery
    values = np.zeros((4, 512), np.float32)
    seen_alpha = False
    for i, row in enumerate(manifest.rows):
        if row.appearance_id == "local:synthetic-alpha":
            values[i, :2] = [1, 0] if seen_alpha else [0.6, 0.8]
            seen_alpha = True
        else:
            values[i, :2] = [0.8, 0.6]
    rewrite_index(directory, vectors=values)
    result = query(vector_gallery)
    assert [c.appearance_id for c in result.candidates] == [
        "local:synthetic-alpha",
        "local:synthetic-beta",
    ]
    assert result.candidates[0].cosine_similarity == 1
    assert result.candidates[1].cosine_similarity == pytest.approx(0.8)
    assert result.tied_first_candidate_count == 1


@pytest.mark.parametrize("multiple", [False, True])
def test_rejected_photo_returns_unknown_without_model_inference(vector_gallery, multiple):
    encoder = vector_gallery[3]
    encoder.encode = lambda _: pytest.fail("Rejected photo must not be embedded")
    result = query(vector_gallery, content=photo(blank=not multiple, multiple=multiple))
    assert result.status == "unknown" and result.reason == "photo_processing_rejected"
    assert result.processing.rejection_reasons
    assert not result.candidates and result.total_candidate_count == 0


def test_experimental_threshold_abstains_without_claiming_calibration(vector_gallery):
    encoder = vector_gallery[3]
    encoder.vector[:] = 0
    encoder.vector[1] = 1
    result = query(vector_gallery, minimum_similarity=0.5)
    assert result.status == "unknown"
    assert result.reason == "below_experimental_similarity_threshold"
    assert not result.thresholds_calibrated


@pytest.mark.parametrize(
    "case", ["checksum", "nan", "zero", "shape", "dtype", "norm", "rows", "config"]
)
def test_invalid_index_cannot_be_used_even_with_recomputed_metadata(vector_gallery, case):
    _, _, _, _, directory, manifest = vector_gallery
    values = np.zeros((4, 512), np.float32)
    values[:, 0] = 1
    if case == "checksum":
        (directory / "vectors.npy").write_bytes(b"bad data")
    elif case == "rows":
        rewrite_index(directory, mutation=lambda m: m.rows.reverse())
    elif case == "config":
        rewrite_index(
            directory, mutation=lambda m: setattr(m.rows[0].processing, "status", "rejected")
        )
    else:
        if case == "nan":
            values[0, 0] = np.nan
        elif case == "zero":
            values[0, :] = 0
        elif case == "shape":
            values = values[:, :2]
        elif case == "dtype":
            values = values.astype(np.float64)
        elif case == "norm":
            values *= 2
        rewrite_index(directory, vectors=values)
    with pytest.raises(ValueError):
        query(vector_gallery)


@pytest.mark.parametrize(
    "case", ["review", "file", "encoder", "gallery", "implementation", "manifest"]
)
def test_revocations_and_version_changes_fail_closed(vector_gallery, case):
    catalog, data, gallery, encoder, directory, _ = vector_gallery
    if case == "review":
        appearance = catalog.store.appearances()[0]
        catalog.store.put_appearance(appearance.model_copy(update={"stale": True}))
    elif case == "file":
        asset = catalog.store.assets()[0]
        (data / asset.local_path).write_bytes(b"modified")
    elif case == "encoder":
        encoder.spec.library_versions["torch"] = "changed"
    elif case == "gallery":
        rewrite_index(directory, mutation=lambda m: setattr(m, "gallery_id", "0" * 64))
    elif case == "implementation":
        rewrite_index(directory, mutation=lambda m: setattr(m, "retrieval_code_sha256", "0" * 64))
    else:
        value = json.loads((directory / "manifest.json").read_text())
        value["gallery_id"] = "0" * 64
        (directory / "manifest.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        query(vector_gallery)


def test_mid_inference_review_change_prevents_returning_candidates(vector_gallery):
    catalog, _, _, encoder, _, _ = vector_gallery
    original = encoder.encode

    def revoke(image):
        appearance = catalog.store.appearances()[0]
        catalog.store.put_appearance(appearance.model_copy(update={"stale": True}))
        return original(image)

    encoder.encode = revoke
    with pytest.raises(ValueError, match="changed since"):
        query(vector_gallery)


@pytest.mark.parametrize("vector", [np.zeros(512), np.ones(2), np.full(512, np.inf)])
def test_invalid_encoder_vectors_are_rejected(vector):
    with pytest.raises(ValueError):
        unit_vector(vector, 512)


def test_bad_reference_aborts_build_without_publishing(benchmark, tmp_path, monkeypatch):
    from rxsentinel import retrieval

    _, catalog, data = benchmark
    gallery = tmp_path / "gallery"
    export_gallery(catalog, data, gallery, scope="synthetic")
    original = retrieval.process_photo

    def reject(content, config):
        report, artifacts = original(content, config)
        report.status = "rejected"
        report.rejection_reasons.append("synthetic-test-rejection")
        return report, artifacts

    monkeypatch.setattr(retrieval, "process_photo", reject)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="Reference .* rejected"):
        build_vectors(gallery, output, FixtureEncoder(), catalog=catalog, data_dir=data)
    assert not output.exists()


def test_requires_current_source_and_bounds_query_parameters(vector_gallery, monkeypatch):
    _, _, gallery, encoder, directory, _ = vector_gallery
    with pytest.raises(ValueError, match="current source"):
        validate_vectors(directory, gallery, encoder, catalog=None, data_dir=None)
    for args in ({"limit": 0}, {"limit": True}, {"minimum_similarity": np.nan}):
        with pytest.raises(ValueError):
            query(vector_gallery, **args)
    monkeypatch.setattr("rxsentinel.retrieval.MAX_VECTORS_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        query(vector_gallery)


def test_cli_synthetic_query_ignores_production_configuration(
    vector_gallery, tmp_path, monkeypatch, capsys
):
    _, data, gallery, encoder, directory, _ = vector_gallery
    monkeypatch.setattr("rxsentinel.pipelines.retrieve.ResNet18Encoder", lambda _: encoder)
    monkeypatch.setattr(
        "rxsentinel.pipelines.retrieve.configured_catalog",
        lambda _: pytest.fail("Synthetic CLI must not access production configuration"),
    )
    path = tmp_path / "query.png"
    path.write_bytes(photo())
    monkeypatch.setattr(
        "sys.argv",
        [
            "retrieve",
            "query",
            "--gallery-dir",
            str(gallery),
            "--index-dir",
            str(directory),
            "--synthetic-directory",
            str(data),
            "--photo",
            str(path),
        ],
    )
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "candidates" and not result["query_identity_confirmed"]


def test_wrong_weight_bytes_are_rejected_without_unpickling(tmp_path):
    path = tmp_path / "untrusted.pth"
    path.write_bytes(b"not official weights")
    with pytest.raises(ValueError, match="checksum mismatch"):
        weights_bytes(path)


def test_vector_header_cannot_allocate_an_unbounded_array(vector_gallery, monkeypatch):
    _, _, _, _, directory, _ = vector_gallery
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {"descr": "<f4", "fortran_order": False, "shape": (10**12, 512)}
    )
    content = buffer.getvalue()
    (directory / "vectors.npy").write_bytes(content)
    rewrite_index(
        directory,
        mutation=lambda m: setattr(m, "vectors_sha256", hashlib.sha256(content).hexdigest()),
    )
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Must reject before allocating"))
    with pytest.raises(ValueError, match="Invalid vector array"):
        query(vector_gallery)


def test_query_reference_overlap_is_explicit(vector_gallery):
    _, _, gallery, _, _, _ = vector_gallery
    manifest = json.loads((gallery / "manifest.json").read_text())
    content = (gallery / manifest["reference_assets"][0]["local_path"]).read_bytes()
    result = query(vector_gallery, content=content)
    assert any("duplicates a reference" in warning for warning in result.warnings)


def test_mid_build_review_change_prevents_publication(benchmark, tmp_path):
    _, catalog, data = benchmark
    gallery = tmp_path / "gallery"
    export_gallery(catalog, data, gallery, scope="synthetic")
    encoder = FixtureEncoder()
    original = encoder.encode

    def revoke(image):
        appearance = catalog.store.appearances()[0]
        catalog.store.put_appearance(appearance.model_copy(update={"stale": True}))
        return original(image)

    encoder.encode = revoke
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        build_vectors(gallery, output, encoder, catalog=catalog, data_dir=data)
    assert not output.exists()
    assert not list(tmp_path.glob(".rxsentinel-vectors-*"))


def test_weights_download_failure_publishes_no_file(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from rxsentinel.vision import embedding

    class Response:
        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield b"untrusted download"

    @contextmanager
    def stream(*args, **kwargs):
        yield Response()

    monkeypatch.setattr(embedding.httpx, "stream", stream)
    target = tmp_path / "weights.pth"
    with pytest.raises(ValueError, match="checksum mismatch"):
        embedding.download_weights(target)
    assert not target.exists()
    assert not list(tmp_path.glob(".rxsentinel-weights-*"))


def test_real_encoder_smoke_is_optional_and_does_not_use_random_weights():
    from rxsentinel.vision.embedding import DEFAULT_WEIGHTS

    if not DEFAULT_WEIGHTS.is_file():
        pytest.skip("Pinned weights are an explicit optional download")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    encoder = ResNet18Encoder()
    image = np.full((224, 224, 3), 127, np.uint8)
    one, two = encoder.encode(image), encoder.encode(image)
    assert one.shape == (512,) and np.isfinite(one).all()
    assert np.array_equal(one, two)
    assert np.linalg.norm(unit_vector(one, 512)) == pytest.approx(1)
