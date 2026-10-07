import hashlib
import json

import pytest
from pydantic import ValidationError
from rxsentinel.catalog import Catalog
from rxsentinel.matching_evaluation import MatchingBenchmark, run_benchmark
from rxsentinel.pipelines.evaluate_matching import main, make_demo


@pytest.fixture
def benchmark(tmp_path):
    root = tmp_path / "synthetic-dataset"
    manifest = make_demo(root)
    catalog = Catalog(f"sqlite+pysqlite:///{(root / 'catalog.sqlite').as_posix()}")
    yield manifest, catalog, root
    catalog.store.engine.dispose()


def rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_benchmark_measures_known_misses_and_unknown_collisions(benchmark, tmp_path):
    manifest, catalog, root = benchmark
    output = tmp_path / "results"
    result = run_benchmark(manifest, output, catalog, root)
    metrics = result["metrics"]
    assert metrics["known_count"] == 4 and metrics["unknown_count"] == 2
    assert metrics["known_top1_recall"] == metrics["known_top3_recall"] == 0.5
    assert metrics["known_abstention_count"] == 1
    assert metrics["known_wrong_first_candidate_count"] == 1
    assert metrics["unknown_candidate_return_count"] == 1
    assert metrics["unknown_candidate_return_rate"] == 0.5
    assert metrics["capture_session_count"] == 3
    assert metrics["accepted_identity_count"] == 0
    assert metrics["accepted_identity_error_rate"] is None
    assert metrics["latency_p95_ms"] >= metrics["latency_median_ms"] >= 0
    assert result["conditions"]["transcription-error"]["known_wrong_first_candidate_count"] == 1
    assert result["appearance_coverage"]["local:synthetic-alpha"]["known_count"] == 3
    assert not result["m4_exit_gate_satisfied"] and not result["photo_identification_available"]
    assert (output / "manifest.json").read_bytes() == manifest.read_bytes()
    for row in result["cases"]:
        path = output / row["case_id"] / "source.png"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["input_sha256"]
        assert (output / row["match_report"]).exists()
        report = json.loads((path.parent / "report.json").read_text())
        assert report["reference_snapshot_sha256"] == result["reference_snapshot_sha256"]
        assert not report["query_identity_confirmed"]
    with pytest.raises(ValueError, match="already exists"):
        run_benchmark(manifest, output, catalog, root)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_photo",
        "duplicate_id",
        "limit",
        "missing_truth",
        "unknown_evidence",
        "transcription",
        "partition",
        "scope",
    ],
)
def test_manifest_rejects_invalid_protocol(benchmark, case):
    path, _, _ = benchmark
    value = json.loads(path.read_text())
    if case == "duplicate_photo":
        value["cases"][1]["input_sha256"] = value["cases"][0]["input_sha256"]
    elif case == "duplicate_id":
        value["cases"][1]["case_id"] = value["cases"][0]["case_id"]
    elif case == "limit":
        value["cases"][0]["query"]["limit"] = 1
    elif case == "missing_truth":
        value["cases"][0].pop("expected_appearance_id")
    elif case == "unknown_evidence":
        value["cases"][-1]["evidence"] = []
    elif case == "transcription":
        value["observations_transcribed_from_query"] = False
    elif case == "partition":
        value["partition"] = "reference"
    else:
        value["scope"] = "reviewed-collection"
    with pytest.raises(ValidationError):
        MatchingBenchmark.model_validate(value)


@pytest.mark.parametrize(
    "case",
    [
        "snapshot",
        "truth",
        "reference_query",
        "source_hash",
        "file_change",
        "unknown_session",
        "evidence",
        "path_escape",
        "implementation",
    ],
)
def test_invalid_data_does_not_publish_a_partial_benchmark(benchmark, tmp_path, case):
    manifest, catalog, root = benchmark
    if case == "snapshot":
        rewrite(manifest, lambda m: m.update(reference_snapshot_sha256="0" * 64))
    elif case == "truth":
        rewrite(
            manifest, lambda m: m["cases"][0].update(expected_appearance_id="local:synthetic-beta")
        )
    elif case == "reference_query":
        asset = next(
            a
            for a in catalog.store.assets()
            if a.partition == "reference" and a.appearance_id == "local:synthetic-alpha"
        )
        rewrite(
            manifest,
            lambda m: m["cases"][0].update(
                asset_id=asset.asset_id,
                capture_session=asset.capture_session,
                input_sha256=asset.sha256,
            ),
        )
    elif case == "source_hash":
        rewrite(manifest, lambda m: m["cases"][0].update(input_sha256="0" * 64))
    elif case == "file_change":
        (root / "unknown-collision.png").write_bytes(b"changed image")
    elif case == "unknown_session":
        rewrite(
            manifest, lambda m: m["cases"][-1].update(capture_session="synthetic-alpha-reference")
        )
    elif case == "evidence":
        rewrite(manifest, lambda m: m["cases"][-1]["evidence"][0].update(sha256="0" * 64))
    elif case == "implementation":
        rewrite(manifest, lambda m: m.update(matcher_code_sha256="0" * 64))
    else:
        rewrite(manifest, lambda m: m["cases"][-1].update(image="../outside.png"))
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        run_benchmark(manifest, output, catalog, root)
    assert not output.exists()


@pytest.mark.parametrize("budget", ["MAX_BENCHMARK_BYTES", "MAX_BENCHMARK_PIXELS"])
def test_resource_budget_is_enforced_before_publication(benchmark, tmp_path, monkeypatch, budget):
    manifest, catalog, root = benchmark
    monkeypatch.setattr(f"rxsentinel.matching_evaluation.{budget}", 1)
    with pytest.raises(ValueError, match="budget"):
        run_benchmark(manifest, tmp_path / "failed", catalog, root)
    assert not (tmp_path / "failed").exists()


def test_catalog_change_after_a_case_aborts_whole_run(benchmark, tmp_path, monkeypatch):
    from rxsentinel import matching_evaluation

    manifest, catalog, root = benchmark
    original = matching_evaluation.match_observations

    def changing(catalog, directory, query):
        result = original(catalog, directory, query)
        if query.imprint_front:
            asset = next(a for a in catalog.store.assets() if a.partition == "reference")
            (directory / asset.local_path).write_bytes(b"changed during benchmark")
        return result

    monkeypatch.setattr(matching_evaluation, "match_observations", changing)
    with pytest.raises(ValueError):
        run_benchmark(manifest, tmp_path / "failed", catalog, root)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".rxsentinel-match-evaluation-*"))


@pytest.mark.parametrize("kind", ["known", "unknown"])
def test_missing_class_denominators_are_null(benchmark, tmp_path, kind):
    manifest, catalog, root = benchmark
    rewrite(manifest, lambda m: m.update(cases=[c for c in m["cases"] if c["kind"] == kind]))
    result = run_benchmark(manifest, tmp_path / "known-only", catalog, root)
    if kind == "known":
        assert result["metrics"]["unknown_count"] == 0
        assert result["metrics"]["unknown_candidate_return_rate"] is None
    else:
        assert result["metrics"]["known_count"] == 0
        assert result["metrics"]["known_top1_recall"] is None


def test_correct_second_candidate_counts_for_top3_but_not_top1(benchmark, tmp_path):
    manifest, catalog, root = benchmark
    rewrite(manifest, lambda m: m["cases"][1]["query"].pop("imprint_back"))
    result = run_benchmark(manifest, tmp_path / "rank-two", catalog, root)
    assert result["metrics"]["known_top1_correct_count"] == 1
    assert result["metrics"]["known_top3_contains_truth_count"] == 2


def test_synthetic_scope_rejects_non_synthetic_catalog_products(benchmark, tmp_path):
    manifest, catalog, root = benchmark
    product = catalog.store.all()[0]
    catalog.store.import_products(
        [product.model_copy(update={"product_id": "non-synthetic-fixture"})]
    )
    with pytest.raises(ValueError, match="isolated synthetic"):
        run_benchmark(manifest, tmp_path / "failed-scope", catalog, root)


def test_reviewed_protocol_and_empty_gallery_gate(benchmark, tmp_path):
    from rxsentinel.matching import snapshot

    manifest, catalog, root = benchmark
    rewrite(manifest, lambda m: (m.update(scope="reviewed-collection"), m.pop("synthetic_catalog")))
    result = run_benchmark(manifest, tmp_path / "reviewed-fixture", catalog, root)
    assert result["scope"] == "reviewed-collection"
    # This is still an isolated synthetic collection, not production evidence.
    for appearance in catalog.store.appearances():
        catalog.store.put_appearance(appearance.model_copy(update={"stale": True}))
    rewrite(manifest, lambda m: m.update(reference_snapshot_sha256=snapshot(catalog, root)[3]))
    with pytest.raises(ValueError, match="nonempty eligible"):
        run_benchmark(manifest, tmp_path / "no-gallery", catalog, root)


def test_html_escaping_and_evidence_retention(benchmark, tmp_path):
    manifest, catalog, root = benchmark
    rewrite(manifest, lambda m: m.update(dataset_id="<script>unsafe</script>"))
    output = tmp_path / "escaped"
    run_benchmark(manifest, output, catalog, root)
    assert "<script>" not in (output / "report.html").read_text()
    spec = MatchingBenchmark.model_validate_json(manifest.read_text())
    for record in spec.cases[-1].evidence:
        assert (output / "unknown-collision" / f"evidence-{record.sha256}.bin").read_bytes() == (
            root / record.local_path
        ).read_bytes()


def test_synthetic_cli_does_not_use_configured_database(benchmark, tmp_path, monkeypatch, capsys):
    manifest, _, _ = benchmark
    monkeypatch.setattr(
        "rxsentinel.pipelines.evaluate_matching.configured_catalog",
        lambda _: pytest.fail("Synthetic run must not open production database"),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark", "run", "--manifest", str(manifest), "--output-dir", str(tmp_path / "cli")],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["scope"] == "synthetic"


def test_synthetic_catalog_cannot_escape_manifest_directory(benchmark, tmp_path, monkeypatch):
    manifest, _, _ = benchmark
    rewrite(manifest, lambda m: m.update(synthetic_catalog="../production.sqlite"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "run",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "escaped"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert not (tmp_path / "escaped").exists()
