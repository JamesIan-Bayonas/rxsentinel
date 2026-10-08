import hashlib
import json

import pytest
from pydantic import ValidationError
from rxsentinel.catalog import Catalog
from rxsentinel.pipelines.evaluate_visual import main, make_visual_demo
from rxsentinel.visual_evaluation import VisualBenchmark, cutoff_rows, metrics, run_benchmark
from test_retrieval import FixtureEncoder


@pytest.fixture
def visual_dataset(tmp_path, monkeypatch):
    encoder = FixtureEncoder()
    monkeypatch.setattr("rxsentinel.pipelines.evaluate_visual.ResNet18Encoder", lambda _: encoder)
    root = tmp_path / "synthetic-visual"
    manifest = make_visual_demo(root)
    catalog = Catalog(f"sqlite+pysqlite:///{(root / 'catalog.sqlite').as_posix()}")
    yield manifest, catalog, root, encoder
    catalog.store.engine.dispose()


def run(dataset, output):
    path, catalog, root, encoder = dataset
    return run_benchmark(path, output, catalog, root, root / "gallery", root / "vectors", encoder)


def rewrite(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


def test_visual_benchmark_counts_misses_collisions_and_ties(visual_dataset, tmp_path):
    manifest, _, root, _ = visual_dataset
    result = run(visual_dataset, tmp_path / "report")
    values = result["metrics"]
    assert values["case_count"] == 6
    assert values["known_count"] == 4 and values["unknown_count"] == 2
    assert values["known_top1_recall"] == 0.25
    assert values["known_top3_recall"] == 0.5
    assert values["known_abstention_count"] == 2
    assert values["known_wrong_first_candidate_count"] == 1
    assert values["unknown_candidate_return_count"] == 1
    assert values["unknown_candidate_return_rate"] == 0.5
    assert values["processing_rejection_count"] == 3
    assert values["known_processing_rejection_count"] == 2
    assert values["unknown_processing_rejection_count"] == 1
    assert values["ambiguous_result_count"] == 3
    assert values["known_unambiguous_top1_count"] == 0
    assert values["accepted_identity_count"] == 0 and values["accepted_identity_error_rate"] is None
    assert values["latency_p95_ms"] >= values["latency_median_ms"] >= 0
    assert len(result["validation_cutoff_comparison"]) == 3
    assert result["gallery_appearance_count"] == result["query_appearance_count"] == 2
    assert not result["gallery_appearances_without_known_queries"]
    assert not result["thresholds_calibrated"] and not result["m4_exit_gate_satisfied"]
    assert result["conditions"]["missing-object"]["known_abstention_count"] == 1
    assert len(result["capture_sessions"]) == 3
    assert (tmp_path / "report/manifest.json").read_bytes() == manifest.read_bytes()
    for row in result["cases"]:
        directory = tmp_path / "report" / row["case_id"]
        assert (
            hashlib.sha256((directory / "source.png").read_bytes()).hexdigest()
            == row["input_sha256"]
        )
        saved = json.loads((directory / "report.json").read_text())
        assert saved["index_id"] == result["index_id"] and not saved["query_identity_confirmed"]
    assert (root / "manual-manifest.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        run(visual_dataset, tmp_path / "report")


def test_cutoff_comparison_preserves_rejected_cases_and_does_not_select_threshold(
    visual_dataset, tmp_path
):
    result = run(visual_dataset, tmp_path / "report")
    rows = result["cases"]
    for row in rows:
        row["candidate_scores"] = [0.7] * len(row["candidate_scores"])
    lowered = metrics(cutoff_rows(rows, 0.6))
    raised = metrics(cutoff_rows(rows, 0.8))
    assert lowered["case_count"] == raised["case_count"] == 6
    assert lowered["known_top3_recall"] == 0.5
    assert raised["known_top3_recall"] == 0
    assert raised["known_abstention_count"] == 4
    assert raised["unknown_candidate_return_count"] == 0
    assert raised["processing_rejection_count"] == 3
    assert result["minimum_similarity"] is None


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_photo",
        "duplicate_id",
        "no_truth",
        "no_asset",
        "unknown_review",
        "unknown_evidence",
        "blank_condition",
        "duplicate_condition",
        "test_sweep",
        "threshold_sweep",
        "nan_cutoff",
        "scope",
    ],
)
def test_manifest_rejects_invalid_protocol(visual_dataset, case):
    manifest = visual_dataset[0]
    value = json.loads(manifest.read_text())
    if case == "duplicate_photo":
        value["cases"][1]["input_sha256"] = value["cases"][0]["input_sha256"]
    elif case == "duplicate_id":
        value["cases"][1]["case_id"] = value["cases"][0]["case_id"]
    elif case == "no_truth":
        value["cases"][0].pop("expected_appearance_id")
    elif case == "no_asset":
        value["cases"][0].pop("asset_id")
    elif case == "unknown_review":
        value["cases"][-1].pop("reuse_reviewer")
    elif case == "unknown_evidence":
        value["cases"][-1]["evidence"] = []
    elif case == "blank_condition":
        value["cases"][0]["conditions"] = [" "]
    elif case == "duplicate_condition":
        value["cases"][0]["conditions"] = ["same", "same"]
    elif case == "test_sweep":
        value["partition"] = "test"
    elif case == "threshold_sweep":
        value["minimum_similarity"] = 0.8
    elif case == "nan_cutoff":
        value["validation_cutoffs"] = [float("nan")]
    else:
        value["scope"] = "reviewed-collection"
    with pytest.raises(ValidationError):
        VisualBenchmark.model_validate(value)


@pytest.mark.parametrize(
    "case",
    [
        "index",
        "gallery",
        "evaluator",
        "truth",
        "reference_query",
        "partition",
        "query_hash",
        "unknown_file",
        "unknown_session",
        "evidence",
        "path_escape",
        "synthetic_catalog",
    ],
)
def test_bad_data_never_publishes_partial_results(visual_dataset, tmp_path, case):
    manifest, catalog, root, _ = visual_dataset
    if case in {"index", "gallery", "evaluator"}:
        field = {
            "index": "index_id",
            "gallery": "gallery_id",
            "evaluator": "evaluator_code_sha256",
        }[case]
        rewrite(manifest, lambda m: m.update({field: "0" * 64}))
    elif case == "truth":
        rewrite(
            manifest, lambda m: m["cases"][0].update(expected_appearance_id="local:synthetic-beta")
        )
    elif case == "reference_query":
        asset = next(a for a in catalog.store.assets() if a.partition == "reference")
        rewrite(
            manifest,
            lambda m: m["cases"][0].update(
                asset_id=asset.asset_id,
                input_sha256=asset.sha256,
                capture_session=asset.capture_session,
            ),
        )
    elif case == "partition":
        rewrite(manifest, lambda m: m.update(partition="test", validation_cutoffs=[]))
    elif case == "query_hash":
        rewrite(manifest, lambda m: m["cases"][0].update(input_sha256="0" * 64))
    elif case == "unknown_file":
        (root / "unknown-collision.png").write_bytes(b"changed")
    elif case == "unknown_session":
        rewrite(
            manifest, lambda m: m["cases"][-1].update(capture_session="synthetic-alpha-reference")
        )
    elif case == "evidence":
        rewrite(manifest, lambda m: m["cases"][-1]["evidence"][0].update(sha256="0" * 64))
    elif case == "path_escape":
        rewrite(manifest, lambda m: m["cases"][-1].update(image="../outside.png"))
    else:
        rewrite(manifest, lambda m: m.update(synthetic_catalog="../outside.sqlite"))
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        run(visual_dataset, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".rxsentinel-visual-evaluation-*"))


@pytest.mark.parametrize("budget", ["MAX_EVALUATION_BYTES", "MAX_EVALUATION_PIXELS"])
def test_resource_budgets_apply_before_publication(visual_dataset, tmp_path, monkeypatch, budget):
    monkeypatch.setattr(f"rxsentinel.visual_evaluation.{budget}", 1)
    with pytest.raises(ValueError, match="budget"):
        run(visual_dataset, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


@pytest.mark.parametrize("case", ["evidence", "manifest", "reviews"])
def test_mid_run_changes_abort_entire_evaluation(visual_dataset, tmp_path, monkeypatch, case):
    from rxsentinel import visual_evaluation

    manifest, catalog, root, _ = visual_dataset
    original = visual_evaluation.retrieve_photo
    changed = False

    def mutate(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            if case == "evidence":
                (root / "packaging.txt").write_text("changed unknown context")
            elif case == "manifest":
                rewrite(manifest, lambda m: m.update(dataset_id="changed after start"))
            else:
                appearance = catalog.store.appearances()[0]
                catalog.store.put_appearance(appearance.model_copy(update={"stale": True}))
        return result

    monkeypatch.setattr(visual_evaluation, "retrieve_photo", mutate)
    with pytest.raises(ValueError):
        run(visual_dataset, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_unknown_only_evaluation_has_no_fabricated_known_accuracy(visual_dataset, tmp_path):
    rewrite(
        visual_dataset[0],
        lambda m: m.update(cases=[c for c in m["cases"] if c["kind"] == "unknown"]),
    )
    result = run(visual_dataset, tmp_path / "unknown-only")
    assert result["metrics"]["known_top1_recall"] is None
    assert result["metrics"]["known_top3_recall"] is None
    assert result["query_appearance_count"] == 0
    assert len(result["gallery_appearances_without_known_queries"]) == 2


def test_cli_uses_synthetic_catalog_without_production_access(
    visual_dataset, tmp_path, monkeypatch, capsys
):
    manifest, _, root, _ = visual_dataset
    monkeypatch.setattr(
        "rxsentinel.pipelines.evaluate_visual.configured_catalog",
        lambda _: pytest.fail("Synthetic run must not open production configuration"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate-visual",
            "run",
            "--manifest",
            str(manifest),
            "--gallery-dir",
            str(root / "gallery"),
            "--index-dir",
            str(root / "vectors"),
            "--output-dir",
            str(tmp_path / "report"),
        ],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["scope"] == "synthetic"


def test_demo_failure_publishes_no_dataset(tmp_path, monkeypatch):
    encoder = FixtureEncoder()
    encoder.vector[:] = 0
    monkeypatch.setattr("rxsentinel.pipelines.evaluate_visual.ResNet18Encoder", lambda _: encoder)
    with pytest.raises(ValueError, match="zero norm"):
        make_visual_demo(tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".rxsentinel-visual-demo-*"))
