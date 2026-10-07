import hashlib
import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from rxsentinel.pipelines.collection import review
from rxsentinel.pipelines.evaluate import main, make_demo
from rxsentinel.vision.evaluation import EvaluationManifest, run_evaluation
from test_collection import approval, imported
from test_collection import collection as collection


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def reviewed_manifest(collection, tmp_path, *, approve=True):
    store, directory = imported(collection)
    if approve:
        review(store, directory, approval(collection))
    asset = next(a for a in store.assets() if a.partition == "test")
    Image.fromarray(np.full((asset.height, asset.width), 255, np.uint8)).save(tmp_path / "mask.png")
    manifest = {
        "dataset_id": "Synthetic approved workflow fixture",
        "scope": "reviewed-collection",
        "partition": "test",
        "annotator": "Synthetic test author",
        "annotation_basis": "Entire small synthetic image; not a real pill annotation",
        "cases": [
            {
                "case_id": "approved-test",
                "kind": "positive",
                "asset_id": asset.asset_id,
                "input_sha256": asset.sha256,
                "annotation": "mask.png",
                "annotation_sha256": sha(tmp_path / "mask.png"),
                "capture_session": asset.capture_session,
                "conditions": ["synthetic-small-image"],
            }
        ],
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(manifest))
    return path, store, directory


def test_batch_measures_failures_without_dropping_rejected_cases(tmp_path):
    path = make_demo(tmp_path / "fixtures")
    output = tmp_path / "results"
    result = run_evaluation(path, output)
    metrics = result["metrics"]
    assert metrics["case_count"] == 4
    assert metrics["positive_count"] == metrics["negative_count"] == 2
    assert metrics["positive_localization_recall"] == 0.5
    assert metrics["positive_rejected_count"] == 1
    assert metrics["negative_false_detection_count"] == 1
    assert metrics["negative_false_detection_rate"] == 0.5
    assert metrics["mean_crop_completeness_all_positives"] == 0.5
    assert result["conditions"]["low-contrast"]["positive_localized_count"] == 0
    assert result["scope"] == "synthetic"
    assert not result["m3_exit_gate_satisfied"]
    assert not result["medication_identification_available"]
    report = json.loads((output / "positive-clear" / "report.json").read_text())
    assert result["config_sha256"] == report["config_sha256"]
    assert (output / "manifest.json").read_bytes() == path.read_bytes()
    assert (output / "report.html").exists()
    with pytest.raises(ValueError, match="already exists"):
        run_evaluation(path, output)


@pytest.mark.parametrize(
    "case", ["duplicate_id", "duplicate_photo", "training", "nan", "missing_review"]
)
def test_manifest_rejects_invalid_dataset_contract(tmp_path, case):
    path = make_demo(tmp_path / "fixtures")
    data = json.loads(path.read_text())
    if case == "duplicate_id":
        data["cases"][1]["case_id"] = data["cases"][0]["case_id"]
    elif case == "duplicate_photo":
        data["cases"][1]["input_sha256"] = data["cases"][0]["input_sha256"]
    elif case == "training":
        data["partition"] = "train"
    elif case == "nan":
        data["localization_iou_threshold"] = float("nan")
    else:
        data["scope"] = "reviewed-collection"
        data["cases"] = [data["cases"][-1]]
    with pytest.raises(ValidationError):
        EvaluationManifest.model_validate(data)


@pytest.mark.parametrize(
    "case", ["source_hash", "mask_hash", "mask_label", "path_escape", "dimensions"]
)
def test_preflight_failure_publishes_no_partial_dataset(tmp_path, case):
    path = make_demo(tmp_path / "fixtures")
    if case == "source_hash":
        rewrite(path, lambda m: m["cases"][0].update(input_sha256="0" * 64))
    elif case == "mask_hash":
        rewrite(path, lambda m: m["cases"][0].update(annotation_sha256="0" * 64))
    elif case == "mask_label":
        rewrite(path, lambda m: m["cases"][0].update(kind="negative"))
    elif case == "path_escape":
        rewrite(path, lambda m: m["cases"][0].update(image="../outside.png"))
    else:
        mask = path.parent / "positive-clear-mask.png"
        Image.new("L", (10, 10), 255).save(mask)
        rewrite(path, lambda m: m["cases"][0].update(annotation_sha256=sha(mask)))
    output = tmp_path / "results"
    with pytest.raises(ValueError):
        run_evaluation(path, output)
    assert not output.exists()


def test_wrong_localization_does_not_count_as_recall(tmp_path):
    path = make_demo(tmp_path / "fixtures")
    mask = path.parent / "positive-clear-mask.png"
    truth = np.zeros((600, 800), np.uint8)
    truth[20:100, 20:100] = 255
    Image.fromarray(truth).save(mask)
    rewrite(path, lambda m: m["cases"][0].update(annotation_sha256=sha(mask)))
    result = run_evaluation(path, tmp_path / "results")
    assert result["metrics"]["positive_processed_count"] == 1
    assert result["metrics"]["positive_wrong_crop_count"] == 1
    assert result["metrics"]["positive_localization_recall"] == 0


def test_one_sided_dataset_reports_missing_denominator_as_null(tmp_path):
    path = make_demo(tmp_path / "fixtures")
    rewrite(path, lambda m: m.update(cases=[m["cases"][0]]))
    result = run_evaluation(path, tmp_path / "results")
    assert result["metrics"]["negative_count"] == 0
    assert result["metrics"]["negative_false_detection_rate"] is None


def test_html_escapes_dataset_name(tmp_path):
    path = make_demo(tmp_path / "fixtures")
    rewrite(path, lambda m: m.update(dataset_id="<script>unsafe</script>"))
    output = tmp_path / "results"
    run_evaluation(path, output)
    page = (output / "report.html").read_text()
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_reviewed_asset_binds_live_collection_and_includes_rejection(collection, tmp_path):
    path, store, directory = reviewed_manifest(collection, tmp_path)
    result = run_evaluation(path, tmp_path / "results", store=store, data_dir=directory)
    assert result["collection_fingerprints"]
    assert result["scope"] == "reviewed-collection"
    assert result["metrics"]["positive_rejected_count"] == 1
    assert result["metrics"]["negative_false_detection_rate"] is None
    assert not result["m3_exit_gate_satisfied"]


@pytest.mark.parametrize("case", ["pending", "partition", "session", "tamper", "missing_catalog"])
def test_reviewed_evaluation_rejects_ineligible_or_changed_data(collection, tmp_path, case):
    path, store, directory = reviewed_manifest(collection, tmp_path, approve=case != "pending")
    if case == "partition":
        rewrite(path, lambda m: m.update(partition="validation"))
    elif case == "session":
        rewrite(path, lambda m: m["cases"][0].update(capture_session="another-session"))
    elif case == "tamper":
        asset = next(a for a in store.assets() if a.partition == "test")
        (directory / asset.local_path).write_bytes(b"changed image")
    output = tmp_path / "results"
    with pytest.raises(ValueError):
        run_evaluation(
            path, output, store=None if case == "missing_catalog" else store, data_dir=directory
        )
    assert not output.exists()


def test_mid_run_query_tampering_prevents_publication(collection, tmp_path, monkeypatch):
    from rxsentinel.vision import evaluation

    path, store, directory = reviewed_manifest(collection, tmp_path)
    original = evaluation.process_photo

    def tampering(source, config):
        result = original(source, config)
        asset = next(a for a in store.assets() if a.partition == "test")
        (directory / asset.local_path).write_bytes(b"tampered during run")
        return result

    monkeypatch.setattr(evaluation, "process_photo", tampering)
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="lost eligibility"):
        run_evaluation(path, output, store=store, data_dir=directory)
    assert not output.exists()
    assert not list(tmp_path.glob(".rxsentinel-evaluation-*"))


def test_reviewed_negative_evidence_is_preserved_and_leakage_rejected(collection, tmp_path):
    path, store, directory = reviewed_manifest(collection, tmp_path)
    demo = make_demo(tmp_path / "fixtures")
    negative = json.loads(demo.read_text())["cases"][2]
    for key in ("image", "annotation"):
        negative[key] = f"fixtures/{negative[key]}"
    evidence = tmp_path / "negative-permission.txt"
    evidence.write_text("Synthetic test permission, not real photo permission")
    negative.update(
        reuse_evidence=evidence.name,
        reuse_evidence_sha256=sha(evidence),
        reuse_reviewer="Synthetic reviewer",
        reuse_basis="Synthetic fixture only",
    )
    rewrite(path, lambda m: m["cases"].append(negative))
    output = tmp_path / "results"
    run_evaluation(path, output, store=store, data_dir=directory)
    assert (
        output / negative["case_id"] / "reuse-evidence.bin"
    ).read_bytes() == evidence.read_bytes()
    negative["capture_session"] = "synthetic-reference"
    rewrite(path, lambda m: m.update(cases=[m["cases"][0], negative]))
    with pytest.raises(ValueError, match="leaks"):
        run_evaluation(path, tmp_path / "leaked", store=store, data_dir=directory)


def test_cli_demo_and_run_do_not_open_database(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "rxsentinel.pipelines.evaluate.configured_catalog",
        lambda _: pytest.fail("Synthetic run must not open a database"),
    )
    monkeypatch.setattr(
        "sys.argv", ["evaluation", "demo", "--directory", str(tmp_path / "fixtures")]
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["scope"] == "synthetic"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluation",
            "run",
            "--manifest",
            str(tmp_path / "fixtures" / "manifest.json"),
            "--output-dir",
            str(tmp_path / "results"),
        ],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["m3_exit_gate_satisfied"] is False
