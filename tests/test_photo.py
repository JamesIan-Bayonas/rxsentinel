import hashlib
import io
import json

import cv2
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from rxsentinel.pipelines.photo import bounded_read, main, synthetic_demo
from rxsentinel.vision.report import publish_report
from rxsentinel.vision.single import (
    ProcessingConfig,
    decode_photo,
    evaluate_mask,
    letterbox,
    process_photo,
)


def encode(rgb, *, mode=None, exif=None):
    buffer = io.BytesIO()
    image = Image.fromarray(rgb) if mode is None else Image.fromarray(rgb, mode)
    image.save(buffer, format="PNG", **({"exif": exif} if exif else {}))
    return buffer.getvalue()


@pytest.mark.parametrize("background,object_color", [(25, 210), (230, 50)])
def test_isolation_on_dark_and_light_backgrounds(background, object_color):
    rgb = np.full((400, 600, 3), background, np.uint8)
    truth = np.zeros((400, 600), np.uint8)
    cv2.ellipse(truth, (300, 200), (100, 65), 20, 0, 360, 255, -1)
    rgb[truth > 0] = object_color
    report, artifacts = process_photo(encode(rgb))
    evaluate_mask(report, artifacts, encode(truth), scope="synthetic")
    assert report.status == "processed"
    assert report.candidate_count == 1
    assert report.annotation_evaluation["mask_iou"] > 0.99
    assert report.annotation_evaluation["crop_completeness"] == 1
    left, top, right, bottom = report.crop_box
    np.testing.assert_array_equal(artifacts["crop_original"], rgb[top:bottom, left:right])
    assert artifacts["model_input"].shape == (224, 224, 3)
    assert artifacts["crop_clahe"].ndim == 2
    assert artifacts["crop_mask"].shape == artifacts["crop_original"].shape[:2]
    assert not report.medication_identification_available


def test_high_resolution_crop_retains_pixels_and_maps_coordinates():
    source, mask = synthetic_demo()
    rgb = np.array(Image.open(io.BytesIO(source)).resize((2400, 1800)))
    truth = np.array(Image.open(io.BytesIO(mask)).resize((2400, 1800), Image.Resampling.NEAREST))
    report, artifacts = process_photo(encode(rgb))
    evaluate_mask(report, artifacts, encode(truth), scope="synthetic")
    assert report.analysis_dimensions == (1024, 768)
    assert report.oriented_dimensions == (2400, 1800)
    left, top, right, bottom = report.crop_box
    np.testing.assert_array_equal(artifacts["crop_original"], rgb[top:bottom, left:right])
    assert report.annotation_evaluation["mask_iou"] > 0.97
    assert report.annotation_evaluation["crop_completeness"] == 1


def test_exif_rotation_and_annotation_coordinate_contract():
    source, mask = synthetic_demo()
    rgb = np.array(Image.open(io.BytesIO(source)))
    exif = Image.Exif()
    exif[274] = 6
    content = encode(rgb, exif=exif)
    report, artifacts = process_photo(content)
    assert report.original_dimensions == (800, 600)
    assert report.oriented_dimensions == (600, 800)
    np.testing.assert_array_equal(artifacts["original"], np.rot90(rgb, k=3))
    with pytest.raises(ValueError, match="dimensions"):
        evaluate_mask(report, artifacts, mask, scope="synthetic")
    rotated = np.rot90(np.array(Image.open(io.BytesIO(mask))), k=3)
    evaluate_mask(report, artifacts, encode(rotated), scope="synthetic")
    assert report.annotation_evaluation["mask_iou"] > 0.99
    with pytest.raises(ValueError, match="already be"):
        evaluate_mask(report, artifacts, encode(rotated, exif=exif), scope="synthetic")


def test_letterboxing_preserves_aspect_and_records_rounding():
    rgb = np.full((100, 300, 3), (10, 20, 30), np.uint8)
    result, transform = letterbox(rgb, 224)
    assert transform["width"] == 224 and transform["height"] == 75
    assert transform["top"] == 74 and transform["left"] == 0
    np.testing.assert_array_equal(result[74:149, 0], np.tile((10, 20, 30), (75, 1)))
    assert np.all(result[:74] == 127)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("empty", "no_separated_object"),
        ("low_contrast", "no_separated_object"),
        ("multiple", "multiple_objects"),
        ("clipped", "object_touches_image_edge"),
        ("nonuniform", "background_not_uniform"),
        ("small", "image_too_small"),
        ("elongated", "unsupported_object_geometry"),
        ("large", "object_too_large"),
    ],
)
def test_abstention_cases(case, reason):
    rgb = np.full((400, 600, 3), 30, np.uint8)
    if case == "low_contrast":
        cv2.circle(rgb, (300, 200), 60, (35, 35, 35), -1)
    elif case == "multiple":
        for point in ((160, 200), (450, 200)):
            cv2.circle(rgb, point, 60, (210, 210, 210), -1)
    elif case == "clipped":
        cv2.circle(rgb, (5, 200), 60, (210, 210, 210), -1)
    elif case == "nonuniform":
        rgb[:, 300:] = 230
    elif case == "small":
        rgb = rgb[:100, :100]
    elif case == "elongated":
        cv2.rectangle(rgb, (100, 190), (500, 210), (210, 210, 210), -1)
    elif case == "large":
        cv2.rectangle(rgb, (20, 20), (580, 380), (210, 210, 210), -1)
    report, artifacts = process_photo(encode(rgb))
    assert report.status == "rejected"
    assert reason in report.rejection_reasons
    assert "crop_original" not in artifacts
    assert report.crop_box is None


def test_quality_warnings_do_not_establish_glare_or_identity():
    rgb = np.full((400, 600, 3), 30, np.uint8)
    cv2.circle(rgb, (300, 200), 80, (255, 255, 255), -1)
    rgb = cv2.GaussianBlur(rgb, (31, 31), 8)
    report, _ = process_photo(encode(rgb))
    assert report.status == "processed"
    assert any("white surface" in warning for warning in report.warnings)
    assert any("Focus uncertain" in warning for warning in report.warnings)


@pytest.mark.parametrize("case", ["alpha", "animated", "invalid_orientation", "corrupt"])
def test_unsupported_decoding(case):
    rgb = np.full((200, 300, 3), 80, np.uint8)
    if case == "alpha":
        content = encode(np.dstack((rgb, np.full((200, 300), 255, np.uint8))))
    elif case == "animated":
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(
            buffer, format="PNG", save_all=True, append_images=[Image.fromarray(rgb + 1)]
        )
        content = buffer.getvalue()
    elif case == "invalid_orientation":
        exif = Image.Exif()
        exif[274] = 9
        content = encode(rgb, exif=exif)
    else:
        content = b"not an image"
    with pytest.raises((ValueError, OSError)):
        decode_photo(content)


def test_bounded_file_and_pixel_decoding(tmp_path, monkeypatch):
    from rxsentinel.pipelines import photo, reference_files

    path = tmp_path / "large.png"
    path.write_bytes(b"x" * 101)
    monkeypatch.setattr(photo, "MAX_IMAGE_BYTES", 100)
    with pytest.raises(ValueError, match="byte limit"):
        bounded_read(path)
    monkeypatch.setattr(reference_files, "MAX_PIXELS", 100)
    with pytest.raises(ValueError, match="pixel count"):
        decode_photo(encode(np.zeros((20, 20, 3), np.uint8)))


def test_mask_validation_and_rejected_prediction():
    source, mask = synthetic_demo()
    report, artifacts = process_photo(source)
    with pytest.raises(ValueError, match="only black"):
        evaluate_mask(report, artifacts, source, scope="synthetic")
    with pytest.raises(ValueError, match="empty"):
        evaluate_mask(report, artifacts, encode(np.zeros((600, 800), np.uint8)), scope="synthetic")
    rejected, outputs = process_photo(encode(np.full((600, 800, 3), 30, np.uint8)))
    evaluate_mask(rejected, outputs, mask, scope="synthetic")
    assert rejected.annotation_evaluation["mask_iou"] == 0
    assert rejected.annotation_evaluation["crop_completeness"] == 0


def test_immutable_report_and_artifact_checksums(tmp_path):
    source, annotation = synthetic_demo()
    report, artifacts = process_photo(source)
    evaluate_mask(report, artifacts, annotation, scope="synthetic")
    output = tmp_path / "run"
    path = publish_report(output, report, artifacts, source)
    saved = json.loads((output / "report.json").read_text())
    assert path.exists()
    assert (output / "source.png").read_bytes() == source
    assert saved["annotation_evaluation"]["scope"] == "synthetic"
    for name, sha in saved["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == sha
    with pytest.raises(ValueError, match="already exists"):
        publish_report(output, report, artifacts, source)


def test_invalid_annotation_does_not_publish_output(tmp_path, monkeypatch):
    source, _ = synthetic_demo()
    image = tmp_path / "input.png"
    image.write_bytes(source)
    output = tmp_path / "failed"
    monkeypatch.setattr(
        "sys.argv",
        ["photo", "--input", str(image), "--annotation", str(image), "--output-dir", str(output)],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert not output.exists()


def test_config_contract_and_cli_demo(tmp_path, monkeypatch, capsys):
    with pytest.raises(ValidationError):
        ProcessingConfig(min_lab_distance=float("nan"))
    with pytest.raises(ValidationError):
        ProcessingConfig(unknown_setting=1)
    output = tmp_path / "demo"
    monkeypatch.setattr("sys.argv", ["photo", "--demo", "--output-dir", str(output)])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["medication_identification_available"] is False
    report = json.loads((output / "report.json").read_text())
    assert report["annotation_evaluation"]["scope"] == "synthetic"
    assert report["warnings"][0].startswith("SYNTHETIC")


def test_cli_rejection_publishes_diagnostics_without_usable_crop(tmp_path, monkeypatch, capsys):
    image = tmp_path / "empty.png"
    image.write_bytes(encode(np.full((400, 600, 3), 30, np.uint8)))
    output = tmp_path / "rejected"
    monkeypatch.setattr("sys.argv", ["photo", "--input", str(image), "--output-dir", str(output)])
    assert main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"
    assert (output / "report.html").exists()
    assert not (output / "crop_original.png").exists()


def test_touching_objects_are_an_explicit_unresolved_limitation():
    rgb = np.full((400, 600, 3), 30, np.uint8)
    for point in ((265, 200), (335, 200)):
        cv2.circle(rgb, point, 60, (210, 210, 210), -1)
    report, _ = process_photo(encode(rgb))
    # Connected contours cannot establish pill count; this is not instance segmentation.
    assert report.candidate_count == 1
    assert report.status == "processed"
    assert any("touching objects" in warning for warning in report.warnings)
    assert not report.medication_identification_available
