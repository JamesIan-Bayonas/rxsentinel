"""Versioned, plain-background contour baseline with explicit abstention."""

import hashlib
import io
import math
from datetime import UTC, datetime
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from rxsentinel.pipelines.reference_files import inspect_image

BASELINE_VERSION = "plain-background-0.1"


class ProcessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    analysis_size: int = Field(default=1024, ge=256, le=1600)
    model_size: int = Field(default=224, ge=32, le=1024)
    min_area_fraction: float = Field(default=0.005, gt=0, le=0.1)
    max_area_fraction: float = Field(default=0.65, ge=0.2, lt=1)
    min_lab_distance: float = Field(default=15, ge=5, le=60)
    max_border_variation: float = Field(default=25, ge=5, le=60)
    min_solidity: float = Field(default=0.8, ge=0.5, le=1)
    max_aspect_ratio: float = Field(default=4, ge=1, le=10)
    crop_margin_fraction: float = Field(default=0.1, ge=0, le=0.5)
    focus_warning_threshold: float = Field(default=20, ge=0, le=1000)


class ProcessingReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    baseline_version: str = BASELINE_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: Literal["processed", "rejected"] = "rejected"
    experimental: bool = True
    medication_identification_available: bool = False
    input_sha256: str
    input_format: str
    original_dimensions: tuple[int, int]
    oriented_dimensions: tuple[int, int]
    exif_orientation: int | None
    coordinate_space: str = "EXIF-oriented RGB image; boxes are [left, top, right, bottom), pixels"
    config: ProcessingConfig
    config_sha256: str
    analysis_dimensions: tuple[int, int] = (0, 0)
    candidate_count: int = 0
    crop_box: tuple[int, int, int, int] | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    rejection_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    letterbox: dict[str, int | float] | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    annotation_evaluation: dict[str, str | float] | None = None
    library_versions: dict[str, str] = Field(default_factory=dict)


def decode_photo(content: bytes) -> tuple[np.ndarray, str, tuple[int, int], int | None]:
    kind, width, height = inspect_image(content)
    with Image.open(io.BytesIO(content)) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated or multi-frame images are unsupported")
        if "A" in image.getbands() or "transparency" in image.info:
            raise ValueError("Transparent images are unsupported; capture an opaque photo")
        if image.mode not in ("RGB", "L"):
            raise ValueError("Only RGB or grayscale photos are supported")
        orientation = image.getexif().get(274)
        if orientation is not None and orientation not in range(1, 9):
            raise ValueError("Invalid EXIF orientation")
        rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    return rgb, kind, (width, height), orientation


def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, dict[str, int | float]]:
    height, width = rgb.shape[:2]
    scale = min(size / width, size / height)
    out_width, out_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(
        rgb,
        (out_width, out_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    left, top = (size - out_width) // 2, (size - out_height) // 2
    result = np.full((size, size, 3), 127, dtype=np.uint8)
    result[top : top + out_height, left : left + out_width] = resized
    return result, {
        "width": out_width,
        "height": out_height,
        "left": left,
        "top": top,
        "scale_x": out_width / width,
        "scale_y": out_height / height,
        "padding_rgb": 127,
    }


def process_photo(
    content: bytes,
    config: ProcessingConfig | None = None,
) -> tuple[ProcessingReport, dict[str, np.ndarray]]:
    config = config or ProcessingConfig()
    rgb, kind, original_size, orientation = decode_photo(content)
    height, width = rgb.shape[:2]
    config_json = config.model_dump_json()
    report = ProcessingReport(
        input_sha256=hashlib.sha256(content).hexdigest(),
        input_format=kind,
        original_dimensions=original_size,
        oriented_dimensions=(width, height),
        exif_orientation=orientation,
        config=config,
        config_sha256=hashlib.sha256(config_json.encode()).hexdigest(),
        library_versions={
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        warnings=[
            "Object shape does not establish that it is a pill or identify a medication",
            "Quality thresholds are heuristic and have not been calibrated on real pills",
        ],
    )
    artifacts = {"original": rgb}
    scale = min(1, config.analysis_size / max(width, height))
    small = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    sh, sw = small.shape[:2]
    report.analysis_dimensions = (sw, sh)
    if min(sw, sh) < 128:
        report.rejection_reasons.append("image_too_small")
        return report, artifacts

    # Float Lab uses L=0..100, rather than OpenCV's packed uint8 representation.
    lab = cv2.cvtColor(small.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    border_size = max(2, round(min(sw, sh) * 0.03))
    border = np.ones((sh, sw), dtype=bool)
    border[border_size:-border_size, border_size:-border_size] = False
    background = np.median(lab[border], axis=0)
    distances = np.linalg.norm(lab - background, axis=2)
    variation = float(np.percentile(distances[border], 90))
    threshold = max(config.min_lab_distance, variation * 2)
    report.metrics.update(border_lab_variation_p90=variation, lab_distance_threshold=threshold)
    if variation > config.max_border_variation:
        report.rejection_reasons.append("background_not_uniform")
        return report, artifacts
    mask = (distances > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= sw * sh * config.min_area_fraction]
    candidates.sort(key=lambda c: cv2.boundingRect(c)[:2])
    report.candidate_count = len(candidates)
    overlay = small.copy()
    cv2.drawContours(overlay, candidates, -1, (255, 0, 0), 2)
    artifacts.update(analysis_overlay=overlay, analysis_mask=mask)
    if len(candidates) != 1:
        report.rejection_reasons.append(
            "no_separated_object" if not candidates else "multiple_objects"
        )
        return report, artifacts

    contour = candidates[0]
    x, y, cw, ch = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    solidity = area / max(cv2.contourArea(cv2.convexHull(contour)), 1)
    aspect = max(cw / ch, ch / cw)
    report.metrics.update(
        object_area_fraction=area / (sw * sh), solidity=solidity, aspect_ratio=aspect
    )
    if x <= 1 or y <= 1 or x + cw >= sw - 1 or y + ch >= sh - 1:
        report.rejection_reasons.append("object_touches_image_edge")
    if area / (sw * sh) > config.max_area_fraction:
        report.rejection_reasons.append("object_too_large")
    if solidity < config.min_solidity or aspect > config.max_aspect_ratio:
        report.rejection_reasons.append("unsupported_object_geometry")
    if report.rejection_reasons:
        return report, artifacts

    isolated = np.zeros((sh, sw), np.uint8)
    cv2.drawContours(isolated, [contour], -1, 255, cv2.FILLED)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    pixels = gray[isolated > 0]
    focus = float(cv2.Laplacian(gray, cv2.CV_32F)[isolated > 0].var())
    report.metrics.update(
        object_gray_median=float(np.median(pixels)),
        object_laplacian_variance=focus,
        bright_pixel_fraction=float(np.mean(pixels >= 250)),
        dark_pixel_fraction=float(np.mean(pixels <= 5)),
    )
    if focus < config.focus_warning_threshold:
        report.warnings.append("Focus uncertain: weak texture or blur; inspect or retake the photo")
    if np.mean(pixels >= 250) > 0.1:
        report.warnings.append("Bright clipping or white surface: possible lost imprint detail")
    if np.median(pixels) < 25:
        report.warnings.append("Dark object or underexposure: inspect imprint visibility")
    report.warnings.append(
        "Shadows, touching objects and background-like regions can fool this baseline"
    )
    full_mask = cv2.resize(isolated, (width, height), interpolation=cv2.INTER_NEAREST)
    left, top, mask_width, mask_height = cv2.boundingRect(full_mask)
    right, bottom = left + mask_width, top + mask_height
    margin = math.ceil(max(right - left, bottom - top) * config.crop_margin_fraction)
    left, top = max(0, left - margin), max(0, top - margin)
    right, bottom = min(width, right + margin), min(height, bottom + margin)
    crop = rgb[top:bottom, left:right].copy()
    crop_mask = full_mask[top:bottom, left:right].copy()
    enhanced = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(
        cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    )
    model_copy, transform = letterbox(crop, config.model_size)
    report.status, report.crop_box, report.letterbox = (
        "processed",
        (left, top, right, bottom),
        transform,
    )
    artifacts.update(
        full_mask=full_mask,
        crop_original=crop,
        crop_mask=crop_mask,
        crop_clahe=enhanced,
        model_input=model_copy,
    )
    return report, artifacts


def evaluate_mask(
    report: ProcessingReport,
    artifacts: dict[str, np.ndarray],
    content: bytes,
    *,
    scope: Literal["synthetic", "unverified-local"],
) -> None:
    """Measure one manual mask, including rejected detections as an empty prediction."""
    rgb, _, _, orientation = decode_photo(content)
    if orientation not in (None, 1):
        raise ValueError("Annotation must already be in the oriented photo coordinate space")
    if (rgb.shape[1], rgb.shape[0]) != report.oriented_dimensions:
        raise ValueError("Annotation dimensions must equal the oriented photo dimensions")
    if not np.all((rgb == 0) | (rgb == 255)) or not np.all(rgb == rgb[:, :, :1]):
        raise ValueError("Annotation must contain only black background and white object pixels")
    truth = rgb[:, :, 0] == 255
    if not truth.any():
        raise ValueError("Positive-object annotation is empty")
    prediction = artifacts.get("full_mask", np.zeros(truth.shape, np.uint8)) > 0
    intersection, union = np.sum(truth & prediction), np.sum(truth | prediction)
    completeness = 0.0
    if report.crop_box:
        left, top, right, bottom = report.crop_box
        completeness = float(truth[top:bottom, left:right].sum() / truth.sum())
    report.annotation_evaluation = {
        "scope": scope,
        "annotation_sha256": hashlib.sha256(content).hexdigest(),
        "mask_iou": float(intersection / union),
        "object_pixel_recall": float(intersection / truth.sum()),
        "crop_completeness": completeness,
        "note": "One positive-object mask; not medication accuracy or held-out dataset validation",
    }
