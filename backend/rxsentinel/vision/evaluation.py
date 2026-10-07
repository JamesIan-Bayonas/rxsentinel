"""Bounded M3 dataset evaluation. Never establishes medication identity or clinical accuracy."""

import hashlib
import html
import io
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rxsentinel.collection_support import collection_status
from rxsentinel.pipelines.photo import bounded_read
from rxsentinel.vision.report import publish_report
from rxsentinel.vision.single import ProcessingConfig, decode_photo, evaluate_mask, process_photo


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    kind: Literal["positive", "negative"]
    image: str | None = None
    asset_id: str | None = None
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    annotation: str
    annotation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    capture_session: str = Field(min_length=1, max_length=160)
    conditions: list[str] = Field(min_length=1, max_length=20)
    reuse_evidence: str | None = None
    reuse_evidence_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    reuse_reviewer: str | None = None
    reuse_basis: str | None = None

    @model_validator(mode="after")
    def source_choice(self):
        if bool(self.image) == bool(self.asset_id):
            raise ValueError("Choose exactly one image path or catalog asset_id")
        if self.kind == "negative" and self.asset_id:
            raise ValueError("Negative scenes must not assert a catalog pill identity")
        if any(not condition.strip() for condition in self.conditions):
            raise ValueError("Conditions cannot be blank")
        return self


class EvaluationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    version: Literal["m3-evaluation-1"] = "m3-evaluation-1"
    dataset_id: str = Field(min_length=1, max_length=160)
    scope: Literal["synthetic", "unverified-local", "reviewed-collection"]
    partition: Literal["validation", "test"]
    annotator: str = Field(min_length=1, max_length=160)
    annotation_basis: str = Field(min_length=1, max_length=2000)
    config: ProcessingConfig = Field(default_factory=ProcessingConfig)
    localization_iou_threshold: float = Field(default=0.5, gt=0, le=1)
    cases: list[EvaluationCase] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_cases(self):
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("Case IDs must be unique")
        if len({c.input_sha256 for c in self.cases}) != len(self.cases):
            raise ValueError("Duplicate source images cannot count as independent cases")
        for case in self.cases:
            if self.scope == "reviewed-collection":
                if case.kind == "positive" and not case.asset_id:
                    raise ValueError("Reviewed positive cases require a catalog asset_id")
                if case.kind == "negative" and not all(
                    (
                        case.reuse_evidence,
                        case.reuse_evidence_sha256,
                        case.reuse_reviewer,
                        case.reuse_basis,
                    )
                ):
                    raise ValueError(
                        "Reviewed negative scenes require explicit reuse evidence/review"
                    )
            elif case.asset_id:
                raise ValueError("Catalog assets require reviewed-collection scope")
        return self


def local_file(root: Path, relative: str) -> bytes:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError("Evaluation file must be inside the manifest directory")
    return bounded_read(path)


def checked_hash(content: bytes, expected: str, label: str) -> None:
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"{label} checksum mismatch")


def collection_states(store, data_dir):
    products, appearances, assets = store.all(), store.appearances(), store.assets()
    states = {
        a.appearance_id: collection_status(
            data_dir,
            a,
            [asset for asset in assets if asset.appearance_id == a.appearance_id],
            [p for p in products if p.product_id in a.product_ids],
            assets,
        )
        for a in appearances
    }
    return states, {a.asset_id: a for a in assets}


def prepare_cases(manifest, root, *, store=None, data_dir=None):
    states, assets = {}, {}
    if manifest.scope == "reviewed-collection":
        if store is None or data_dir is None:
            raise ValueError("Reviewed evaluation requires the configured relational catalog")
        states, assets = collection_states(store, data_dir)
    prepared, total_bytes, total_pixels, bindings = [], 0, 0, {}
    for case in manifest.cases:
        evidence = None
        if case.asset_id:
            asset = assets.get(case.asset_id)
            if asset is None:
                raise ValueError(f"Unknown catalog asset: {case.asset_id}")
            state = states[asset.appearance_id]
            entry = next(a for a in state["assets"] if a["asset_id"] == case.asset_id)
            if not state["reference_ready"] or not entry["eligible"]:
                raise ValueError(f"Catalog asset is not eligible: {case.asset_id}")
            if (
                asset.partition != manifest.partition
                or asset.capture_session != case.capture_session
            ):
                raise ValueError(
                    "Catalog partition/capture session differs from evaluation manifest"
                )
            if asset.sha256 != case.input_sha256:
                raise ValueError("Catalog image checksum differs from evaluation manifest")
            source = bounded_read(data_dir / asset.local_path)
            bindings[asset.appearance_id] = state["fingerprint"]
        else:
            source = local_file(root, case.image)
            if manifest.scope == "reviewed-collection":
                evidence = local_file(root, case.reuse_evidence)
                checked_hash(evidence, case.reuse_evidence_sha256, "Negative reuse evidence")
                # Negative scenes must also be independent of every catalog partition.
                if any(
                    a.sha256 == case.input_sha256
                    or (
                        a.capture_session == case.capture_session
                        and a.partition != manifest.partition
                    )
                    for a in assets.values()
                ):
                    raise ValueError("Negative scene leaks a catalog image or capture session")
        annotation = local_file(root, case.annotation)
        checked_hash(source, case.input_sha256, "Input image")
        checked_hash(annotation, case.annotation_sha256, "Annotation")
        rgb, _, _, _ = decode_photo(source)
        truth, kind, _, orientation = decode_photo(annotation)
        if kind != "PNG" or orientation not in (None, 1) or truth.shape != rgb.shape:
            raise ValueError("Annotation must be an oriented PNG matching source dimensions")
        if not np.all((truth == 0) | (truth == 255)) or not np.all(truth == truth[:, :, :1]):
            raise ValueError("Annotation must be binary black/white")
        if bool(np.any(truth)) != (case.kind == "positive"):
            raise ValueError("Positive masks need an object; negative masks must be entirely black")
        total_bytes += len(source) + len(annotation) + (len(evidence) if evidence else 0)
        total_pixels += rgb.shape[0] * rgb.shape[1]
        if total_bytes > 128 * 1024**2 or total_pixels > 100_000_000:
            raise ValueError("Evaluation batch exceeds byte/pixel budget")
        prepared.append((case, source, annotation, evidence))
    return prepared, bindings


def summarize(rows):
    positives = [r for r in rows if r["kind"] == "positive"]
    negatives = [r for r in rows if r["kind"] == "negative"]
    detected = sum(r["status"] == "processed" for r in positives)
    false_positive = sum(r["status"] == "processed" for r in negatives)
    localized = sum(r["localized"] for r in positives)
    return {
        "case_count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "positive_processed_count": detected,
        "positive_rejected_count": len(positives) - detected,
        "positive_localized_count": localized,
        "positive_wrong_crop_count": detected - localized,
        "positive_localization_recall": localized / len(positives) if positives else None,
        "capture_session_count": len({r["capture_session"] for r in rows}),
        "negative_false_detection_count": false_positive,
        "negative_rejected_count": len(negatives) - false_positive,
        "positive_processing_rate": detected / len(positives) if positives else None,
        "negative_false_detection_rate": false_positive / len(negatives) if negatives else None,
        "mean_mask_iou_all_positives": (
            sum(r["mask_iou"] for r in positives) / len(positives) if positives else None
        ),
        "mean_crop_completeness_all_positives": (
            sum(r["crop_completeness"] for r in positives) / len(positives) if positives else None
        ),
    }


def run_evaluation(manifest_path, output, *, store=None, data_dir=None):
    raw = bounded_read(manifest_path)
    manifest = EvaluationManifest.model_validate_json(raw)
    output = output.resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    prepared, bindings = prepare_cases(
        manifest, manifest_path.parent, store=store, data_dir=data_dir
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rxsentinel-evaluation-", dir=output.parent) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        rows = []
        for case, source, annotation, evidence in prepared:
            report, artifacts = process_photo(source, manifest.config)
            if case.kind == "positive":
                evaluate_mask(
                    report,
                    artifacts,
                    annotation,
                    scope="synthetic" if manifest.scope == "synthetic" else "unverified-local",
                )
            with Image.open(io.BytesIO(annotation)) as image:
                artifacts["annotation"] = np.array(image.convert("L"))
            report.warnings.insert(
                0, f"Dataset evaluation scope: {manifest.scope}; no identity accuracy"
            )
            publish_report(stage / case.case_id, report, artifacts, source)
            if evidence:
                (stage / case.case_id / "reuse-evidence.bin").write_bytes(evidence)
            row = {
                "case_id": case.case_id,
                "kind": case.kind,
                "asset_id": case.asset_id,
                "capture_session": case.capture_session,
                "conditions": case.conditions,
                "input_sha256": case.input_sha256,
                "annotation_sha256": case.annotation_sha256,
                "status": report.status,
                "candidate_count": report.candidate_count,
                "rejection_reasons": report.rejection_reasons,
                "warnings": report.warnings,
                "report": f"{case.case_id}/report.html",
            }
            if case.kind == "positive":
                row.update(
                    {
                        k: report.annotation_evaluation[k]
                        for k in (
                            "mask_iou",
                            "object_pixel_recall",
                            "crop_completeness",
                        )
                    }
                )
                row["localized"] = (
                    report.status == "processed"
                    and row["mask_iou"] >= manifest.localization_iou_threshold
                )
            rows.append(row)
        if manifest.scope == "reviewed-collection":
            current, current_assets = collection_states(store, data_dir)
            if any(
                current.get(key, {}).get("fingerprint") != value
                or not current[key]["reference_ready"]
                for key, value in bindings.items()
            ):
                raise ValueError(
                    "Collection review changed during evaluation; rerun with current data"
                )
            for case in manifest.cases:
                if case.asset_id and not any(
                    a["asset_id"] == case.asset_id and a["eligible"]
                    for state in current.values()
                    for a in state["assets"]
                ):
                    raise ValueError("Catalog query asset lost eligibility during evaluation")
                if case.kind == "negative" and any(
                    a.sha256 == case.input_sha256
                    or (
                        a.capture_session == case.capture_session
                        and a.partition != manifest.partition
                    )
                    for a in current_assets.values()
                ):
                    raise ValueError(
                        "Negative scene acquired a catalog partition conflict during evaluation"
                    )
        conditions = sorted({condition for row in rows for condition in row["conditions"]})
        summary = {
            "evaluation_version": "m3-evaluation-1",
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": manifest.dataset_id,
            "scope": manifest.scope,
            "partition": manifest.partition,
            "localization_iou_threshold": manifest.localization_iou_threshold,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "config_sha256": report.config_sha256,
            "baseline_version": report.baseline_version,
            "library_versions": report.library_versions,
            "collection_fingerprints": bindings,
            "metrics": summarize(rows),
            "conditions": {
                condition: summarize([r for r in rows if condition in r["conditions"]])
                for condition in conditions
            },
            "cases": rows,
            "medication_identification_available": False,
            "m3_exit_gate_satisfied": False,
            "limitations": [
                "Processing rate alone does not measure correct localization",
                "Localization recall uses the recorded IoU threshold, not medication identity",
                "Rejected positive cases remain in mask/completeness denominators",
                "Dataset partition labels do not prove an untouched test set",
                "No quantitative acceptance targets or clinical validation are asserted",
            ],
        }
        (stage / "manifest.json").write_bytes(raw)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        table = "".join(
            f'<tr><td><a href="{r["report"]}">{html.escape(r["case_id"])}</a></td>'
            f"<td>{r['kind']}</td><td>{r['status']}</td>"
            f"<td>{html.escape(', '.join(r['rejection_reasons']))}</td></tr>"
            for r in rows
        )
        labels = {
            "positive_count": "Photos with an annotated object",
            "negative_count": "Photos expected to contain no pill",
            "positive_localized_count": "Objects isolated at the chosen overlap threshold",
            "positive_rejected_count": "Object photos rejected",
            "positive_wrong_crop_count": "Object photos with an incorrect crop",
            "negative_false_detection_count": "No-pill photos that produced a false detection",
            "capture_session_count": "Capture sessions represented",
            "mean_mask_iou_all_positives": "Mean mask overlap, including rejected photos",
            "mean_crop_completeness_all_positives": "Mean crop completeness (rejections included)",
        }
        metric_rows = "".join(
            f"<tr><td>{label}</td><td>{summary['metrics'][key]}</td></tr>"
            for key, label in labels.items()
        )
        page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RxSentinel dataset evaluation</title><style>
body{{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:1rem}}
td,th{{padding:.6rem;text-align:left;border-bottom:1px solid #aaa}}
pre{{white-space:pre-wrap}}</style><h1>Dataset evaluation: {html.escape(manifest.dataset_id)}</h1>
<p>Scope: {manifest.scope}. Partition: {manifest.partition}. No medication identification.</p>
<p>This report does not close the real-photo evaluation gate.</p>
<table>{metric_rows}</table>
<table><tr><th>Case</th><th>Expected scene</th><th>Result</th><th>Rejection reasons</th></tr>
{table}</table><p><a href="summary.json">Metrics, conditions, provenance and limitations</a></p>
</html>"""
        (stage / "report.html").write_text(page, encoding="utf-8")
        stage.rename(output)
    return summary
