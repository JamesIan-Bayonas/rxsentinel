"""Frozen known/unknown visual retrieval evaluation, separate from identity acceptance."""

import hashlib
import html
import json
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

from pydantic import Field, model_validator

from rxsentinel.collection_support import digest
from rxsentinel.matching import snapshot
from rxsentinel.matching_evaluation import summarize
from rxsentinel.pipelines.photo import bounded_read
from rxsentinel.pipelines.retrieve import write_report
from rxsentinel.retrieval import retrieve_photo, validate_vectors
from rxsentinel.schemas import FileEvidence, StrictModel
from rxsentinel.vision.evaluation import checked_hash, local_file
from rxsentinel.vision.single import decode_photo

MAX_EVALUATION_BYTES = 128 * 1024**2
MAX_EVALUATION_PIXELS = 100_000_000
Similarity = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]


def evaluation_code_sha256():
    from rxsentinel import collection_support, matching_evaluation, schemas

    paths = [Path(__file__)] + [
        Path(module.__file__) for module in (collection_support, matching_evaluation, schemas)
    ]
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


class VisualCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    kind: Literal["known", "unknown"]
    expected_appearance_id: str | None = None
    asset_id: str | None = None
    image: str | None = None
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    capture_session: str = Field(min_length=1, max_length=160)
    conditions: list[str] = Field(min_length=1, max_length=20)
    evidence: list[FileEvidence] = Field(default_factory=list, max_length=20)
    reuse_reviewer: str | None = Field(default=None, min_length=1, max_length=160)
    reuse_basis: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_case(self):
        if self.kind == "known" and not (
            self.expected_appearance_id and self.asset_id and not self.image
        ):
            raise ValueError("Known queries need a truth appearance and registered query asset")
        if self.kind == "unknown" and not (
            self.image and not self.asset_id and not self.expected_appearance_id
        ):
            raise ValueError(
                "Unknown queries need a local image without an asserted gallery identity"
            )
        if self.kind == "unknown" and not (
            {e.purpose for e in self.evidence} & {"ownership", "license"}
            and {e.purpose for e in self.evidence} & {"packaging", "product_record"}
            and self.reuse_reviewer
            and self.reuse_basis
        ):
            raise ValueError(
                "Unknown queries need reuse review and rights/identity-context evidence"
            )
        if len({e.evidence_id for e in self.evidence}) != len(self.evidence):
            raise ValueError("Evidence IDs must be unique")
        if any(not c.strip() for c in self.conditions) or len(set(self.conditions)) != len(
            self.conditions
        ):
            raise ValueError("Condition labels must be nonblank and unique")
        return self


class VisualBenchmark(StrictModel):
    version: Literal["m4-visual-evaluation-1"] = "m4-visual-evaluation-1"
    dataset_id: str = Field(min_length=1, max_length=160)
    scope: Literal["synthetic", "reviewed-collection"]
    partition: Literal["validation", "test"]
    gallery_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    index_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluator_code_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    annotator: str = Field(min_length=1, max_length=160)
    annotation_basis: str = Field(min_length=1, max_length=2000)
    synthetic_catalog: str | None = None
    minimum_similarity: Similarity | None = None
    validation_cutoffs: list[Similarity] = Field(default_factory=list, max_length=20)
    cases: list[VisualCase] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_protocol(self):
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("Case IDs must be unique")
        if len({c.input_sha256 for c in self.cases}) != len(self.cases):
            raise ValueError("Duplicate query photos cannot count as independent cases")
        if (self.scope == "synthetic") != bool(self.synthetic_catalog):
            raise ValueError("Only synthetic evaluations require a synthetic catalog path")
        if self.validation_cutoffs and (
            self.partition != "validation" or self.minimum_similarity is not None
        ):
            raise ValueError("Cutoff exploration requires an unthresholded validation run")
        if len(set(self.validation_cutoffs)) != len(self.validation_cutoffs):
            raise ValueError("Validation cutoffs must be unique")
        return self


def preflight(spec, root, catalog, data_dir, gallery_dir, index_dir, encoder):
    if spec.evaluator_code_sha256 != evaluation_code_sha256():
        raise ValueError("Evaluation implementation differs from the frozen protocol")
    index, gallery, _ = validate_vectors(
        index_dir, gallery_dir, encoder, catalog=catalog, data_dir=data_dir
    )
    if (index.index_id, gallery.gallery_id, gallery.scope) != (
        spec.index_id,
        spec.gallery_id,
        spec.scope,
    ):
        raise ValueError("Gallery, index or scope differs from the frozen protocol")
    _, assets, products, reference_sha, states = snapshot(catalog, data_dir)
    if reference_sha != gallery.source_reference_snapshot_sha256:
        raise ValueError("Source snapshot changed during evaluation preflight")
    if spec.scope == "synthetic":
        database = catalog.store.engine.url.database if catalog.store else None
        if (
            not database
            or catalog.store.engine.dialect.name != "sqlite"
            or Path(database).resolve() != (root / spec.synthetic_catalog).resolve()
            or not Path(database).resolve().is_relative_to(root.resolve())
            or data_dir.resolve() != root.resolve()
            or any(not p.product_id.startswith("synthetic:") for p in products.values())
        ):
            raise ValueError("Synthetic evaluation requires its isolated local SQLite catalog")
    indexed_assets = {a.asset_id: a for a in assets}
    gallery_ids = {a.appearance_id for a in gallery.appearances}
    reference_hashes = {a.sha256 for a in gallery.reference_assets}
    reference_sessions = {a.capture_session for a in gallery.reference_assets if a.capture_session}
    prepared, total_bytes, total_pixels = [], 0, 0
    for case in spec.cases:
        if case.input_sha256 in reference_hashes or case.capture_session in reference_sessions:
            raise ValueError("Query photo or capture session overlaps the reference gallery")
        if any(
            a.capture_session == case.capture_session and a.partition != spec.partition
            for a in assets
        ):
            raise ValueError("Query capture session crosses dataset partitions")
        evidence = []
        if case.kind == "known":
            asset = indexed_assets.get(case.asset_id)
            if (
                asset is None
                or asset.appearance_id != case.expected_appearance_id
                or case.expected_appearance_id not in gallery_ids
            ):
                raise ValueError("Known query asset/truth is not linked to an indexed appearance")
            state = states[asset.appearance_id]
            item = next(a for a in state["assets"] if a["asset_id"] == case.asset_id)
            if not state["reference_ready"] or not item["eligible"]:
                raise ValueError("Known query lacks intact current identity/reuse review")
            if (
                asset.partition != spec.partition
                or asset.intended_use != "research_evaluation"
                or asset.capture_session != case.capture_session
                or asset.sha256 != case.input_sha256
            ):
                raise ValueError("Known query partition, purpose, session or hash differs")
            image = local_file(data_dir, asset.local_path)
        else:
            if any(a.sha256 == case.input_sha256 for a in assets):
                raise ValueError("Unknown query duplicates a registered catalog image")
            image = local_file(root, case.image)
            for record in case.evidence:
                content = local_file(root, record.local_path)
                checked_hash(content, record.sha256, "Unknown-case evidence")
                evidence.append((record, content))
        checked_hash(image, case.input_sha256, "Query image")
        rgb, kind, _, _ = decode_photo(image)
        total_bytes += len(image) + sum(len(b) for _, b in evidence)
        total_pixels += rgb.shape[0] * rgb.shape[1]
        del rgb
        if total_bytes > MAX_EVALUATION_BYTES or total_pixels > MAX_EVALUATION_PIXELS:
            raise ValueError("Visual evaluation exceeds encoded byte or source pixel budget")
        prepared.append((case, image, kind, evidence))
    return index, gallery, prepared


def metrics(rows):
    result = summarize(rows)
    result.update(
        processing_rejection_count=sum(r["processing_status"] == "rejected" for r in rows),
        known_processing_rejection_count=sum(
            r["kind"] == "known" and r["processing_status"] == "rejected" for r in rows
        ),
        unknown_processing_rejection_count=sum(
            r["kind"] == "unknown" and r["processing_status"] == "rejected" for r in rows
        ),
        known_unambiguous_top1_count=sum(
            r["kind"] == "known" and r["top1_correct"] and not r["ambiguous"] for r in rows
        ),
    )
    return result


def cutoff_rows(rows, cutoff):
    """Filtering a descending top-three prefix preserves top-1/top-3 membership under a cutoff."""
    filtered = []
    for row in rows:
        ids = [
            identity
            for identity, score in zip(
                row["candidate_appearance_ids"], row["candidate_scores"], strict=True
            )
            if score >= cutoff
        ]
        update = {
            **row,
            "candidate_appearance_ids": ids,
            "status": "candidates" if ids else "unknown",
            "ambiguous": bool(ids) and row["ambiguous"],
        }
        if row["kind"] == "known":
            update.update(
                top1_correct=bool(ids and ids[0] == row["expected_appearance_id"]),
                top3_contains_truth=row["expected_appearance_id"] in ids,
            )
        filtered.append(update)
    return filtered


def render_summary(summary):
    values = {
        "known_count": "Queries with an indexed appearance",
        "unknown_count": "Queries labeled outside the gallery",
        "capture_session_count": "Capture sessions represented",
        "known_top1_recall": "Correct appearance ranked first (all known queries)",
        "known_top3_recall": "Correct appearance in top three (all known queries)",
        "known_wrong_first_candidate_count": "Incorrect first suggestions on known queries",
        "known_abstention_count": "Known queries receiving unknown",
        "unknown_candidate_return_count": "Out-of-gallery queries receiving candidates",
        "processing_rejection_count": "Rejected photos (retained in metric denominators)",
        "ambiguous_result_count": "Queries with tied first-ranked appearances",
        "accepted_identity_count": "Automatically confirmed identities",
    }
    metric_rows = ""
    for key, label in values.items():
        value = summary["metrics"][key]
        display = "Not measured (no examples)" if value is None else str(value)
        if value is not None and key.endswith("recall"):
            display = f"{value:.1%}"
        metric_rows += f"<tr><td>{label}</td><td>{display}</td></tr>"
    case_rows = "".join(
        f'<tr><td><a href="{r["match_report"]}">{html.escape(r["case_id"])}</a></td>'
        f"<td>{r['kind']}</td><td>{r['status']}</td>"
        f"<td>{html.escape(r['reason'].replace('_', ' '))}</td></tr>"
        for r in summary["cases"]
    )
    sweeps = "".join(
        f"<tr><td>{row['minimum_similarity']}</td>"
        f"<td>{row['metrics']['known_top1_recall']}</td>"
        f"<td>{row['metrics']['known_abstention_count']}</td>"
        f"<td>{row['metrics']['unknown_candidate_return_count']}</td></tr>"
        for row in summary["validation_cutoff_comparison"]
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RxSentinel visual retrieval evaluation</title><style>
body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}}
td,th{{text-align:left;padding:.6rem;border-bottom:1px solid #aaa}}</style>
<h1>Visual retrieval evaluation: {html.escape(summary["dataset_id"])}</h1>
<p>Scope: {summary["scope"]}; partition: {summary["partition"]}.</p>
<p>Candidate rankings, not accepted identities. Synthetic results are software checks.</p>
<table>{metric_rows}</table>
<h2>Cases</h2><table><tr><th>Case</th><th>Expected coverage</th><th>Result</th><th>Reason</th></tr>
{case_rows}</table><h2>Validation-only cutoff exploration</h2>
<p>No cutoff is selected automatically. These results do not calibrate clinical acceptance.</p>
<table><tr><th>Cutoff</th><th>Known top-1 recall</th><th>Known unknown results</th>
<th>Unknown queries with candidates</th></tr>{sweeps}</table>
<p><a href="summary.json">Full metrics, conditions, coverage, versions and limitations</a></p>
</html>"""


def run_benchmark(manifest_path, output, catalog, data_dir, gallery_dir, index_dir, encoder):
    raw = bounded_read(manifest_path)
    spec = VisualBenchmark.model_validate_json(raw)
    output = output.resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new evaluation directory")
    index, gallery, prepared = preflight(
        spec, manifest_path.parent, catalog, data_dir, gallery_dir, index_dir, encoder
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rxsentinel-visual-evaluation-", dir=output.parent
    ) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        rows = []
        for case, image, kind, evidence in prepared:
            start = perf_counter()
            result = retrieve_photo(
                image,
                index_dir,
                gallery_dir,
                encoder,
                catalog=catalog,
                data_dir=data_dir,
                limit=3,
                minimum_similarity=spec.minimum_similarity,
            )
            latency = (perf_counter() - start) * 1000
            if result.index_id != spec.index_id or result.gallery_id != spec.gallery_id:
                raise ValueError("Index/gallery changed during the benchmark")
            directory = stage / case.case_id
            write_report(result, directory)
            source_name = "source.jpg" if kind == "JPEG" else "source.png"
            (directory / source_name).write_bytes(image)
            page = (directory / "report.html").read_text(encoding="utf-8")
            page = page.replace(
                "</html>",
                f'<h2>Query photo</h2><img src="{source_name}" alt="Query photo" '
                'style="max-width:100%;max-height:450px"></html>',
            )
            (directory / "report.html").write_text(page, encoding="utf-8")
            for record, content in evidence:
                (directory / f"evidence-{record.sha256}.bin").write_bytes(content)
            ids = [c.appearance_id for c in result.candidates]
            row = {
                "case_id": case.case_id,
                "kind": case.kind,
                "expected_appearance_id": case.expected_appearance_id,
                "query_asset_id": case.asset_id,
                "input_sha256": case.input_sha256,
                "capture_session": case.capture_session,
                "conditions": case.conditions,
                "status": result.status,
                "reason": result.reason,
                "processing_status": result.processing.status,
                "processing_rejection_reasons": result.processing.rejection_reasons,
                "candidate_appearance_ids": ids,
                "candidate_scores": [c.cosine_similarity for c in result.candidates],
                "total_candidate_count": result.total_candidate_count,
                "tied_first_candidate_count": result.tied_first_candidate_count,
                "ambiguous": result.tied_first_candidate_count > 1,
                "latency_ms": latency,
                "match_report": f"{case.case_id}/report.html",
            }
            if case.kind == "known":
                row.update(
                    top1_correct=bool(ids and ids[0] == case.expected_appearance_id),
                    top3_contains_truth=case.expected_appearance_id in ids,
                )
            rows.append(row)
        # Queries and local unknown-case evidence are mutable too; recheck before publication.
        preflight(spec, manifest_path.parent, catalog, data_dir, gallery_dir, index_dir, encoder)
        if bounded_read(manifest_path) != raw:
            raise ValueError("Benchmark manifest changed during evaluation")
        conditions = sorted({c for row in rows for c in row["conditions"]})
        truth_ids = sorted({r["expected_appearance_id"] for r in rows if r["kind"] == "known"})
        summary = {
            "version": spec.version,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": spec.dataset_id,
            "scope": spec.scope,
            "partition": spec.partition,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "evaluator_code_sha256": spec.evaluator_code_sha256,
            "gallery_id": gallery.gallery_id,
            "index_id": index.index_id,
            "encoder": index.encoder.model_dump(mode="json"),
            "minimum_similarity": spec.minimum_similarity,
            "python_version": platform.python_version(),
            "gallery_appearance_count": len(gallery.appearances),
            "query_appearance_count": len(truth_ids),
            "gallery_appearances_without_known_queries": sorted(
                {a.appearance_id for a in gallery.appearances} - set(truth_ids)
            ),
            "metrics": metrics(rows),
            "cases": rows,
            "conditions": {
                c: metrics([r for r in rows if c in r["conditions"]]) for c in conditions
            },
            "appearance_coverage": {
                a: metrics([r for r in rows if r["expected_appearance_id"] == a]) for a in truth_ids
            },
            "capture_sessions": {
                s: metrics([r for r in rows if r["capture_session"] == s])
                for s in sorted({r["capture_session"] for r in rows})
            },
            "validation_cutoff_comparison": [
                {"minimum_similarity": c, "metrics": metrics(cutoff_rows(rows, c))}
                for c in spec.validation_cutoffs
            ],
            "current_source_verified": True,
            "thresholds_calibrated": False,
            "photo_identification_available": False,
            "m4_exit_gate_satisfied": False,
            "limitations": [
                "No identity is automatically accepted; accepted identity error rate is unmeasured",
                "Synthetic results do not measure real medication identification accuracy",
                "Cutoff exploration is validation-only and does not select a clinical threshold",
                "Human evidence assertions do not independently prove ground-truth correctness",
                "Capture-session photos are correlated; rates describe these cases only",
                "A test partition label does not prove that the test dataset was never inspected",
                "Query latency includes source verification; initial model loading is excluded",
                "Model suggestions can be returned for unrelated out-of-gallery objects",
            ],
        }
        (stage / "manifest.json").write_bytes(raw)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (stage / "report.html").write_text(render_summary(summary), encoding="utf-8")
        stage.rename(output)
    return summary
