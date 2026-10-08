"""Fixed-gallery evaluation of manual candidate suggestions, never accepted identities."""

import hashlib
import html
import json
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from rxsentinel.matching import MATCHER_VERSION, MatchQuery, match_observations, snapshot
from rxsentinel.pipelines.match import write_report
from rxsentinel.pipelines.photo import bounded_read
from rxsentinel.schemas import FileEvidence, StrictModel
from rxsentinel.vision.evaluation import checked_hash, local_file
from rxsentinel.vision.single import decode_photo

MAX_BENCHMARK_BYTES = 128 * 1024**2
MAX_BENCHMARK_PIXELS = 100_000_000


def matcher_code_sha256():
    from rxsentinel import collection_support, matching, schemas

    digest = hashlib.sha256()
    for module in (matching, collection_support, schemas):
        digest.update(module.__name__.encode())
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


class MatchingCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    kind: Literal["known", "unknown"]
    expected_appearance_id: str | None = None
    query: MatchQuery
    asset_id: str | None = None
    image: str | None = None
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    capture_session: str = Field(min_length=1, max_length=160)
    conditions: list[str] = Field(min_length=1, max_length=20)
    evidence: list[FileEvidence] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def valid_case(self):
        if self.query.limit != 3:
            raise ValueError("Benchmark queries must request exactly three candidates")
        if self.kind == "known" and not (
            self.expected_appearance_id and self.asset_id and not self.image
        ):
            raise ValueError("Known cases require a truth appearance and registered query asset")
        if self.kind == "unknown" and not (
            self.image and not self.asset_id and not self.expected_appearance_id
        ):
            raise ValueError("Unknown cases require a local image and no asserted gallery identity")
        if self.kind == "unknown" and not (
            {e.purpose for e in self.evidence} & {"ownership", "license"}
            and {e.purpose for e in self.evidence} & {"packaging", "product_record"}
        ):
            raise ValueError("Unknown cases need reuse and identity-context evidence")
        if len({e.evidence_id for e in self.evidence}) != len(self.evidence):
            raise ValueError("Evidence IDs must be unique within each case")
        if any(not c.strip() for c in self.conditions):
            raise ValueError("Condition labels cannot be blank")
        return self


class MatchingBenchmark(StrictModel):
    version: Literal["m4-manual-evaluation-1"] = "m4-manual-evaluation-1"
    matcher_version: Literal[MATCHER_VERSION] = MATCHER_VERSION
    matcher_code_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_id: str = Field(min_length=1, max_length=160)
    scope: Literal["synthetic", "reviewed-collection"]
    partition: Literal["validation", "test"]
    reference_snapshot_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    annotator: str = Field(min_length=1, max_length=160)
    annotation_basis: str = Field(min_length=1, max_length=2000)
    observations_transcribed_from_query: Literal[True]
    synthetic_catalog: str | None = None
    cases: list[MatchingCase] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def valid_dataset(self):
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("Case IDs must be unique")
        if len({c.input_sha256 for c in self.cases}) != len(self.cases):
            raise ValueError("Duplicate query photos cannot count as independent cases")
        if (self.scope == "synthetic") != bool(self.synthetic_catalog):
            raise ValueError("Only synthetic runs require a local synthetic_catalog path")
        return self


def preflight(spec, root, catalog, data_dir):
    if spec.matcher_code_sha256 != matcher_code_sha256():
        raise ValueError("Matcher implementation differs from the frozen benchmark manifest")
    appearances, assets, products, fingerprint, states = snapshot(catalog, data_dir)
    if fingerprint != spec.reference_snapshot_sha256:
        raise ValueError("Reference snapshot differs from the frozen benchmark manifest")
    gallery = match_observations(catalog, data_dir, MatchQuery())
    if not gallery.eligible_appearance_count:
        raise ValueError("Benchmark requires a nonempty eligible reference gallery")
    if gallery.reference_snapshot_sha256 != fingerprint:
        raise ValueError("Reference snapshot changed during benchmark preflight")
    if spec.scope == "synthetic":
        database = catalog.store.engine.url.database if catalog.store else None
        if (
            not database
            or catalog.store.engine.dialect.name != "sqlite"
            or Path(database).resolve() != (root / spec.synthetic_catalog).resolve()
            or not Path(database).resolve().is_relative_to(root.resolve())
            or any(not p.product_id.startswith("synthetic:") for p in products.values())
        ):
            raise ValueError("Synthetic benchmarks require their isolated synthetic SQLite catalog")
    indexed_assets = {a.asset_id: a for a in assets}
    prepared, total_bytes, total_pixels = [], 0, 0
    for case in spec.cases:
        evidence = []
        if case.kind == "known":
            asset = indexed_assets.get(case.asset_id)
            if not asset or asset.appearance_id != case.expected_appearance_id:
                raise ValueError("Query asset does not belong to the annotated appearance")
            state = states[asset.appearance_id]
            item = next(a for a in state["assets"] if a["asset_id"] == case.asset_id)
            if not state["reference_ready"] or not item["eligible"]:
                raise ValueError("Known query has no eligible reviewed reference/query evidence")
            if asset.partition != spec.partition or asset.capture_session != case.capture_session:
                raise ValueError(
                    "Known query partition or capture session differs from the manifest"
                )
            if asset.sha256 != case.input_sha256:
                raise ValueError("Known query checksum differs from its registered asset")
            image = bounded_read(data_dir / asset.local_path)
        else:
            image = local_file(root, case.image)
            if any(
                a.sha256 == case.input_sha256
                or (a.capture_session == case.capture_session and a.partition != spec.partition)
                for a in assets
            ):
                raise ValueError(
                    "Unknown query duplicates a catalog image or crosses capture partitions"
                )
            for record in case.evidence:
                content = local_file(root, record.local_path)
                checked_hash(content, record.sha256, "Unknown-case evidence")
                evidence.append((record, content))
        checked_hash(image, case.input_sha256, "Query image")
        rgb, kind, _, _ = decode_photo(image)
        total_bytes += len(image) + sum(len(content) for _, content in evidence)
        total_pixels += rgb.shape[0] * rgb.shape[1]
        if total_bytes > MAX_BENCHMARK_BYTES or total_pixels > MAX_BENCHMARK_PIXELS:
            raise ValueError("Benchmark exceeds encoded byte or source pixel budget")
        prepared.append((case, image, kind, evidence))
    return prepared


def summarize(rows):
    known = [r for r in rows if r["kind"] == "known"]
    unknown = [r for r in rows if r["kind"] == "unknown"]
    top1 = sum(r["top1_correct"] for r in known)
    top3 = sum(r["top3_contains_truth"] for r in known)
    abstained = sum(r["status"] == "unknown" for r in known)
    unknown_candidates = sum(r["status"] == "candidates" for r in unknown)
    times = [r["latency_ms"] for r in rows]
    return {
        "case_count": len(rows),
        "known_count": len(known),
        "unknown_count": len(unknown),
        "capture_session_count": len({r["capture_session"] for r in rows}),
        "known_top1_correct_count": top1,
        "known_top3_contains_truth_count": top3,
        "known_top1_recall": top1 / len(known) if known else None,
        "known_top3_recall": top3 / len(known) if known else None,
        "known_abstention_count": abstained,
        "known_abstention_rate": abstained / len(known) if known else None,
        "known_wrong_first_candidate_count": len(known) - abstained - top1,
        "unknown_candidate_return_count": unknown_candidates,
        "unknown_candidate_return_rate": unknown_candidates / len(unknown) if unknown else None,
        "unknown_abstention_count": len(unknown) - unknown_candidates,
        "unknown_abstention_rate": (len(unknown) - unknown_candidates) / len(unknown)
        if unknown
        else None,
        "overall_abstention_rate": sum(r["status"] == "unknown" for r in rows) / len(rows)
        if rows
        else None,
        "ambiguous_result_count": sum(r["ambiguous"] for r in rows),
        "latency_median_ms": float(np.median(times)) if times else None,
        "latency_p95_ms": float(np.percentile(times, 95)) if times else None,
        "accepted_identity_count": 0,
        "accepted_identity_error_rate": None,
    }


def run_benchmark(manifest_path, output, catalog, data_dir):
    raw = bounded_read(manifest_path)
    spec = MatchingBenchmark.model_validate_json(raw)
    output = output.resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    prepared = preflight(spec, manifest_path.parent, catalog, data_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rxsentinel-match-evaluation-", dir=output.parent
    ) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        rows = []
        for case, image, kind, evidence in prepared:
            start = perf_counter()
            result = match_observations(catalog, data_dir, case.query)
            latency = (perf_counter() - start) * 1000
            if (
                result.reference_snapshot_sha256 != spec.reference_snapshot_sha256
                or result.reason == "references_changed_during_search"
            ):
                raise ValueError(
                    "References changed during benchmark; no mixed-snapshot result published"
                )
            directory = stage / case.case_id
            write_report(result, directory)
            (directory / f"source.{'jpg' if kind == 'JPEG' else 'png'}").write_bytes(image)
            source_name = "source.jpg" if kind == "JPEG" else "source.png"
            page_path = directory / "report.html"
            page = page_path.read_text(encoding="utf-8")
            page = page.replace(
                "</html>",
                f"<h2>Query image (not a reference)</h2>"
                f'<img src="{source_name}" alt="Query image" '
                'style="max-width:100%;max-height:450px"></html>',
            )
            page_path.write_text(page, encoding="utf-8")
            for record, content in evidence:
                (directory / f"evidence-{record.sha256}.bin").write_bytes(content)
            ids = [candidate.appearance_id for candidate in result.candidates]
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
                "candidate_appearance_ids": ids,
                "total_candidate_count": result.total_candidate_count,
                "truncated": result.truncated,
                "ambiguous": result.ambiguous,
                "latency_ms": latency,
                "match_report": f"{case.case_id}/report.html",
            }
            if case.kind == "known":
                row.update(
                    top1_correct=bool(ids and ids[0] == case.expected_appearance_id),
                    top3_contains_truth=case.expected_appearance_id in ids[:3],
                )
            rows.append(row)
        # Revalidate query bytes, reviews and all snapshot/partition bindings before publication.
        preflight(spec, manifest_path.parent, catalog, data_dir)
        conditions = sorted({condition for row in rows for condition in row["conditions"]})
        truth_ids = sorted({r["expected_appearance_id"] for r in rows if r["kind"] == "known"})
        summary = {
            "version": spec.version,
            "matcher_version": MATCHER_VERSION,
            "matcher_code_sha256": spec.matcher_code_sha256,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": spec.dataset_id,
            "scope": spec.scope,
            "partition": spec.partition,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "reference_snapshot_sha256": spec.reference_snapshot_sha256,
            "python_version": platform.python_version(),
            "metrics": summarize(rows),
            "cases": rows,
            "conditions": {
                c: summarize([r for r in rows if c in r["conditions"]]) for c in conditions
            },
            "appearance_coverage": {
                a: summarize([r for r in rows if r["expected_appearance_id"] == a])
                for a in truth_ids
            },
            "photo_identification_available": False,
            "m4_exit_gate_satisfied": False,
            "limitations": [
                "Manual transcription benchmark, not photo or OCR accuracy",
                "Unknown candidate returns are not accepted identities",
                "No confidence calibration or automatic acceptance is evaluated",
                "Human evidence assertions cannot prove the correctness of annotations",
                "A test partition label alone does not prove an untouched test protocol",
                "Photos from the same capture session are correlated",
            ],
        }
        (stage / "manifest.json").write_bytes(raw)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        table = "".join(
            f'<tr><td><a href="{r["match_report"]}">{html.escape(r["case_id"])}</a></td>'
            f"<td>{r['kind']}</td><td>{r['status']}</td>"
            f"<td>{html.escape(r['reason'].replace('_', ' '))}</td></tr>"
            for r in rows
        )
        labels = {
            "known_count": "Queries with a reviewed gallery appearance",
            "unknown_count": "Queries labeled outside the gallery",
            "capture_session_count": "Capture sessions represented",
            "known_top1_recall": "Correct appearance ranked first (all known queries)",
            "known_top3_recall": "Correct appearance in the first three (all known queries)",
            "known_abstention_count": "Known queries receiving unknown",
            "known_wrong_first_candidate_count": "Known queries with an incorrect first suggestion",
            "unknown_candidate_return_count": "Out-of-gallery queries receiving suggestions",
            "ambiguous_result_count": "Queries with ambiguous candidate results",
            "accepted_identity_count": "Automatically confirmed identities",
        }
        metric_rows = ""
        for key, label in labels.items():
            value = summary["metrics"][key]
            display = "Not measured (no examples)" if value is None else str(value)
            if value is not None and key.endswith("recall"):
                display = f"{value:.1%}"
            metric_rows += f"<tr><td>{label}</td><td>{display}</td></tr>"
        page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RxSentinel manual matching evaluation</title><style>
body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}}
td,th{{text-align:left;padding:.6rem;border-bottom:1px solid #aaa}}
pre{{white-space:pre-wrap}}</style>
<h1>Manual matching evaluation: {html.escape(spec.dataset_id)}</h1>
<p>Scope: {spec.scope}; partition: {spec.partition}. No medication identity is confirmed.</p>
<p>This measures manually transcribed observations, not recognition from photos.</p>
<table>{metric_rows}</table>
<table><tr><th>Case</th><th>Expected coverage</th><th>Result</th><th>Reason</th></tr>{table}</table>
<p><a href="summary.json">Full counts, condition results, provenance and limitations</a></p>
</html>"""
        (stage / "report.html").write_text(page, encoding="utf-8")
        stage.rename(output)
    return summary
