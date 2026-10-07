"""Publish a complete local processing report without overwriting previous runs."""

import hashlib
import html
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from rxsentinel.vision.single import ProcessingReport


def publish_report(
    output: Path,
    report: ProcessingReport,
    artifacts: dict[str, np.ndarray],
    source: bytes,
) -> Path:
    output = output.resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rxsentinel-processing-", dir=output.parent) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        original_name = "source.jpg" if report.input_format == "JPEG" else "source.png"
        (stage / original_name).write_bytes(source)
        report.artifacts["source_bytes"] = original_name
        report.artifact_sha256[original_name] = hashlib.sha256(source).hexdigest()
        for name, pixels in artifacts.items():
            filename = f"{name}.png"
            Image.fromarray(pixels).save(stage / filename)
            report.artifacts[name] = filename
            report.artifact_sha256[filename] = hashlib.sha256(
                (stage / filename).read_bytes()
            ).hexdigest()
        (stage / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        entries = "".join(
            f"<li>{html.escape(item)}</li>" for item in report.rejection_reasons + report.warnings
        )
        figures = "".join(
            f"<figure><figcaption>{html.escape(name)}</figcaption>"
            f'<img src="{filename}" alt="{html.escape(name)}"></figure>'
            for name, filename in report.artifacts.items()
            if name != "source_bytes"
        )
        detail = html.escape(
            json.dumps(
                {
                    "metrics": report.metrics,
                    "crop_box": report.crop_box,
                    "letterbox": report.letterbox,
                    "evaluation": report.annotation_evaluation,
                },
                indent=2,
            )
        )
        page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RxSentinel experimental photo processing</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}}
figure{{display:inline-block;vertical-align:top;margin:1rem;max-width:460px}}
img{{max-width:100%;max-height:450px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>Experimental single-object processing: {report.status}</h1>
<p>No medication identity, dosage or safety conclusion is generated.</p>
<p>Input SHA-256: <code>{report.input_sha256}</code>. Baseline: {report.baseline_version}.</p>
<ul>{entries}</ul><p><a href="report.json">Full versioned JSON report and file checksums</a></p>
<pre>{detail}</pre>{figures}</html>"""
        (stage / "report.html").write_text(page, encoding="utf-8")
        # Atomic directory publication on the same filesystem; Windows refuses an existing target.
        stage.rename(output)
    return output / "report.html"
