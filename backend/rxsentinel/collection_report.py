"""Local visual inspection page and an explicitly pending review draft."""

import html
import json
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from rxsentinel.collection_support import collection_status, file_valid


def write_inspection(data_dir, appearance, assets, products, all_assets):
    state = collection_status(data_dir, appearance, assets, products, all_assets)
    directory = data_dir / "collection-reviews"
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        f"{appearance.appearance_id.split(':', 1)[1]}-{state['fingerprint'][:12]}-{uuid4().hex[:6]}"
    )
    report_path = directory / f"{name}.html"
    draft_path = directory / f"{name}.review.json"
    escape = html.escape

    def relative(path):
        return escape(os.path.relpath(data_dir / path, directory).replace("\\", "/"), quote=True)

    cards = []
    for asset in assets:
        status = next(s for s in state["assets"] if s["asset_id"] == asset.asset_id)
        photo = (
            f'<img src="{relative(asset.local_path)}" alt="Submitted {escape(asset.side)} photo">'
            if file_valid(data_dir, asset, "assets", image=True)
            else "<p>Photo integrity check failed.</p>"
        )
        cards.append(
            f"<article><h2>{escape(asset.partition)} · {escape(asset.side)}</h2>{photo}"
            f"<p>Photographer: {escape(asset.photographer or 'unknown')}; "
            f"origin: {escape(asset.photo_origin or 'unknown')}; "
            f"session: {escape(asset.capture_session or 'missing')}</p>"
            f"<p>Reuse: {escape(asset.reuse_status)}; eligible: {status['eligible']}</p>"
            f"<p>{escape(', '.join(status['missing_requirements']))}</p>"
            f"<details><summary>Asset ID and checksum</summary><pre>{escape(asset.asset_id)}\n"
            f"{escape(asset.sha256)}</pre></details></article>"
        )
    evidence = []
    for entry in appearance.submitted_evidence:
        link = (
            f'<a href="{relative(entry.local_path)}">Open evidence</a>'
            if file_valid(data_dir, entry, "evidence")
            else "Evidence integrity check failed"
        )
        evidence.append(
            f"<li><strong>{escape(entry.evidence_id)}</strong> · {escape(entry.purpose)}: "
            f"{escape(entry.description)} — {link}</li>"
        )
    context = escape(json.dumps([p.model_dump(mode="json") for p in products], indent=2))
    appearance_text = escape(
        json.dumps(
            {
                k: getattr(appearance, k)
                for k in (
                    "imprint_front",
                    "imprint_back",
                    "blank_imprint_verified",
                    "color",
                    "shape",
                    "manufacturer",
                )
            },
            indent=2,
        )
    )
    draft = {
        "appearance_id": appearance.appearance_id,
        "expected_fingerprint": state["fingerprint"],
        "reviewer": "",
        "identity_decision": "pending",
        "reuse_decision": "pending",
        "basis": "",
        "identity_evidence_ids": [
            e.evidence_id
            for e in appearance.submitted_evidence
            if e.purpose in {"packaging", "product_record"}
        ],
        "reuse_evidence_ids": [
            e.evidence_id
            for e in appearance.submitted_evidence
            if e.purpose in {"ownership", "license"}
        ],
        "reuse_asset_ids": [a.asset_id for a in assets],
        "valid_until": (datetime.now(UTC) + timedelta(days=90)).isoformat(),
    }
    draft_path.write_text(json.dumps(draft, indent=2), encoding="utf-8")
    markup = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>RxSentinel collection review</title><style>"
        "body{font:16px system-ui;color:#173042;background:#f4f7fa;max-width:1000px;"
        "margin:40px auto;padding:0 20px}article,section{background:white;padding:20px;"
        "margin:20px 0;border:1px solid #cbd8e3;border-radius:12px}img{max-width:100%;"
        "max-height:340px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}"
        "</style><h1>Collection review</h1>"
        f"<p>{escape(appearance.appearance_id)} · reference ready: {state['reference_ready']}</p>"
        "<p>Review the exact product, strength, form, manufacturer, and both tablet faces against "
        "packaging or records. Review permission separately for every selected image. "
        "Decisions are reviewer assertions for this research collection.</p>"
        f"<p>Missing requirements: {escape(', '.join(state['missing_requirements']) or 'none')}</p>"
        "<section><h2>Submitted appearance</h2>"
        f"<pre>{appearance_text}</pre>"
        "<details><summary>Exact catalog product context</summary>"
        f"<pre>{context}</pre></details></section>"
        + "".join(cards)
        + "<section><h2>Evidence</h2><ul>"
        + "".join(evidence)
        + "</ul></section>"
        "<section><h2>Record a decision</h2><p>Open the pending draft, enter your name and actual "
        "review basis, choose approve/reject/pending separately, "
        "and select the evidence and asset IDs "
        "that your decision covers. Then submit it with the collection review command.</p>"
        f'<a href="{escape(draft_path.name, quote=True)}">Open pending review draft</a>'
        "<p>This page records no approval. Changed files or collection versions require inspection "
        "again. Reference readiness does not enable photo identification.</p></section></html>"
    )
    report_path.write_text(markup, encoding="utf-8")
    return {
        "inspection_report": str(report_path),
        "pending_review_draft": str(draft_path),
        "fingerprint": state["fingerprint"],
        "reference_ready": state["reference_ready"],
    }
