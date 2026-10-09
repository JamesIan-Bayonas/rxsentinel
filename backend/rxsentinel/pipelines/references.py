"""Compare archived leads against latest available SPLs, without promoting identities."""

import argparse
import asyncio
import hashlib
import html
import json
import re
import zipfile
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import httpx
from defusedxml import ElementTree as ET
from sqlalchemy.exc import SQLAlchemyError

from rxsentinel.catalog import configured_catalog
from rxsentinel.pipelines.reference_files import (
    ARCHIVE_URL,
    RangeReader,
    archived_image,
    save_image,
)
from rxsentinel.schemas import EvidenceSource, ImageAsset, LabelIngredient, ReferenceCheck

LABEL_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/"
NS = {"h": "urn:hl7-org:v3"}
CHARACTERS = {
    "SPLIMPRINT": "imprint",
    "SPLCOLOR": "color",
    "SPLSHAPE": "shape",
    "SPLSCORE": "score_marks",
}


def text_at(node, path):
    found = node.find(path, NS)
    return " ".join("".join(found.itertext()).split()) if found is not None else None


def attribute_at(node, path, key):
    found = node.find(path, NS)
    return found.get(key) if found is not None else None


def canonical(value):
    return " ".join(value.casefold().split()) if value else None


def strength_key(value):
    if not value:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)/(\d+(?:\.\d+)?)", value.strip())
    if not match:
        return None  # Unsupported units/denominators stay incomplete; do not guess.
    amount, unit, denominator = match.groups()
    if Decimal(denominator) != 1:
        return None
    return Decimal(amount), unit.casefold()


def compare_value(old, new):
    if not old or not new:
        return "missing"
    return "match" if canonical(old) == canonical(new) else "mismatch"


def compare_label(xml: bytes, set_id: str, product, appearance, snapshot_id, source):
    root = ET.fromstring(xml)
    if root.tag != "{urn:hl7-org:v3}document":
        raise ValueError("Expected an SPL XML document")
    actual_set = attribute_at(root, "h:setId", "root")
    if actual_set != set_id:
        raise ValueError("Returned SPL set ID does not match the requested label")
    check = ReferenceCheck(
        check_id=f"{snapshot_id}:{appearance.appearance_id}",
        snapshot_id=snapshot_id,
        appearance_id=appearance.appearance_id,
        product_id=product.product_id,
        product_ndc=product.product_ndc,
        set_id=set_id,
        source=source,
        xml_sha256=hashlib.sha256(xml).hexdigest(),
        label_version=attribute_at(root, "h:versionNumber", "value"),
        label_effective_time=attribute_at(root, "h:effectiveTime", "value"),
        labeler=text_at(root, "h:author/h:assignedEntity/h:representedOrganization/h:name"),
        outcome="not_found",
        notes=[
            "Latest available label for this set ID; availability does not prove marketing status.",
            "Automated consistency evidence is not an identity review or image reuse permission.",
            "Pillbox archival images remain inspection-only; ineligible for pill identification.",
        ],
    )
    matches = []
    for outer in root.findall(".//h:manufacturedProduct", NS):
        inner = outer.find("h:manufacturedProduct", NS)
        if inner is not None and attribute_at(inner, "h:code", "code") == product.product_ndc:
            matches.append((outer, inner))
    if len(matches) != 1:
        check.outcome = "ambiguous" if matches else "not_found"
        check.notes.append(
            "Exact product NDC must occur once in this label; no package-code inference."
        )
        return check
    outer, inner = matches[0]
    check.dosage_form = attribute_at(inner, "h:formCode", "displayName")
    for ingredient in inner.findall("h:ingredient", NS):
        if ingredient.get("classCode") not in {"ACTIB", "ACTIM", "ACTIR"}:
            continue
        name = text_at(ingredient, "h:ingredientSubstance/h:name")
        numerator = ingredient.find("h:quantity/h:numerator", NS)
        denominator = ingredient.find("h:quantity/h:denominator", NS)
        strength = None
        if (
            numerator is not None
            and denominator is not None
            and denominator.get("unit", "1") == "1"
            and numerator.get("value")
            and numerator.get("unit")
            and denominator.get("value")
        ):
            strength = (
                f"{numerator.get('value')} {numerator.get('unit')}/{denominator.get('value')}"
            )
        if name:
            check.ingredients.append(LabelIngredient(name=name, strength=strength))
    for characteristic in outer.findall("h:subjectOf/h:characteristic", NS):
        field = CHARACTERS.get(attribute_at(characteristic, "h:code", "code"))
        if not field:
            continue
        value = characteristic.find("h:value", NS)
        if value is None:
            continue
        contents = (
            value.get("displayName") or value.get("value") or text_at(characteristic, "h:value")
        )
        if contents:
            if field in check.characteristics:
                raise ValueError("Multiple physical values require a dedicated parser/review")
            check.characteristics[field] = contents
    comparisons = {
        "dosage_form": compare_value(product.dosage_form, check.dosage_form),
        "labeler": "missing"
        if not product.manufacturers or not check.labeler
        else "match"
        if canonical(check.labeler) in {canonical(m) for m in product.manufacturers}
        else "mismatch",
    }
    expected = Counter(canonical(i.source_name) for i in product.ingredients)
    observed = Counter(canonical(i.name) for i in check.ingredients)
    comparisons["active_ingredients"] = (
        "missing" if not observed else ("match" if expected == observed else "mismatch")
    )
    old_strengths = [
        (canonical(i.source_name), strength_key(i.strength)) for i in product.ingredients
    ]
    new_strengths = [(canonical(i.name), strength_key(i.strength)) for i in check.ingredients]
    comparisons["strengths"] = (
        "missing"
        if not new_strengths or any(v is None for _, v in old_strengths + new_strengths)
        else ("match" if Counter(old_strengths) == Counter(new_strengths) else "mismatch")
    )
    for field in CHARACTERS.values():
        old = getattr(appearance, "imprint_unassigned" if field == "imprint" else field)
        comparisons[field] = compare_value(old, check.characteristics.get(field))
    check.comparisons = comparisons
    check.outcome = (
        "mismatch"
        if "mismatch" in comparisons.values()
        else "incomplete"
        if "missing" in comparisons.values()
        else "consistent"
    )
    return check


async def fetch_label(client, url):
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        if "xml" not in response.headers.get("content-type", "").casefold():
            raise ValueError("Provider returned a non-XML label response")
        content = bytearray()
        async for chunk in response.aiter_bytes():
            content.extend(chunk)
            if len(content) > 4 * 1024**2:
                raise ValueError("Label exceeds the supported XML size")
        return bytes(content)


def inspection_report(data_dir, checks, assets):
    escape = html.escape
    cards = []
    for check in checks:
        linked = [a for a in assets if a.appearance_id == check.appearance_id]
        figures = "".join(
            f'<figure><img src="{escape(a.local_path, quote=True)}" '
            'alt="Archived inspection image">'
            f"<figcaption>{escape(a.archive_member or a.asset_id)} · {a.width} × {a.height} · "
            f"reuse {escape(a.reuse_status)} · {escape(a.intended_use)}</figcaption></figure>"
            for a in linked
        )
        comparisons = "".join(
            f"<li>{escape(k)}: {escape(v)}</li>" for k, v in check.comparisons.items()
        )
        cards.append(
            f"<article><h2>{escape(check.product_ndc)} — {escape(check.outcome)}</h2>"
            f"<p>Label version {escape(check.label_version or 'unknown')}; "
            f"effective {escape(check.label_effective_time or 'unknown')}; "
            f"{escape(check.labeler or 'unknown labeler')}</p>"
            f"<p>{escape(json.dumps(check.characteristics))}</p><ul>{comparisons}</ul>{figures}"
            f'<p><a href="{escape(str(check.source.url), quote=True)}">Source XML</a></p></article>'
        )
    markup = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>RxSentinel reference inspection</title><style>"
        "body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;"
        "background:#f5f7fa;"
        "color:#172b3a}article{background:white;border:1px solid #ccd6df;border-radius:12px;"
        "padding:20px;margin:20px 0}img{max-width:100%;max-height:320px}figcaption{font-size:14px}"
        "</style><h1>Reference inspection</h1><p>Archived images are for inspection only. "
        "Label consistency does not confirm a loose pill. "
        "Identity and reuse reviews remain open.</p>" + "".join(cards) + "</html>"
    )
    (data_dir / "reference-review.html").write_text(markup, encoding="utf-8")


async def check_references(data_dir: Path, inspect_archived_images=False):
    catalog = configured_catalog(data_dir)
    if not catalog.store:
        raise ValueError("Configure the relational database before checking references")
    directories = sorted((data_dir / "source-discovery").glob("pillbox-discovery-*/manifest.json"))
    if not directories:
        raise ValueError("Run appearance discovery first")
    discovery_dir = directories[-1].parent
    raw = (discovery_dir / "raw.json").read_bytes()
    manifest = json.loads((discovery_dir / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != manifest["raw_sha256"]:
        raise ValueError("Archived discovery snapshot failed its checksum check")
    records = json.loads(raw)["records"]
    if len(records) > 100:
        raise ValueError("This reference feasibility command supports at most 100 records")
    appearances = {a.appearance_id: a for a in catalog.store.appearances()}
    products = {p.product_id: p for p in catalog.all()}
    now = datetime.now(UTC)
    snapshot_id = now.strftime("references-%Y%m%dT%H%M%S%fZ")
    directory = data_dir / "reference-checks" / snapshot_id
    directory.mkdir(parents=True, exist_ok=False)
    checks, label_files, image_results, assets = [], [], [], []
    cache = {}
    async with httpx.AsyncClient(timeout=40, follow_redirects=False) as client:
        for record in records:
            appearance = appearances[f"pillbox:{record['id']}"]
            if len(appearance.product_ids) != 1:
                raise ValueError("Archived lead must have exactly one product for this check")
            product = products[appearance.product_ids[0]]
            if record["product_code"] != product.product_ndc:
                raise ValueError("Discovery product code no longer matches its catalog link")
            set_id = str(UUID(record["setid"]))
            url = LABEL_BASE + set_id + ".xml"
            source = EvidenceSource(
                name="DailyMed latest available SPL XML",
                url=url,
                retrieved_at=now,
                section=f"Product NDC {product.product_ndc}",
            )
            if set_id not in cache:
                try:
                    cache[set_id] = await fetch_label(client, url)
                except (httpx.HTTPError, ValueError):
                    cache[set_id] = None
                if cache[set_id] is not None:
                    filename = f"{set_id}.xml"
                    (directory / filename).write_bytes(cache[set_id])
                    label_files.append(
                        {
                            "file": filename,
                            "url": url,
                            "sha256": hashlib.sha256(cache[set_id]).hexdigest(),
                        }
                    )
            if cache[set_id] is None:
                check = ReferenceCheck(
                    check_id=f"{snapshot_id}:{appearance.appearance_id}",
                    snapshot_id=snapshot_id,
                    appearance_id=appearance.appearance_id,
                    product_id=product.product_id,
                    product_ndc=product.product_ndc,
                    set_id=set_id,
                    source=source,
                    outcome="provider_error",
                    notes=["Latest label unavailable; no identity promoted."],
                )
            else:
                # Malformed XML or contradictory SPL structure aborts the run before DB writes.
                check = compare_label(
                    cache[set_id], set_id, product, appearance, snapshot_id, source
                )
            checks.append(check)
    archive_metadata = None
    if inspect_archived_images:
        with httpx.Client(timeout=40, follow_redirects=False) as client:
            reader = RangeReader(client)
            with zipfile.ZipFile(reader) as archive:
                for record in records:
                    if str(record.get("has_image", "")).casefold() != "true":
                        continue
                    identifier = record.get("splimage") or ""
                    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", identifier):
                        raise ValueError("Unsupported archived image identifier")
                    appearance_id = f"pillbox:{record['id']}"
                    try:
                        member, content = archived_image(archive, identifier)
                        relative, sha, width, height = save_image(data_dir, content)
                    except (ValueError, OSError, zipfile.BadZipFile) as error:
                        image_results.append(
                            {
                                "appearance_id": appearance_id,
                                "status": "unavailable",
                                "reason": type(error).__name__,
                            }
                        )
                        continue
                    asset = ImageAsset(
                        asset_id=f"pillbox:{record['id']}:{sha[:20]}",
                        appearance_id=appearance_id,
                        local_path=relative,
                        sha256=sha,
                        width=width,
                        height=height,
                        side="unknown",
                        archive_member=member,
                        intended_use="inspection_only",
                        source=EvidenceSource(
                            name="Pillbox archived image; "
                            f"origin {record.get('image_source', 'unknown')}",
                            url=ARCHIVE_URL,
                            retrieved_at=now,
                            section=member,
                        ),
                    )
                    assets.append(asset)
                    image_results.append(
                        {
                            "appearance_id": appearance_id,
                            "status": "downloaded",
                            "asset_id": asset.asset_id,
                            "sha256": sha,
                            "member": member,
                        }
                    )
            archive_metadata = {
                "url": ARCHIVE_URL,
                "size": reader.size,
                "etag": reader.etag,
                "last_modified": reader.modified,
                "bytes_transferred": reader.transferred,
            }
    summary = {
        "snapshot_id": snapshot_id,
        "retrieved_at": now.isoformat(),
        "discovery_snapshot_id": manifest["snapshot_id"],
        "checks": len(checks),
        "outcomes": dict(Counter(c.outcome for c in checks)),
        "label_files": label_files,
        "images": image_results,
        "archive": archive_metadata,
        "inspection_images_downloaded": len(assets),
        "identities_promoted": 0,
        "reuse_permissions_granted": 0,
        "visual_identification_ready": False,
        "archive_use_notice": "https://www.nlm.nih.gov/pubs/techbull/ja20/ja20_pillbox_discontinue.html",
        "copyright_policy": "https://www.nlm.nih.gov/web_policies.html#copyright",
    }
    (directory / "checks.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in checks], indent=2), encoding="utf-8"
    )
    summary["checks_sha256"] = hashlib.sha256((directory / "checks.json").read_bytes()).hexdigest()
    (directory / "assets.json").write_text(
        json.dumps([a.model_dump(mode="json") for a in assets], indent=2), encoding="utf-8"
    )
    summary["assets_sha256"] = hashlib.sha256((directory / "assets.json").read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    catalog.store.import_reference_checks(checks, assets, summary)
    inspection_report(data_dir, checks, assets)
    catalog.store.engine.dispose()
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Check current-label evidence and inspect archived images"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--inspect-archived-images",
        action="store_true",
        help="Download flagged archived images for inspection only, never identification",
    )
    args = parser.parse_args()
    try:
        result = asyncio.run(check_references(args.data_dir, args.inspect_archived_images))
    except (ValueError, OSError, httpx.HTTPError, zipfile.BadZipFile, SQLAlchemyError) as error:
        parser.exit(
            1, f"Reference check failed: {type(error).__name__}. No identities are promoted.\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
