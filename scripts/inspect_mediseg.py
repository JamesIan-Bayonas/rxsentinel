"""Inventory downloaded MEDISEG images and prepare an exploratory M3 sample."""

import argparse
import csv
import hashlib
import html
import io
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rxsentinel.pipelines.reference_files import inspect_image
from rxsentinel.vision.evaluation import EvaluationManifest


def sha(content):
    return hashlib.sha256(content).hexdigest()


def polygon_mask(annotation, width, height):
    mask = Image.new("L", (width, height))
    draw = ImageDraw.Draw(mask)
    polygons = annotation["segmentation"]
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("Expected nonempty COCO polygons")
    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            raise ValueError("Invalid COCO polygon")
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError("Non-finite polygon coordinates")
        draw.polygon([tuple(point) for point in points], fill=255)
    return mask


def contact_sheet(output):
    rows = json.loads((output / "appearances.json").read_text(encoding="utf-8"))
    sheet = Image.new("RGB", (6 * 260, ((len(rows) + 5) // 6) * 260), "#eef2f7")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=13)
    for index, row in enumerate(rows):
        x, y = (index % 6) * 260, (index // 6) * 260
        code = row["id"]
        with Image.open(output / f"images/{code}.jpg") as source:
            with Image.open(output / f"masks/{code}.png") as mask:
                left, top, right, bottom = mask.getbbox()
            crop = source.crop(
                (
                    max(0, left - 10),
                    max(0, top - 10),
                    min(source.width, right + 10),
                    min(source.height, bottom + 10),
                )
            )
            crop.thumbnail((230, 180))
            sheet.paste(crop, (x + (260 - crop.width) // 2, y + (180 - crop.height) // 2))
        label = code + "\n" + "\n".join(textwrap.wrap(row["name"], width=28)[:3])
        draw.multiline_text((x + 12, y + 188), label, fill="#17243a", font=font, spacing=2)
    sheet.save(output / "contact-sheet.png")


def inventory(root, output_name="inspection"):
    source = root / "MEDISEG/32pills"
    metadata = {
        row["id"]: row
        for row in csv.DictReader((root / "MEDISEG/metadata.csv").open(encoding="utf-8-sig"))
    }
    registry_path = root / "registration-checks-current/records.json"
    registry = (
        {row["id"]: row for row in json.loads(registry_path.read_text(encoding="utf-8"))}
        if registry_path.is_file()
        else {}
    )
    annotations = json.loads((source / "annotations.json").read_text(encoding="utf-8"))
    images = {image["id"]: image for image in annotations["images"]}
    if len(images) != len(annotations["images"]):
        raise ValueError("Duplicate image IDs")
    by_image, by_class = defaultdict(list), defaultdict(list)
    for annotation in annotations["annotations"]:
        if annotation["image_id"] not in images:
            raise ValueError("Annotation points to missing image")
        by_image[annotation["image_id"]].append(annotation)
        by_class[annotation["category_id"]].append(annotation)
    assets, hashes = [], Counter()
    for image in images.values():
        path = (source / "images" / image["file_name"]).resolve()
        if not path.is_relative_to((source / "images").resolve()):
            raise ValueError("Image path escapes source directory")
        content = path.read_bytes()
        kind, width, height = inspect_image(content)
        if (width, height) != (image["width"], image["height"]):
            raise ValueError("Image dimensions disagree with COCO metadata")
        digest = sha(content)
        hashes[digest] += 1
        assets.append(
            {
                "image_id": image["id"],
                "file": path.relative_to(root.resolve()).as_posix(),
                "sha256": digest,
                "format": kind,
                "width": width,
                "height": height,
                "bytes": len(content),
                "instance_count": len(by_image[image["id"]]),
                "source_date_captured": image.get("date_captured"),
                "capture_session_verified": False,
            }
        )
    asset_by_id = {asset["image_id"]: asset for asset in assets}
    if Path(output_name).name != output_name:
        raise ValueError("Output must be a direct child directory name")
    output = root / output_name
    if output.exists():
        raise ValueError("Inspection output exists; refusing overwrite")
    output.mkdir()
    (output / "images").mkdir()
    (output / "masks").mkdir()
    cases, rows, selected_hashes, sections = [], [], set(), []
    for category in annotations["categories"]:
        if category["name"] not in metadata:
            if by_class[category["id"]]:
                raise ValueError("Used category lacks drug metadata")
            continue
        details = metadata[category["name"]]
        current = registry.get(category["name"])
        singles = [a for a in by_class[category["id"]] if len(by_image[a["image_id"]]) == 1]
        singles.sort(key=lambda a: (-a["area"], a["image_id"]))
        selected, source_stems = [], set()
        for annotation in singles:
            image = images[annotation["image_id"]]
            stem = image["file_name"].split(".rf.")[0]
            digest = asset_by_id[image["id"]]["sha256"]
            if stem in source_stems or digest in selected_hashes:
                continue
            source_stems.add(stem)
            selected_hashes.add(digest)
            selected.append(annotation)
            if len(selected) == 3:
                break
        if not selected:
            raise ValueError("No unique single-instance sample for a medication class")
        first = selected[0]
        image = images[first["image_id"]]
        path = source / "images" / image["file_name"]
        content = path.read_bytes()
        mask = polygon_mask(first, image["width"], image["height"])
        rgb = np.asarray(Image.open(io.BytesIO(content)).convert("RGB"))
        pixels = rgb[np.asarray(mask) > 0]
        color = np.median(pixels, axis=0).astype(int).tolist()
        contours, _ = cv2.findContours(np.asarray(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        _, (side_a, side_b), _ = cv2.minAreaRect(contour)
        ratio = max(side_a, side_b) / max(min(side_a, side_b), 1)
        label = category["name"]
        image_name, mask_name = f"images/{label}.jpg", f"masks/{label}.png"
        (output / image_name).write_bytes(content)
        mask.save(output / mask_name)
        case = {
            "case_id": label,
            "kind": "positive",
            "image": image_name,
            "input_sha256": sha(content),
            "annotation": mask_name,
            "annotation_sha256": sha((output / mask_name).read_bytes()),
            "capture_session": "MEDISEG-publisher-session-unknown",
            "conditions": ["publisher-real-photo", "single-annotated-instance", label],
        }
        cases.append(case)
        row = {
            **details,
            "category_id": category["id"],
            "annotation_count": len(by_class[category["id"]]),
            "single_instance_image_count": len(singles),
            "representative_image": path.relative_to(root).as_posix(),
            "representative_image_sha256": sha(content),
            "observed_mask_median_rgb": color,
            "observed_rotated_box_aspect_ratio": round(ratio, 3),
            "imprint_front": None,
            "imprint_back": None,
            "front_back_assignment_verified": False,
            "current_registration_verified": False,
            "production_product_link": None,
            "identity_review_status": "publisher_label_only",
            "current_registration_check": current,
            "sample_images": [asset_by_id[a["image_id"]]["file"] for a in selected],
        }
        rows.append(row)
        photos = "".join(
            f'<a href="../{html.escape(name, quote=True)}"><img loading="lazy" '
            f'src="../{html.escape(name, quote=True)}" alt="{html.escape(label)} source photo"></a>'
            for name in row["sample_images"]
        )
        ingredients = ", ".join(filter(None, [details["ingredients/0"], details["ingredients/1"]]))
        sections.append(
            f"<section><h2>{html.escape(label)} — {html.escape(details['name'])}</h2>"
            f"<p><strong>Current registry metadata: "
            f"{html.escape(current['outcome'] if current else 'not checked')}.</strong> "
            f"This does not verify the photographed appearance.</p>"
            f"<p>Ingredients: {html.escape(ingredients)}. Certificate holder: "
            f"{html.escape(details['certificate_holder'])}.</p>"
            f"<p>{len(by_class[category['id']])} annotated instances; {len(singles)} "
            f"single-instance images. Sample mask median RGB: {color}; "
            f"rotated-box aspect ratio: {ratio:.3f}. These are image measurements, "
            f"not verified colour/shape or imprint characteristics.</p>"
            f'<p><a href="{html.escape(details["url"], quote=True)}">Publisher-linked '
            f'Hong Kong product record</a></p><div class="photos">{photos}</div></section>'
        )
    manifest = EvaluationManifest(
        dataset_id="MEDISEG-v2-32-class-development-inspection",
        scope="unverified-local",
        partition="validation",
        annotator="MEDISEG dataset authors; polygon masks rasterized by RxSentinel",
        annotation_basis="Publisher COCO polygons; one largest single-instance example per class. "
        "Development inspection only; source capture sessions and current identities unverified.",
        cases=cases,
    )
    (output / "evaluation-manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    (output / "appearances.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output / "appearances.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "image-inventory.json").write_text(json.dumps(assets, indent=2), encoding="utf-8")
    summary = {
        "dataset_version": 2,
        "market": "Hong Kong",
        "license": "CC BY 4.0",
        "image_count": len(images),
        "unique_image_sha256_count": len(hashes),
        "duplicate_image_file_count": sum(n - 1 for n in hashes.values()),
        "decoded_image_count": len(assets),
        "instance_annotation_count": len(annotations["annotations"]),
        "medication_class_count": len(rows),
        "single_instance_image_count": sum(len(by_image[i]) == 1 for i in images),
        "multi_instance_image_count": sum(len(by_image[i]) > 1 for i in images),
        "zero_annotation_image_count": sum(not by_image[i] for i in images),
        "representative_image_count": sum(len(row["sample_images"]) for row in rows),
        "exploratory_evaluation_case_count": len(cases),
        "source_annotation_sha256": sha((source / "annotations.json").read_bytes()),
        "source_metadata_sha256": sha((root / "MEDISEG/metadata.csv").read_bytes()),
        "registration_check_outcomes": dict(Counter(r["outcome"] for r in registry.values())),
        "registration_check_sha256": sha(registry_path.read_bytes()) if registry else None,
        "capture_sessions_verified": False,
        "production_catalog_imported": False,
        "m1_reference_gate_satisfied": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RxSentinel downloaded pill appearance collection</title>
<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem;
background:#f5f7fb;color:#17243a}
section{background:white;padding:1.5rem;margin:1rem 0;border-radius:12px}a{color:#1459a8}
.photos{display:flex;flex-wrap:wrap;gap:12px}.photos img{width:300px;max-width:100%;height:auto}
pre{white-space:pre-wrap}h2{font-size:1.2rem}</style>
<h1>Downloaded pill appearance collection: MEDISEG v2</h1>
<p>Photographs and medication labels supplied by MEDISEG authors.
Hong Kong scope; no U.S. NDC linkage. Imprints, front/back pairing, current registration
and independent capture sessions remain unverified. This collection supports research inspection
and exploratory segmentation; it does not pass the production reference gate.</p>
<p>Attribution: Wai Ip Chu, Shashi Hirani, Giacomo Tarroni and Ling Li, MEDISEG;
<a href="https://doi.org/10.25383/city.28574786.v2">dataset DOI</a>,
<a href="https://arxiv.org/abs/2603.10825">publication</a>;
<a href="../MEDISEG/LICENSE">CC BY 4.0 license</a>. Source photographs are unaltered;
binary evaluation masks were derived from the authors' polygons.</p>
<p><a href="appearances.csv">Medication and appearance inventory CSV</a> ·
<a href="appearances.json">Detailed provenance JSON</a> · <a href="summary.json">Counts</a></p>
"""
        + f"<pre>{html.escape(json.dumps(summary, indent=2))}</pre>"
        + "".join(sections)
        + "</html>"
    )
    (output / "report.html").write_text(page, encoding="utf-8")
    contact_sheet(output)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/external/mediseg-v2"))
    parser.add_argument("--output-name", default="inspection")
    args = parser.parse_args()
    inventory(args.directory.resolve(), args.output_name)


if __name__ == "__main__":
    main()
