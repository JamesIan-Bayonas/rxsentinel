"""Portable, versioned reference inputs. No vectors, training or identity inference."""

import html
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from rxsentinel.collection_support import collection_status, digest, file_valid
from rxsentinel.matching import snapshot
from rxsentinel.pipelines.photo import bounded_read
from rxsentinel.schemas import Appearance, ImageAsset, Product, StrictModel
from rxsentinel.vision.evaluation import checked_hash, local_file
from rxsentinel.vision.single import decode_photo

MAX_GALLERY_BYTES = 1024**3


class ReferenceGallery(StrictModel):
    version: Literal["reference-gallery-1"] = "reference-gallery-1"
    gallery_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    scope: Literal["synthetic", "reviewed-collection"]
    created_at: datetime
    valid_until: datetime
    source_reference_snapshot_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_collection_fingerprints: dict[str, str]
    products: list[Product] = Field(min_length=1, max_length=1000)
    appearances: list[Appearance] = Field(min_length=1, max_length=500)
    reference_assets: list[ImageAsset] = Field(min_length=1, max_length=2000)
    files: dict[str, str] = Field(max_length=5000)

    @model_validator(mode="after")
    def validate_structure(self):
        for rows, key in (
            (self.products, "product_id"),
            (self.appearances, "appearance_id"),
            (self.reference_assets, "asset_id"),
        ):
            if len({getattr(r, key) for r in rows}) != len(rows):
                raise ValueError("Gallery record IDs must be unique")
        if (
            not self.created_at.tzinfo
            or not self.valid_until.tzinfo
            or self.created_at >= self.valid_until
        ):
            raise ValueError(
                "Gallery dates must be timezone-aware with a future expiry at creation"
            )
        if set(self.source_collection_fingerprints) != {a.appearance_id for a in self.appearances}:
            raise ValueError("Source collection fingerprints must cover every exported appearance")
        if self.scope == "synthetic" and any(
            not p.product_id.startswith("synthetic:") for p in self.products
        ):
            raise ValueError("Synthetic galleries require explicitly synthetic products")
        return self


def gallery_id(manifest):
    payload = manifest.model_dump(mode="json")
    payload.pop("gallery_id")
    return digest(payload)


def without_local_paths(value):
    if isinstance(value, dict):
        return {k: without_local_paths(v) for k, v in value.items() if k != "local_path"}
    if isinstance(value, list):
        return [without_local_paths(v) for v in value]
    return value


def validate_gallery(directory, *, catalog=None, data_dir=None):
    manifest = ReferenceGallery.model_validate_json(bounded_read(directory / "manifest.json"))
    if gallery_id(manifest) != manifest.gallery_id:
        raise ValueError("Gallery manifest fingerprint mismatch")
    if manifest.valid_until <= datetime.now(UTC):
        raise ValueError("Gallery review window has expired")
    expected = {a.local_path: a.sha256 for a in manifest.reference_assets}
    for appearance in manifest.appearances:
        for evidence in appearance.identity_evidence:
            expected[evidence.local_path] = evidence.sha256
    for asset in manifest.reference_assets:
        if asset.partition != "reference" or asset.intended_use != "research_reference":
            raise ValueError("Only reference images can be included in a gallery")
        for evidence in asset.reuse_evidence:
            expected[evidence.local_path] = evidence.sha256
        if not file_valid(directory, asset, "assets", image=True):
            raise ValueError("Gallery image failed integrity or path checks")
        pixels, _, size, _ = decode_photo(local_file(directory, asset.local_path))
        if (asset.width, asset.height) != size:
            raise ValueError("Reference image dimensions differ from catalog metadata")
        del pixels
    if manifest.files != expected:
        raise ValueError("Gallery file inventory differs from its reference/evidence records")
    total = 0
    for path, sha in expected.items():
        content = local_file(directory, path)
        checked_hash(content, sha, "Gallery file")
        total += len(content)
        if total > MAX_GALLERY_BYTES:
            raise ValueError("Gallery exceeds byte budget")
    used_products, used_assets = set(), set()
    for appearance in manifest.appearances:
        linked = [p for p in manifest.products if p.product_id in appearance.product_ids]
        assets = [
            a for a in manifest.reference_assets if a.appearance_id == appearance.appearance_id
        ]
        state = collection_status(directory, appearance, assets, linked, manifest.reference_assets)
        if not appearance.appearance_id.startswith("local:") or not state["reference_ready"]:
            raise ValueError("Exported appearance does not have intact current reviewed references")
        if not all(a["eligible"] for a in state["assets"]):
            raise ValueError("Gallery contains an ineligible image")
        used_products.update(appearance.product_ids)
        used_assets.update(a.asset_id for a in assets)
    if used_products != {p.product_id for p in manifest.products} or used_assets != {
        a.asset_id for a in manifest.reference_assets
    }:
        raise ValueError("Gallery contains orphaned products or images")
    if manifest.valid_until != min(a.valid_until for a in manifest.appearances):
        raise ValueError("Gallery expiry differs from its earliest identity review expiry")
    current = False
    if catalog is not None:
        if data_dir is None:
            raise ValueError("Current-source validation requires the source data directory")
        source_appearances, source_assets, source_products, source_sha, states = snapshot(
            catalog, data_dir
        )
        if source_sha != manifest.source_reference_snapshot_sha256:
            raise ValueError("Source catalog/reviews/files changed since gallery export")
        appearance_map = {a.appearance_id: a for a in source_appearances}
        asset_map = {a.asset_id: a for a in source_assets}
        for product in manifest.products:
            original = source_products.get(product.product_id)
            if original is None or original.model_dump() != product.model_dump():
                raise ValueError("Bundle product metadata differs from its current source")
        for appearance in manifest.appearances:
            original = appearance_map.get(appearance.appearance_id)
            if (
                original is None
                or states[appearance.appearance_id]["fingerprint"]
                != (manifest.source_collection_fingerprints[appearance.appearance_id])
            ):
                raise ValueError("Bundle collection binding differs from its current source")
            original = original.model_copy(update={"submitted_evidence": []})
            if without_local_paths(original.model_dump()) != without_local_paths(
                appearance.model_dump()
            ):
                raise ValueError("Bundle appearance metadata differs from its current source")
        for asset in manifest.reference_assets:
            original = asset_map.get(asset.asset_id)
            if original is None or without_local_paths(
                original.model_dump()
            ) != without_local_paths(asset.model_dump()):
                raise ValueError("Bundle reference metadata differs from its current source")
        current = True
    return {
        "gallery_id": manifest.gallery_id,
        "scope": manifest.scope,
        "appearance_count": len(manifest.appearances),
        "reference_image_count": len(manifest.reference_assets),
        "product_count": len(manifest.products),
        "file_count": len(manifest.files),
        "encoded_bytes": total,
        "current_source_verified": current,
        "valid_until": manifest.valid_until.isoformat(),
        "photo_identification_available": False,
    }


def export_gallery(catalog, data_dir, output, *, scope="reviewed-collection"):
    output = output.resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new gallery directory")
    appearances, assets, products, source_sha, states = snapshot(catalog, data_dir)
    selected = []
    for appearance in appearances:
        state = states[appearance.appearance_id]
        allowed = {a["asset_id"] for a in state["assets"] if a["eligible"]}
        references = [
            a
            for a in assets
            if a.asset_id in allowed
            and a.partition == "reference"
            and a.intended_use == "research_reference"
        ]
        sides = {a.side for a in references}
        if (
            appearance.appearance_id.startswith("local:")
            and state["reference_ready"]
            and ("both" in sides or {"front", "back"} <= sides)
        ):
            selected.append((appearance, references))
    if not selected:
        raise ValueError("No eligible current references; cannot export an empty gallery")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rxsentinel-gallery-", dir=output.parent) as temp:
        stage = Path(temp) / "gallery"
        stage.mkdir()
        inventory, total, exported, images, used = {}, 0, [], [], set()

        def copy_record(record, image=False):
            nonlocal total
            if not file_valid(data_dir, record, "assets" if image else "evidence", image=image):
                raise ValueError(
                    "Source reference/evidence changed or is outside its permitted folder"
                )
            content = bounded_read(data_dir / record.local_path)
            checked_hash(content, record.sha256, "Source file")
            suffix = Path(record.local_path).suffix.lower()
            if image:
                _, kind, _, _ = decode_photo(content)
                suffix = ".jpg" if kind == "JPEG" else ".png"
            elif suffix not in {".jpg", ".jpeg", ".png", ".pdf", ".txt", ".json"}:
                raise ValueError("Unsupported evidence file extension")
            relative = f"{'assets' if image else 'evidence'}/gallery/{record.sha256}{suffix}"
            if relative not in inventory:
                total += len(content)
                if total > MAX_GALLERY_BYTES:
                    raise ValueError("Gallery exceeds byte budget")
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                inventory[relative] = record.sha256
            return record.model_copy(update={"local_path": relative})

        for appearance, references in sorted(selected, key=lambda pair: pair[0].appearance_id):
            exported.append(
                appearance.model_copy(
                    update={
                        "identity_evidence": [copy_record(e) for e in appearance.identity_evidence],
                        "submitted_evidence": [],
                    }
                )
            )
            used.update(appearance.product_ids)
            for asset in sorted(references, key=lambda a: a.asset_id):
                images.append(
                    copy_record(asset, image=True).model_copy(
                        update={
                            "reuse_evidence": [copy_record(e) for e in asset.reuse_evidence],
                        }
                    )
                )
        manifest = ReferenceGallery(
            gallery_id="0" * 64,
            scope=scope,
            created_at=datetime.now(UTC),
            valid_until=min(a.valid_until for a in exported),
            source_reference_snapshot_sha256=source_sha,
            source_collection_fingerprints={
                a.appearance_id: states[a.appearance_id]["fingerprint"] for a in exported
            },
            products=[products[key] for key in sorted(used)],
            appearances=exported,
            reference_assets=images,
            files=dict(sorted(inventory.items())),
        )
        manifest.gallery_id = gallery_id(manifest)
        (stage / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        result = validate_gallery(stage, catalog=catalog, data_dir=data_dir)
        figures = "".join(
            f"<figure><figcaption>{html.escape(a.appearance_id)} — {a.side}</figcaption>"
            f'<img src="{a.local_path}" alt="Reviewed reference"></figure>'
            for a in images
        )
        page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RxSentinel reference gallery</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}}
figure{{display:inline-block;margin:1rem;max-width:450px}}img{{max-width:100%;max-height:400px}}</style>
<h1>Reference gallery — {scope}</h1>
<p>{len(exported)} appearances; {len(images)} reference images.</p>
<p>No trained model or photo identification is included. Recheck source reviews before use.</p>
<p><a href="manifest.json">Catalog, evidence, version and file checksums</a></p>{figures}
</html>"""
        (stage / "gallery.html").write_text(page, encoding="utf-8")
        stage.rename(output)
    return result
