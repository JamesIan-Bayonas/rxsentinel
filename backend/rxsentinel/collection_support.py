import hashlib
import json
from datetime import UTC, datetime

from rxsentinel.pipelines.reference_files import MAX_IMAGE_BYTES, inspect_image


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def product_fingerprint(product):
    return digest(product.model_dump(mode="json"))


def collection_fingerprint(appearance, assets, products):
    return digest(
        {
            "appearance": appearance.model_dump(mode="json"),
            "assets": [a.model_dump(mode="json") for a in sorted(assets, key=lambda a: a.asset_id)],
            "products": {p.product_id: product_fingerprint(p) for p in products},
        }
    )


def file_valid(data_dir, record, folder, image=False):
    try:
        root = (data_dir / folder).resolve()
        path = (data_dir / record.local_path).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size > MAX_IMAGE_BYTES
        ):
            return False
        content = path.read_bytes()
        if not content or hashlib.sha256(content).hexdigest() != record.sha256:
            return False
        if image:
            inspect_image(content)
        return True
    except (ValueError, OSError):
        return False


def appearance_gaps(data_dir, appearance, products, now=None):
    now = now or datetime.now(UTC)
    gaps = []
    if appearance.appearance_id.startswith("pillbox:"):
        gaps.append("archival_inspection_only")
    if appearance.identity_link_status != "verified" or appearance.stale:
        gaps.append("identity_review")
    if not (
        appearance.color
        and appearance.shape
        and appearance.manufacturer
        and (
            appearance.imprint_front
            or appearance.imprint_back
            or appearance.imprint_unassigned
            or appearance.blank_imprint_verified
        )
    ):
        gaps.append("appearance_characteristics")
    if appearance.appearance_id.startswith("local:"):
        if (
            not appearance.valid_until
            or appearance.valid_until.tzinfo is None
            or appearance.valid_until <= now
        ):
            gaps.append("identity_review_expired_or_missing")
        if not appearance.identity_evidence or not all(
            file_valid(data_dir, e, "evidence") for e in appearance.identity_evidence
        ):
            gaps.append("identity_evidence_integrity")
        if set(appearance.product_ids) != {
            p.product_id for p in products
        } or appearance.product_fingerprints != {
            p.product_id: product_fingerprint(p) for p in products
        }:
            gaps.append("product_snapshot_changed")
    if not products or any(not i.rxcui for p in products for i in p.ingredients):
        gaps.append("ingredient_mapping")
    return gaps


def asset_gaps(data_dir, asset):
    gaps = []
    if asset.intended_use == "inspection_only" or asset.archive_member:
        gaps.append("inspection_only")
    if asset.reuse_status != "permitted":
        gaps.append("reuse_review")
    if not file_valid(data_dir, asset, "assets", image=True):
        gaps.append("image_file_integrity")
    if asset.reuse_evidence and not all(
        file_valid(data_dir, e, "evidence") for e in asset.reuse_evidence
    ):
        gaps.append("reuse_evidence_integrity")
    if not asset.capture_session:
        gaps.append("capture_session")
    return gaps


def dataset_conflicts(assets):
    sessions, hashes = {}, {}
    conflicts = set()
    for asset in assets:
        if asset.capture_session:
            sessions.setdefault(asset.capture_session, set()).add(asset.partition)
        hashes.setdefault(asset.sha256, set()).add(asset.partition)
    for asset in assets:
        if (asset.capture_session and len(sessions[asset.capture_session]) > 1) or len(
            hashes[asset.sha256]
        ) > 1:
            conflicts.add(asset.asset_id)
    return conflicts


def collection_status(data_dir, appearance, assets, products, all_assets):
    gaps = appearance_gaps(data_dir, appearance, products)
    conflicts = dataset_conflicts(all_assets)
    files = []
    for asset in assets:
        missing = asset_gaps(data_dir, asset)
        if asset.asset_id in conflicts:
            missing.append("partition_leakage")
        files.append(
            {
                "asset_id": asset.asset_id,
                "side": asset.side,
                "partition": asset.partition,
                "capture_session": asset.capture_session,
                "eligible": not gaps and not missing,
                "missing_requirements": missing,
            }
        )
    references = [f for f in files if f["eligible"] and f["partition"] == "reference"]
    sides = {f["side"] for f in references}
    has_sides = "both" in sides or {"front", "back"} <= sides
    return {
        "appearance_id": appearance.appearance_id,
        "fingerprint": collection_fingerprint(appearance, assets, products),
        "identity_link_status": appearance.identity_link_status,
        "valid_until": appearance.valid_until.isoformat() if appearance.valid_until else None,
        "missing_requirements": gaps + ([] if has_sides else ["eligible_front_back_references"]),
        "reference_ready": not gaps and has_sides,
        "evaluation_partitions": sorted(
            {f["partition"] for f in files if f["eligible"] and f["partition"] != "reference"}
        ),
        "assets": files,
    }
