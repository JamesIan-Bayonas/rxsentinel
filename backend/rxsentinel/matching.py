"""M4 manual-observation baseline, restricted to intact current reviewed references."""

import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field
from sqlalchemy import select

from rxsentinel.collection_support import collection_status, digest
from rxsentinel.database import tables
from rxsentinel.schemas import Appearance, EvidenceSource, ImageAsset, Product, StrictModel

MATCHER_VERSION = "manual-imprint-0.1"


class MatchQuery(StrictModel):
    imprint_front: str | None = Field(default=None, min_length=1, max_length=80)
    imprint_back: str | None = Field(default=None, min_length=1, max_length=80)
    color: str | None = Field(default=None, min_length=1, max_length=80)
    shape: str | None = Field(default=None, min_length=1, max_length=80)
    limit: int = Field(default=3, ge=1, le=10, strict=True)


class ReferenceImage(StrictModel):
    asset_id: str
    side: str
    sha256: str
    source: EvidenceSource
    reuse_reviewer: str
    reuse_reviewed_at: datetime


class MatchCandidate(StrictModel):
    appearance_id: str
    products: list[Product]
    imprint_front: str | None
    imprint_back: str | None
    imprint_unassigned: str | None
    color: str | None
    shape: str | None
    manufacturer: str | None
    evidence_points: int
    point_components: dict[str, int]
    comparisons: dict[str, Literal["match", "mismatch", "not_observed", "reference_missing"]]
    matched_reference_sides: list[str]
    uncertainties: list[str]
    source: EvidenceSource
    source_version: str
    identity_reviewer: str
    identity_reviewed_at: datetime
    valid_until: datetime
    collection_fingerprint: str
    reference_images: list[ReferenceImage]
    requires_human_review: Literal[True] = True


class MatchReport(StrictModel):
    matcher_version: str = MATCHER_VERSION
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: Literal["candidates", "unknown"]
    reason: Literal[
        "no_eligible_references",
        "imprint_required",
        "unsupported_imprint",
        "no_exact_imprint_match",
        "exact_imprint_candidates",
        "ambiguous_matches",
        "references_changed_during_search",
    ]
    query: MatchQuery
    query_sha256: str
    reference_snapshot_sha256: str
    eligible_appearance_count: int
    excluded_appearance_count: int
    exclusion_reasons: dict[str, int]
    total_candidate_count: int = 0
    truncated: bool = False
    ambiguous: bool = False
    candidates: list[MatchCandidate] = Field(default_factory=list)
    photo_identification_available: Literal[False] = False
    query_identity_confirmed: Literal[False] = False
    warnings: list[str] = Field(
        default_factory=lambda: [
            "Manual observations only; no OCR, photo inference or calibrated probability",
            "Evidence points rank exact imprint matches; they do not establish identity",
            "Limited catalog coverage: the actual medication may be outside this catalog",
            "Check both sides against packaging before using a medication identity in an audit",
        ]
    )


def normalize(value: str) -> str:
    # Preserve punctuation, internal separators and ambiguous characters such as O/0 or I/1.
    return " ".join(value.split()).casefold()


def snapshot(catalog, data_dir):
    if not catalog.store:
        return [], [], {}, digest([]), {}
    with catalog.store.collection_transaction() as connection:
        products = [
            Product.model_validate_json(p)
            for p in connection.scalars(select(tables.products.c.payload))
        ]
        appearances = [
            Appearance.model_validate_json(p)
            for p in connection.scalars(select(tables.appearances.c.payload))
        ]
        assets = [
            ImageAsset.model_validate_json(p)
            for p in connection.scalars(select(tables.image_assets.c.payload))
        ]
        states = {
            a.appearance_id: collection_status(
                data_dir,
                a,
                [asset for asset in assets if asset.appearance_id == a.appearance_id],
                [p for p in products if p.product_id in a.product_ids],
                assets,
            )
            for a in sorted(appearances, key=lambda a: a.appearance_id)
        }
    return appearances, assets, {p.product_id: p for p in products}, digest(states), states


def imprint_matches(query, appearance):
    observed = [normalize(s) for s in (query.imprint_front, query.imprint_back) if s]
    reference = {
        side: normalize(text)
        for side, text in (
            ("front", appearance.imprint_front),
            ("back", appearance.imprint_back),
            ("unassigned", appearance.imprint_unassigned),
        )
        if text
    }
    if len(observed) == 1:
        return [side for side, text in reference.items() if text == observed[0]]
    # Front/back naming is arbitrary. Compare both assignments and preserve duplicates.
    if (
        "front" in reference
        and "back" in reference
        and Counter(observed) == Counter([reference["front"], reference["back"]])
    ):
        return ["front", "back"]
    return []


def compare(observed, reference):
    if observed is None:
        return "not_observed"
    if reference is None:
        return "reference_missing"
    return "match" if normalize(observed) == normalize(reference) else "mismatch"


def match_observations(catalog, data_dir: Path, query: MatchQuery) -> MatchReport:
    # Legacy metadata-only SQLite catalogs have no eligible reference collections.
    if catalog.store:
        appearances, assets, products, snapshot_sha, states = snapshot(catalog, data_dir)
    else:
        appearances, assets, products, snapshot_sha, states = [], [], {}, digest([]), {}
    eligible, exclusions = [], Counter()
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
        gaps = list(state["missing_requirements"])
        if not appearance.appearance_id.startswith("local:"):
            gaps.append("current_local_collection_required")
        if not ("both" in sides or {"front", "back"} <= sides):
            gaps.append("eligible_research_reference_sides")
        if gaps:
            exclusions.update(set(gaps))
        else:
            eligible.append((appearance, references))
    report = MatchReport(
        status="unknown",
        reason="no_eligible_references",
        query=query,
        query_sha256=digest(query.model_dump(mode="json")),
        reference_snapshot_sha256=snapshot_sha,
        eligible_appearance_count=len(eligible),
        excluded_appearance_count=len(appearances) - len(eligible),
        exclusion_reasons=dict(sorted(exclusions.items())),
    )
    if not eligible:
        return report
    imprints = [s for s in (query.imprint_front, query.imprint_back) if s]
    if not imprints:
        report.reason = "imprint_required"
        return report
    if any(
        not re.fullmatch(r"[A-Za-z0-9 .\-/]+", " ".join(text.split()))
        or not re.search(r"[A-Za-z0-9]", text)
        for text in imprints
    ):
        report.reason = "unsupported_imprint"
        return report
    ranked = []
    for appearance, references in eligible:
        sides = imprint_matches(query, appearance)
        if not sides:
            continue
        comparisons = {
            name: compare(getattr(query, name), getattr(appearance, name))
            for name in ("color", "shape")
        }
        components = {
            "imprint": 10 * len(imprints),
            **{name: int(comparison == "match") for name, comparison in comparisons.items()},
        }
        uncertainties = [
            f"{name}_{value}" for name, value in comparisons.items() if value != "match"
        ]
        if len(imprints) == 1:
            uncertainties.append("only_one_imprint_observed")
        if "unassigned" in sides:
            uncertainties.append("reference_imprint_side_unassigned")
        if len(appearance.product_ids) > 1:
            uncertainties.append("appearance_linked_to_multiple_products")
        ranked.append(
            MatchCandidate(
                appearance_id=appearance.appearance_id,
                products=[products[key] for key in sorted(appearance.product_ids)],
                imprint_front=appearance.imprint_front,
                imprint_back=appearance.imprint_back,
                imprint_unassigned=appearance.imprint_unassigned,
                color=appearance.color,
                shape=appearance.shape,
                manufacturer=appearance.manufacturer,
                evidence_points=sum(components.values()),
                point_components=components,
                comparisons=comparisons,
                matched_reference_sides=sides,
                uncertainties=uncertainties,
                source=appearance.source,
                source_version=appearance.source_version,
                identity_reviewer=appearance.reviewer,
                identity_reviewed_at=appearance.reviewed_at,
                valid_until=appearance.valid_until,
                collection_fingerprint=states[appearance.appearance_id]["fingerprint"],
                reference_images=[
                    ReferenceImage(
                        asset_id=a.asset_id,
                        side=a.side,
                        sha256=a.sha256,
                        source=a.source,
                        reuse_reviewer=a.reviewer,
                        reuse_reviewed_at=a.reviewed_at,
                    )
                    for a in sorted(references, key=lambda a: a.asset_id)
                ],
            )
        )
    # Recheck file/review eligibility before returning a result from a mutable local collection.
    if snapshot(catalog, data_dir)[3] != snapshot_sha:
        report.reason = "references_changed_during_search"
        return report
    if not ranked:
        report.reason = "no_exact_imprint_match"
        return report
    ranked.sort(key=lambda candidate: (-candidate.evidence_points, candidate.appearance_id))
    report.status = "candidates"
    report.ambiguous = len(ranked) > 1 or any(len(c.products) > 1 for c in ranked)
    report.reason = "ambiguous_matches" if report.ambiguous else "exact_imprint_candidates"
    report.total_candidate_count = len(ranked)
    report.truncated = len(ranked) > query.limit
    report.candidates = ranked[: query.limit]
    return report
