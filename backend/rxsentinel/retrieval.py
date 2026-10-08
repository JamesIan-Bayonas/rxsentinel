"""Versioned exhaustive cosine retrieval; candidates never establish medication identity."""

import hashlib
import io
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from rxsentinel.collection_support import digest
from rxsentinel.gallery import ReferenceGallery, validate_gallery
from rxsentinel.pipelines.photo import bounded_read
from rxsentinel.schemas import ImageAsset, Product, StrictModel
from rxsentinel.vision.embedding import EncoderSpec
from rxsentinel.vision.evaluation import checked_hash, local_file
from rxsentinel.vision.single import ProcessingReport, process_photo

MAX_VECTORS_BYTES = 8 * 1024**2


class VectorRow(StrictModel):
    asset_id: str
    appearance_id: str
    input_sha256: str
    processing: ProcessingReport


class VectorManifest(StrictModel):
    version: Literal["visual-gallery-1"] = "visual-gallery-1"
    index_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    gallery_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at: datetime
    encoder: EncoderSpec
    retrieval_code_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    metric: Literal["exhaustive-cosine/best-reference-per-appearance"] = (
        "exhaustive-cosine/best-reference-per-appearance"
    )
    vectors_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rows: list[VectorRow] = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def unique_rows(self):
        if len({r.asset_id for r in self.rows}) != len(self.rows):
            raise ValueError("Vector asset IDs must be unique")
        if not self.created_at.tzinfo:
            raise ValueError("Index creation time must be timezone-aware")
        return self


class VisualCandidate(StrictModel):
    appearance_id: str
    cosine_similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)
    products: list[Product]
    best_reference: ImageAsset
    reference_images: list[ImageAsset]
    imprint_front: str | None
    imprint_back: str | None
    color: str | None
    shape: str | None
    identity_reviewer: str | None
    identity_reviewed_at: datetime | None
    identity_valid_until: datetime | None


class RetrievalReport(StrictModel):
    version: Literal["visual-query-1"] = "visual-query-1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    scope: Literal["synthetic", "reviewed-collection"]
    index_id: str
    gallery_id: str
    encoder: EncoderSpec
    input_sha256: str
    processing: ProcessingReport
    status: Literal["unknown", "candidates"]
    reason: str
    current_source_verified: Literal[True] = True
    query_identity_confirmed: Literal[False] = False
    photo_identification_available: Literal[False] = False
    requires_human_review: Literal[True] = True
    thresholds_calibrated: Literal[False] = False
    minimum_similarity: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    eligible_appearance_count: int
    total_candidate_count: int
    tied_first_candidate_count: int = 0
    candidates: list[VisualCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=lambda: [
            "Similarity is not a probability or identity confirmation",
            "ImageNet features and rejection thresholds have not been validated on real pills",
            "Out-of-gallery objects can receive high similarity scores",
            "Compare imprints, packaging and the medication list before confirming identity",
        ]
    )


def retrieval_code_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def index_id(manifest):
    payload = manifest.model_dump(mode="json")
    payload.pop("index_id")
    return digest(payload)


def unit_vector(vector, dimension):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError("Embedding has an invalid dimension or non-finite values")
    length = float(np.linalg.norm(vector.astype(np.float64)))
    if length < 1e-12:
        raise ValueError("Embedding has zero norm")
    return vector / length


def checked_gallery(directory, catalog, data_dir):
    if catalog is None or data_dir is None:
        raise ValueError("Visual retrieval requires current source catalog and review validation")
    validate_gallery(directory, catalog=catalog, data_dir=data_dir)
    return ReferenceGallery.model_validate_json(bounded_read(directory / "manifest.json"))


def build_vectors(gallery_dir, output, encoder, *, catalog, data_dir):
    output = output.resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new vector directory")
    gallery = checked_gallery(gallery_dir, catalog, data_dir)
    rows, vectors = [], []
    spec = encoder.spec.model_copy(deep=True)
    if spec.processing_config.model_size != 224:
        raise ValueError("ResNet18 preprocessing requires a 224 pixel letterbox")
    for asset in sorted(gallery.reference_assets, key=lambda a: a.asset_id):
        content = local_file(gallery_dir, asset.local_path)
        checked_hash(content, asset.sha256, "Reference image")
        report, artifacts = process_photo(content, spec.processing_config)
        if report.status != "processed":
            raise ValueError(f"Reference {asset.asset_id} rejected: {report.rejection_reasons}")
        vectors.append(unit_vector(encoder.encode(artifacts["model_input"]), spec.dimension))
        rows.append(
            VectorRow(
                asset_id=asset.asset_id,
                appearance_id=asset.appearance_id,
                input_sha256=asset.sha256,
                processing=report,
            )
        )
    if encoder.spec != spec:
        raise ValueError("Encoder specification changed during embedding extraction")
    buffer = io.BytesIO()
    np.save(buffer, np.stack(vectors).astype(np.float32), allow_pickle=False)
    content = buffer.getvalue()
    if len(content) > MAX_VECTORS_BYTES:
        raise ValueError("Vector file exceeds byte budget")
    manifest = VectorManifest(
        index_id="0" * 64,
        gallery_id=gallery.gallery_id,
        created_at=datetime.now(UTC),
        encoder=spec,
        retrieval_code_sha256=retrieval_code_sha256(),
        vectors_sha256=hashlib.sha256(content).hexdigest(),
        rows=rows,
    )
    manifest.index_id = index_id(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rxsentinel-vectors-", dir=output.parent) as temp:
        stage = Path(temp) / "vectors"
        stage.mkdir()
        (stage / "vectors.npy").write_bytes(content)
        (stage / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        validate_vectors(stage, gallery_dir, encoder, catalog=catalog, data_dir=data_dir)
        stage.rename(output)
    return manifest


def validate_vectors(directory, gallery_dir, encoder, *, catalog, data_dir):
    gallery = checked_gallery(gallery_dir, catalog, data_dir)
    manifest = VectorManifest.model_validate_json(bounded_read(directory / "manifest.json"))
    if index_id(manifest) != manifest.index_id:
        raise ValueError("Vector manifest fingerprint mismatch")
    if manifest.gallery_id != gallery.gallery_id:
        raise ValueError("Vectors belong to a different reference gallery")
    if (
        manifest.encoder != encoder.spec
        or manifest.retrieval_code_sha256 != retrieval_code_sha256()
    ):
        raise ValueError(
            "Encoder, transform, runtime or retrieval version mismatch; rebuild vectors"
        )
    expected = sorted(gallery.reference_assets, key=lambda a: a.asset_id)
    if [(r.asset_id, r.appearance_id, r.input_sha256) for r in manifest.rows] != [
        (a.asset_id, a.appearance_id, a.sha256) for a in expected
    ]:
        raise ValueError("Vector rows do not exactly cover the gallery references")
    for row in manifest.rows:
        if (
            row.processing.status != "processed"
            or row.processing.input_sha256 != row.input_sha256
            or row.processing.config != manifest.encoder.processing_config
            or row.processing.crop_box is None
        ):
            raise ValueError("Vector processing evidence differs from its reference/configuration")
    with (directory / "vectors.npy").open("rb") as stream:
        content = stream.read(MAX_VECTORS_BYTES + 1)
    if len(content) > MAX_VECTORS_BYTES:
        raise ValueError("Vector file exceeds byte budget")
    checked_hash(content, manifest.vectors_sha256, "Vector file")
    try:
        header = io.BytesIO(content)
        if np.lib.format.read_magic(header) != (1, 0):
            raise ValueError("Expected a version-1 NPY vector array")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(header)
        if (
            shape != (len(manifest.rows), manifest.encoder.dimension)
            or fortran
            or dtype != np.dtype(np.float32)
            or header.tell() + len(manifest.rows) * manifest.encoder.dimension * 4 != len(content)
        ):
            raise ValueError("Vector header, dimension, dtype or encoded length is invalid")
        vectors = np.load(io.BytesIO(content), allow_pickle=False)
    except (ValueError, OSError, EOFError) as error:
        raise ValueError("Invalid vector array") from error
    if (
        not isinstance(vectors, np.ndarray)
        or vectors.dtype != np.float32
        or vectors.shape != (len(manifest.rows), manifest.encoder.dimension)
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors.astype(np.float64), axis=1), 1, atol=1e-5)
    ):
        raise ValueError(
            "Vectors must be finite unit-length float32 rows of the recorded dimension"
        )
    return manifest, gallery, vectors


def retrieve_photo(
    content, directory, gallery_dir, encoder, *, catalog, data_dir, limit=3, minimum_similarity=None
):
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("Candidate limit must be an integer between 1 and 20")
    if minimum_similarity is not None and (
        not np.isfinite(minimum_similarity) or not -1 <= minimum_similarity <= 1
    ):
        raise ValueError("Minimum similarity must be finite and between -1 and 1")
    manifest, gallery, vectors = validate_vectors(
        directory, gallery_dir, encoder, catalog=catalog, data_dir=data_dir
    )
    processing, artifacts = process_photo(content, manifest.encoder.processing_config)
    result = RetrievalReport(
        scope=gallery.scope,
        index_id=manifest.index_id,
        gallery_id=gallery.gallery_id,
        encoder=manifest.encoder,
        input_sha256=processing.input_sha256,
        processing=processing,
        status="unknown",
        reason="photo_processing_rejected",
        minimum_similarity=minimum_similarity,
        eligible_appearance_count=len(gallery.appearances),
        total_candidate_count=0,
    )
    if result.input_sha256 in {a.sha256 for a in gallery.reference_assets}:
        result.warnings.append(
            "Query duplicates a reference image; this is not an independent evaluation case"
        )
    if processing.status == "processed":
        query = unit_vector(encoder.encode(artifacts["model_input"]), manifest.encoder.dimension)
        similarities = np.clip(vectors @ query, -1, 1)
        assets = {a.asset_id: a for a in gallery.reference_assets}
        products = {p.product_id: p for p in gallery.products}
        ranked = []
        for appearance in gallery.appearances:
            indices = [
                i
                for i, row in enumerate(manifest.rows)
                if row.appearance_id == appearance.appearance_id
            ]
            best = min(indices, key=lambda i: (-float(similarities[i]), manifest.rows[i].asset_id))
            score = float(similarities[best])
            if minimum_similarity is not None and score < minimum_similarity:
                continue
            ranked.append(
                VisualCandidate(
                    appearance_id=appearance.appearance_id,
                    cosine_similarity=score,
                    products=[products[key] for key in appearance.product_ids],
                    best_reference=assets[manifest.rows[best].asset_id],
                    reference_images=[assets[manifest.rows[i].asset_id] for i in indices],
                    imprint_front=appearance.imprint_front,
                    imprint_back=appearance.imprint_back,
                    color=appearance.color,
                    shape=appearance.shape,
                    identity_reviewer=appearance.reviewer,
                    identity_reviewed_at=appearance.reviewed_at,
                    identity_valid_until=appearance.valid_until,
                )
            )
        ranked.sort(key=lambda c: (-c.cosine_similarity, c.appearance_id))
        result.total_candidate_count = len(ranked)
        if ranked:
            result.status, result.reason = "candidates", "uncalibrated_visual_candidates"
            result.tied_first_candidate_count = sum(
                abs(c.cosine_similarity - ranked[0].cosine_similarity) <= 1e-6 for c in ranked
            )
            result.candidates = ranked[:limit]
        else:
            result.reason = "below_experimental_similarity_threshold"
    # Any mid-inference review/file/version change invalidates the whole result.
    final, _, _ = validate_vectors(
        directory, gallery_dir, encoder, catalog=catalog, data_dir=data_dir
    )
    if final.index_id != manifest.index_id:
        raise ValueError("Vector index changed during inference")
    return result
