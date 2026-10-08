from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from rxsentinel.schemas import StrictModel


class EvidenceInput(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=100)
    purpose: Literal["packaging", "product_record", "ownership", "license"]
    description: str = Field(min_length=1, max_length=2000)
    file: str = Field(min_length=1)


class PhotoInput(StrictModel):
    file: str = Field(min_length=1)
    side: Literal["front", "back", "both"]
    partition: Literal["reference", "train", "validation", "test"]
    capture_session: str = Field(min_length=1, max_length=100)
    photo_origin: Literal["own_photo", "external"]
    photographer: str = Field(min_length=1, max_length=200)
    captured_at: datetime

    @model_validator(mode="after")
    def validate_time(self):
        if self.captured_at.tzinfo is None or self.captured_at > datetime.now(UTC):
            raise ValueError("Capture time must include a timezone and cannot be in the future")
        return self


class IntakeManifest(StrictModel):
    appearance_id: str = Field(pattern=r"^local:[a-zA-Z0-9_-]{1,80}$")
    product_id: str = Field(min_length=1, max_length=160)
    collection_version: str = Field(min_length=1, max_length=100)
    imprint_front: str | None = None
    imprint_back: str | None = None
    blank_imprint_verified: bool = False
    color: str | None = None
    shape: str | None = None
    score_marks: str | None = None
    manufacturer: str | None = None
    photos: list[PhotoInput] = Field(min_length=1, max_length=50)
    evidence: list[EvidenceInput] = Field(default_factory=list, max_length=20)


class PhotoBatch(StrictModel):
    appearance_id: str = Field(pattern=r"^local:[a-zA-Z0-9_-]{1,80}$")
    expected_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    photos: list[PhotoInput] = Field(min_length=1, max_length=50)
    evidence: list[EvidenceInput] = Field(default_factory=list, max_length=20)


class AppearanceChanges(StrictModel):
    imprint_front: str | None = None
    imprint_back: str | None = None
    blank_imprint_verified: bool | None = Field(default=None, strict=True)
    color: str | None = None
    shape: str | None = None
    score_marks: str | None = None
    manufacturer: str | None = None


class Amendment(StrictModel):
    appearance_id: str = Field(pattern=r"^local:[a-zA-Z0-9_-]{1,80}$")
    expected_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    reason: str = Field(min_length=1, max_length=2000)
    changes: AppearanceChanges = Field(default_factory=AppearanceChanges)
    evidence: list[EvidenceInput] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def require_changes(self):
        if not self.changes.model_fields_set and not self.evidence:
            raise ValueError("Amendment requires characteristics or additional evidence")
        if (
            "blank_imprint_verified" in self.changes.model_fields_set
            and self.changes.blank_imprint_verified is None
        ):
            raise ValueError("Verified blank must be true or false")
        return self


class ReviewInput(StrictModel):
    appearance_id: str = Field(pattern=r"^local:[a-zA-Z0-9_-]{1,80}$")
    expected_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer: str = Field(min_length=1, max_length=200)
    identity_decision: Literal["approve", "reject", "pending"] = "pending"
    reuse_decision: Literal["approve", "reject", "pending"] = "pending"
    basis: str = Field(min_length=1, max_length=4000)
    identity_evidence_ids: list[str] = Field(default_factory=list)
    reuse_evidence_ids: list[str] = Field(default_factory=list)
    reuse_asset_ids: list[str] = Field(default_factory=list)
    valid_until: datetime | None = None

    @model_validator(mode="after")
    def validate_decisions(self):
        if self.identity_decision == self.reuse_decision == "pending":
            raise ValueError("A review must record at least one decision")
        if self.identity_decision == "approve" and not (
            self.identity_evidence_ids and self.valid_until and self.valid_until.tzinfo
        ):
            raise ValueError("Identity approval requires evidence IDs and a timezone-aware expiry")
        if self.reuse_decision != "pending" and not self.reuse_asset_ids:
            raise ValueError("Reuse decisions require explicit asset IDs")
        if self.reuse_decision == "approve" and not self.reuse_evidence_ids:
            raise ValueError("Reuse approval requires evidence IDs")
        for values in (self.identity_evidence_ids, self.reuse_evidence_ids, self.reuse_asset_ids):
            if len(values) != len(set(values)):
                raise ValueError("Review IDs must be unique")
        return self
