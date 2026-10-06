from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceSource(StrictModel):
    name: str = Field(min_length=1)
    url: HttpUrl
    retrieved_at: datetime
    section: str | None = None


class Ingredient(StrictModel):
    source_name: str = Field(min_length=1)
    strength: str | None = None
    rxcui: str | None = Field(default=None, pattern=r"^\d+$")
    normalized_name: str | None = None
    normalization_source: EvidenceSource | None = None

    @model_validator(mode="after")
    def require_mapping_evidence(self):
        if self.rxcui and (not self.normalized_name or not self.normalization_source):
            raise ValueError("Normalized ingredients require a name and mapping evidence")
        return self


class Product(StrictModel):
    product_id: str = Field(min_length=1)
    product_ndc: str = Field(min_length=1)
    brand_name: str | None = None
    generic_name: str = Field(min_length=1)
    dosage_form: str
    manufacturers: list[str] = Field(default_factory=list)
    ingredients: list[Ingredient] = Field(min_length=1)
    source: EvidenceSource
    snapshot_id: str = Field(min_length=1)
    # NDC metadata does not supply these. Unknown values remain explicitly absent.
    imprint: str | None = None
    color: str | None = None
    shape: str | None = None
    reference_images: list[str] = Field(default_factory=list)
    image_reuse_verified: bool = False


class MedicationEntry(StrictModel):
    entry_id: str = Field(min_length=1, max_length=80)
    product_id: str = Field(min_length=1, max_length=160)
    identity_confirmed: bool = False


class AuditRequest(StrictModel):
    medications: list[MedicationEntry] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def unique_entries(self):
        ids = [entry.entry_id for entry in self.medications]
        if len(ids) != len(set(ids)):
            raise ValueError("entry_id must be unique for every medication entry")
        return self


class InteractionRule(StrictModel):
    rule_id: str
    ingredient_a: str = Field(pattern=r"^\d+$")
    ingredient_b: str = Field(pattern=r"^\d+$")
    title: str
    description: str
    recommendation: str
    source: EvidenceSource
    validation_status: Literal["prototype_source_review", "clinician_reviewed"]

    @model_validator(mode="after")
    def distinct_ingredients(self):
        if self.ingredient_a == self.ingredient_b:
            raise ValueError("An interaction rule must connect distinct ingredients")
        return self


class RuleSet(StrictModel):
    version: str
    coverage: Literal["limited_curated_rules"]
    rules: list[InteractionRule]

    @model_validator(mode="after")
    def unique_rules(self):
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("rule_id must be unique")
        return self


class ExcludedEntry(StrictModel):
    entry_id: str
    reason: Literal["identity_unconfirmed", "unknown_product", "unresolved_ingredients"]


class Finding(StrictModel):
    kind: Literal["possible_duplicate_ingredient", "documented_interaction"]
    title: str
    description: str
    recommendation: str
    entry_ids: list[str]
    ingredient_rxcuis: list[str]
    sources: list[EvidenceSource]
    rule_id: str | None = None
    validation_status: str | None = None


class AuditReport(StrictModel):
    status: Literal["findings_require_review", "no_matching_findings", "insufficient_data"]
    rule_set_version: str
    catalog_snapshots: list[str]
    included_entry_ids: list[str]
    excluded_entries: list[ExcludedEntry]
    findings: list[Finding]
    evaluated_distinct_ingredient_pairs: int
    pairs_without_documented_rules: list[tuple[str, str]]
    interaction_coverage: Literal["limited_curated_rules"]
    limitations: list[str]
