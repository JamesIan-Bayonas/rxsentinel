import pytest
from pydantic import ValidationError
from rxsentinel.safety import SafetyEngine
from rxsentinel.schemas import AuditRequest, MedicationEntry


def request(*products, confirmed=True):
    return AuditRequest(
        medications=[
            MedicationEntry(entry_id=f"entry-{i}", product_id=p, identity_confirmed=confirmed)
            for i, p in enumerate(products)
        ]
    )


def test_brand_generic_duplicates_are_possible_duplicates_not_overdose(catalog, rules):
    report = SafetyEngine(catalog, rules).audit(request("metformin-a", "metformin-b"))
    assert report.status == "findings_require_review"
    assert [f.kind for f in report.findings] == ["possible_duplicate_ingredient"]
    assert report.findings[0].ingredient_rxcuis == ["6809"]
    assert "does not establish an overdose" in report.findings[0].description
    assert len(report.findings[0].sources) == 4


@pytest.mark.parametrize("products", [("aspirin", "warfarin"), ("warfarin", "aspirin")])
def test_interaction_is_symmetric_and_sourced(catalog, rules, products):
    report = SafetyEngine(catalog, rules).audit(request(*products))
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.kind == "documented_interaction"
    assert finding.rule_id == "warfarin-aspirin-bleeding"
    assert finding.sources[0].section == "7.3 Drugs that Increase Bleeding Risk"
    assert finding.validation_status != "clinician_reviewed"


def test_missing_rule_does_not_claim_safety(catalog, rules):
    report = SafetyEngine(catalog, rules).audit(request("metformin-a", "aspirin"))
    assert report.status == "no_matching_findings"
    assert report.pairs_without_documented_rules == [("1191", "6809")]
    assert report.interaction_coverage == "limited_curated_rules"
    assert any("does not establish medication safety" in value for value in report.limitations)


def test_unconfirmed_entries_never_enter_audit(catalog, rules):
    report = SafetyEngine(catalog, rules).audit(request("aspirin", "warfarin", confirmed=False))
    assert report.status == "insufficient_data"
    assert not report.findings
    assert [e.reason for e in report.excluded_entries] == ["identity_unconfirmed"] * 2


def test_unknown_and_partial_mapping_are_excluded(catalog, rules):
    report = SafetyEngine(catalog, rules).audit(request("aspirin", "missing", "unresolved"))
    assert report.included_entry_ids == ["entry-0"]
    assert {e.reason for e in report.excluded_entries} == {
        "unknown_product",
        "unresolved_ingredients",
    }
    assert not report.findings
    assert any("assessment is incomplete" in value for value in report.limitations)


def test_combination_product_compares_each_ingredient(catalog, rules):
    report = SafetyEngine(catalog, rules).audit(request("combination", "metformin-a", "warfarin"))
    assert {f.kind for f in report.findings} == {
        "possible_duplicate_ingredient",
        "documented_interaction",
    }
    assert report.evaluated_distinct_ingredient_pairs == 3


def test_one_combination_product_is_not_its_own_duplicate(catalog, rules):
    report = SafetyEngine(catalog, rules).audit(request("combination"))
    assert not any(f.kind == "possible_duplicate_ingredient" for f in report.findings)


def test_duplicate_entry_ids_rejected():
    with pytest.raises(ValidationError, match="entry_id must be unique"):
        AuditRequest(
            medications=[
                MedicationEntry(entry_id="a", product_id="aspirin"),
                MedicationEntry(entry_id="a", product_id="warfarin"),
            ]
        )
