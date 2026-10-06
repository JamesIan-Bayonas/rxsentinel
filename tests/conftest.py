from datetime import UTC, datetime
from pathlib import Path

import pytest

from rxsentinel.catalog import Catalog
from rxsentinel.schemas import EvidenceSource, Ingredient, Product, RuleSet


@pytest.fixture
def rules():
    path = Path(__file__).resolve().parents[1] / "data" / "rules" / "interactions.json"
    return RuleSet.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.fixture
def product_factory():
    def make(product_id, rxcuis, *, unresolved=False):
        source = EvidenceSource(
            name="Synthetic test fixture; not a real medication product",
            url="https://example.org/test-fixture",
            retrieved_at=datetime(2026, 10, 6, tzinfo=UTC),
        )
        ingredients = [
            Ingredient(
                source_name=f"test-{rxcui}",
                rxcui=rxcui,
                normalized_name=f"test-{rxcui}",
                normalization_source=source,
            )
            for rxcui in rxcuis
        ]
        if unresolved:
            ingredients.append(Ingredient(source_name="unresolved test ingredient"))
        return Product(
            product_id=product_id,
            product_ndc="TEST-NOT-AN-NDC",
            generic_name=product_id,
            dosage_form="TEST TABLET",
            ingredients=ingredients,
            source=source,
            snapshot_id="test-only",
        )

    return make


@pytest.fixture
def catalog(tmp_path, product_factory):
    catalog = Catalog(tmp_path / "catalog.sqlite")
    catalog.import_products(
        [
            product_factory("aspirin", ["1191"]),
            product_factory("warfarin", ["11289"]),
            product_factory("metformin-a", ["6809"]),
            product_factory("metformin-b", ["6809"]),
            product_factory("combination", ["6809", "1191"]),
            product_factory("unresolved", ["1191"], unresolved=True),
        ]
    )
    return catalog
