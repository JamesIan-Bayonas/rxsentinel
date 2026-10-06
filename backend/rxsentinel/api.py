import os
from pathlib import Path

from fastapi import FastAPI, Query

from rxsentinel.catalog import Catalog
from rxsentinel.schemas import AuditReport, AuditRequest, Product, RuleSet
from rxsentinel.safety import SafetyEngine


def create_app(catalog_path: Path | None = None, rules_path: Path | None = None) -> FastAPI:
    data_dir = Path(os.environ.get("RXSENTINEL_DATA_DIR", "data"))
    catalog = Catalog(catalog_path or data_dir / "catalog.sqlite")
    rule_file = rules_path or data_dir / "rules" / "interactions.json"
    rules = RuleSet.model_validate_json(rule_file.read_text(encoding="utf-8"))
    engine = SafetyEngine(catalog, rules)
    app = FastAPI(
        title="RxSentinel",
        version="0.1.0",
        description="Medication reconciliation research prototype. "
        "Manual reviewed identities only; photo identification is not implemented.",
    )

    @app.get("/health")
    def health():
        products = catalog.all()
        return {
            "status": "ok",
            "catalog_product_count": len(products),
            "catalog_ready": bool(products),
            "rule_set_version": rules.version,
            "interaction_rule_count": len(rules.rules),
            "photo_identification_available": False,
        }

    @app.get("/api/v1/catalog/products", response_model=list[Product])
    def products(
        q: str = Query(default="", max_length=120),
        limit: int = Query(default=25, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ):
        query = q.casefold()
        matches = [
            product
            for product in catalog.all()
            if query in f"{product.brand_name or ''} {product.generic_name}".casefold()
        ]
        return matches[offset : offset + limit]

    @app.post("/api/v1/medication-audits", response_model=AuditReport)
    def audit(request: AuditRequest):
        return engine.audit(request)

    return app
