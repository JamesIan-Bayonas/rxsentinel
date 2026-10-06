from pathlib import Path

from fastapi.testclient import TestClient

from rxsentinel.api import create_app

RULES = Path(__file__).resolve().parents[1] / "data" / "rules" / "interactions.json"


def test_empty_catalog_is_reported_and_does_not_get_created(tmp_path):
    path = tmp_path / "missing.sqlite"
    client = TestClient(create_app(path, RULES))
    assert client.get("/health").json()["catalog_ready"] is False
    assert client.get("/api/v1/catalog/products").json() == []
    assert not path.exists()


def test_api_audit_contract(catalog):
    client = TestClient(create_app(catalog.path, RULES))
    response = client.post("/api/v1/medication-audits", json={"medications": [
        {"entry_id": "one", "product_id": "aspirin", "identity_confirmed": True},
        {"entry_id": "two", "product_id": "warfarin", "identity_confirmed": True},
    ]})
    assert response.status_code == 200
    assert response.json()["findings"][0]["rule_id"] == "warfarin-aspirin-bleeding"
    assert client.get("/health").json()["photo_identification_available"] is False


def test_extra_fields_and_oversized_inputs_rejected(catalog):
    client = TestClient(create_app(catalog.path, RULES))
    assert client.post("/api/v1/medication-audits", json={
        "medications": [{"entry_id": "one", "product_id": "aspirin", "dosage": "invented"}]
    }).status_code == 422
    assert client.post("/api/v1/medication-audits", json={"medications": []}).status_code == 422
    assert client.get("/api/v1/catalog/products?limit=1000").status_code == 422
