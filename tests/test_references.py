import hashlib
import io
import json
import zipfile

import httpx
import pytest
from defusedxml.common import DefusedXmlException
from fastapi.testclient import TestClient
from PIL import Image
from rxsentinel.api import create_app
from rxsentinel.catalog import Catalog
from rxsentinel.database.migrate import upgrade_schema
from rxsentinel.pipelines.readiness import asset_file_valid, generate_readiness
from rxsentinel.pipelines.reference_files import RangeReader, archived_image, save_image
from rxsentinel.pipelines.references import check_references, compare_label, fetch_label
from rxsentinel.schemas import Appearance, ImageAsset, Ingredient, LabelIngredient
from sqlalchemy.exc import IntegrityError

SET_ID = "00000000-0000-0000-0000-000000000001"


def spl(code="12345-678", strength="3", imprint="A;3"):
    # Entirely synthetic SPL-shaped fixture, deliberately different from real products.
    return f'''<document xmlns="urn:hl7-org:v3"><setId root="{SET_ID}"/>
    <versionNumber value="2"/><effectiveTime value="20260101"/>
    <author><assignedEntity><representedOrganization><name>Test Labeler</name>
    </representedOrganization></assignedEntity></author>
    <component><manufacturedProduct><manufacturedProduct><code code="{code}"/>
    <formCode displayName="TABLET"/><ingredient classCode="ACTIB">
    <quantity><numerator value="{strength}" unit="mg"/>
    <denominator value="1" unit="1"/></quantity>
    <ingredientSubstance><name>TEST INGREDIENT</name></ingredientSubstance></ingredient>
    <ingredient classCode="IACT"><ingredientSubstance><name>INACTIVE TEST</name>
    </ingredientSubstance></ingredient></manufacturedProduct>
    <subjectOf><characteristic><code code="SPLIMPRINT"/><value>{imprint}</value>
    </characteristic></subjectOf>
    <subjectOf><characteristic><code code="SPLCOLOR"/><value displayName="WHITE"/>
    </characteristic></subjectOf>
    <subjectOf><characteristic><code code="SPLSHAPE"/><value displayName="ROUND"/>
    </characteristic></subjectOf>
    <subjectOf><characteristic><code code="SPLSCORE"/><value value="1"/>
    </characteristic></subjectOf></manufacturedProduct></component></document>'''.encode()


@pytest.fixture
def lead(product_factory):
    product = product_factory("test-product", ["1191"]).model_copy(
        update={
            "product_ndc": "12345-678",
            "manufacturers": ["Test Labeler"],
            "dosage_form": "TABLET",
            "ingredients": [Ingredient(source_name="TEST INGREDIENT", strength="3 mg/1")],
        }
    )
    appearance = Appearance(
        appearance_id="pillbox:7",
        product_ids=[product.product_id],
        source=product.source,
        source_version="test-only",
        imprint_unassigned="A;3",
        color="white",
        shape="round",
        score_marks="1",
        stale=True,
    )
    return product, appearance


def check(xml, lead):
    product, appearance = lead
    return compare_label(xml, SET_ID, product, appearance, "test-only", product.source)


def test_exact_product_and_strength_consistency_does_not_promote_identity(lead):
    result = check(spl(), lead)
    assert result.outcome == "consistent"
    assert result.ingredients == [LabelIngredient(name="TEST INGREDIENT", strength="3 mg/1")]
    assert lead[1].stale and lead[1].identity_link_status == "unverified"
    assert result.xml_sha256 == hashlib.sha256(spl()).hexdigest()


def test_strength_mismatch_does_not_use_a_different_product_in_same_label(lead):
    wrong = spl(strength="10")
    additional = spl(code="12345-999", strength="3")
    section = additional[additional.index(b"<component>") : additional.index(b"</document>")]
    result = check(wrong.replace(b"</document>", section + b"</document>"), lead)
    assert result.outcome == "mismatch"
    assert result.comparisons["strengths"] == "mismatch"
    assert result.ingredients[0].strength == "10 mg/1"


@pytest.mark.parametrize(
    "xml,outcome",
    [
        (spl(code="12345-999"), "not_found"),
        (spl(imprint=""), "incomplete"),
        (spl(imprint="B;3"), "mismatch"),
    ],
)
def test_missing_or_conflicting_evidence_is_preserved(xml, outcome, lead):
    assert check(xml, lead).outcome == outcome


def test_duplicate_product_and_wrong_set_ids_are_rejected(lead):
    xml = spl()
    section = xml[xml.index(b"<component>") : xml.index(b"</document>")]
    assert check(xml.replace(b"</document>", section + b"</document>"), lead).outcome == "ambiguous"
    with pytest.raises(ValueError, match="set ID"):
        check(xml.replace(SET_ID.encode(), b"wrong"), lead)


def test_external_xml_entities_are_not_processed(lead):
    xml = b'<!DOCTYPE document [<!ENTITY secret SYSTEM "file:///private">]>' + spl()
    with pytest.raises(DefusedXmlException):
        check(xml, lead)


@pytest.mark.anyio
async def test_html_provider_response_cannot_be_treated_as_label():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, text="<html>wrong endpoint</html>", headers={"Content-Type": "text/html"}
            )
        )
    ) as client:
        with pytest.raises(ValueError, match="non-XML"):
            await fetch_label(client, "https://example.org/test")


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (24, 16), "white").save(output, format="PNG")
    return output.getvalue()


def test_image_integrity_and_path_containment(tmp_path, lead):
    relative, sha, width, height = save_image(tmp_path, png_bytes())
    asset = ImageAsset(
        asset_id="test",
        appearance_id="pillbox:7",
        local_path=relative,
        sha256=sha,
        source=lead[0].source,
        side="unknown",
    )
    assert (width, height) == (24, 16)
    assert asset_file_valid(tmp_path, asset)
    (tmp_path / relative).write_bytes(b"corrupt")
    assert not asset_file_valid(tmp_path, asset)
    outside = tmp_path / "outside.png"
    outside.write_bytes(png_bytes())
    assert not asset_file_valid(tmp_path, asset.model_copy(update={"local_path": "outside.png"}))
    with pytest.raises(OSError):
        save_image(tmp_path, b"this is not an image")


def zipped_images(ambiguous=False):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("../untrusted/sample.jpg", png_bytes())
        if ambiguous:
            archive.writestr("second/sample.jpg", png_bytes())
    return output.getvalue()


def test_member_names_are_never_extracted_as_paths_and_ambiguity_abstains(tmp_path):
    with zipfile.ZipFile(io.BytesIO(zipped_images())) as archive:
        member, content = archived_image(archive, "sample")
        assert member == "../untrusted/sample.jpg"
        relative, *_ = save_image(tmp_path, content)
        assert relative.startswith("assets/pillbox-inspection/")
    with zipfile.ZipFile(io.BytesIO(zipped_images(True))) as archive:
        with pytest.raises(ValueError, match="ambiguous"):
            archived_image(archive, "sample")


def test_range_reader_only_fetches_requested_bytes():
    data = zipped_images()

    def handler(request):
        if request.method == "HEAD":
            return httpx.Response(200, headers={"Content-Length": str(len(data)), "ETag": '"test"'})
        assert request.headers["if-match"] == '"test"'
        start, end = map(int, request.headers["range"][6:].split("-"))
        return httpx.Response(
            206,
            stream=httpx.ByteStream(data[start : end + 1]),
            headers={"Content-Range": f"bytes {start}-{end}/{len(data)}"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reader = RangeReader(client, "https://example.org/test.zip")
        with zipfile.ZipFile(reader) as archive:
            assert archived_image(archive, "sample")[1] == png_bytes()
        assert reader.transferred < 20 * 1024**2


@pytest.mark.parametrize(
    "status,content_range,content",
    [
        (200, None, b"full download"),
        (206, "wrong", b"x"),
        (206, "bytes 0-0/100", b"xx"),
        (206, "bytes 0-0/100", b""),
    ],
)
def test_ignored_inconsistent_or_truncated_ranges_are_rejected(status, content_range, content):
    def handler(request):
        if request.method == "HEAD":
            return httpx.Response(200, headers={"Content-Length": "100"})
        headers = {"Content-Range": content_range} if content_range else {}
        return httpx.Response(status, stream=httpx.ByteStream(content), headers=headers)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            RangeReader(client, "https://example.org/test.zip").read(1)


@pytest.fixture
def relational(tmp_path, lead):
    catalog = Catalog(f"sqlite+pysqlite:///{(tmp_path / 'catalog.sqlite').as_posix()}")
    upgrade_schema(catalog.store.engine)
    catalog.import_products([lead[0]])
    catalog.store.put_appearance(lead[1])
    yield catalog
    catalog.store.engine.dispose()


@pytest.mark.anyio
async def test_pipeline_persists_checks_and_exposes_typed_api_without_promotion(
    relational,
    lead,
    tmp_path,
    monkeypatch,
):
    directory = tmp_path / "source-discovery" / "pillbox-discovery-test"
    directory.mkdir(parents=True)
    raw = json.dumps(
        {
            "records": [
                {"id": "7", "product_code": "12345-678", "setid": SET_ID, "has_image": "False"}
            ]
        }
    ).encode()
    (directory / "raw.json").write_bytes(raw)
    (directory / "manifest.json").write_text(
        json.dumps({"snapshot_id": "test-only", "raw_sha256": hashlib.sha256(raw).hexdigest()})
    )
    original = httpx.AsyncClient
    monkeypatch.setattr("rxsentinel.pipelines.references.configured_catalog", lambda _: relational)
    monkeypatch.setattr(
        "rxsentinel.pipelines.references.httpx.AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, content=spl(), headers={"Content-Type": "application/xml"}
                )
            ),
            **kwargs,
        ),
    )
    result = await check_references(tmp_path)
    assert result["outcomes"] == {"consistent": 1}
    assert result["identities_promoted"] == result["reuse_permissions_granted"] == 0
    assert relational.store.appearances() == [lead[1]]
    assert relational.store.assets() == []
    assert (tmp_path / "reference-review.html").exists()
    monkeypatch.setenv("RXSENTINEL_DATABASE_URL", str(relational.store.engine.url))
    from pathlib import Path

    rules_file = Path(__file__).resolve().parents[1] / "data/rules/interactions.json"
    with TestClient(create_app(rules_path=rules_file)) as client:
        response = client.get("/api/v1/catalog/reference-checks")
        assert response.status_code == 200
        assert response.json()[0]["outcome"] == "consistent"
        assert client.get("/api/v1/catalog/reference-checks?limit=101").status_code == 422


def test_reference_import_rolls_back_and_preserves_existing_asset_review(relational, lead):
    result = check(spl(), lead)
    asset = ImageAsset(
        asset_id="test",
        appearance_id="pillbox:7",
        local_path="assets/test.png",
        sha256="a" * 64,
        source=lead[0].source,
        side="unknown",
    )
    reviewed = asset.model_copy(update={"reviewer": "test reviewer"})
    relational.store.put_asset(reviewed)
    invalid = result.model_copy(update={"check_id": "missing", "product_id": "missing"})
    with pytest.raises(IntegrityError):
        relational.store.import_reference_checks(
            [result, invalid], [asset], {"snapshot_id": "test"}
        )
    assert relational.store.reference_checks() == []
    relational.store.import_reference_checks([result], [asset], {"snapshot_id": "test"})
    assert relational.store.assets() == [reviewed]


def test_archived_images_never_pass_readiness_even_if_later_marked_reviewed(
    relational,
    lead,
    tmp_path,
    monkeypatch,
):
    relative, sha, _, _ = save_image(tmp_path, png_bytes())
    appearance = lead[1].model_copy(
        update={
            "stale": False,
            "identity_link_status": "verified",
            "reviewer": "test",
            "reviewed_at": lead[0].source.retrieved_at,
        }
    )
    relational.store.put_appearance(appearance)
    asset = ImageAsset(
        asset_id="test",
        appearance_id="pillbox:7",
        local_path=relative,
        sha256=sha,
        source=lead[0].source,
        side="unknown",
        intended_use="research_reference",
        reuse_status="permitted",
        reuse_evidence_url="https://example.org/test-license",
        reuse_basis="Synthetic test license",
        reviewer="test",
        reviewed_at=lead[0].source.retrieved_at,
    )
    relational.store.put_asset(asset)
    monkeypatch.setattr("rxsentinel.pipelines.readiness.configured_catalog", lambda _: relational)
    result = generate_readiness(tmp_path)
    assert result["valid_image_file_count"] == 1
    assert result["appearance_metadata_ready_count"] == 0
    assert not result["visual_identification_ready"]
