import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "acquire_mediseg", Path(__file__).resolve().parents[1] / "scripts/acquire_mediseg.py"
)
acquisition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquisition)

inspection_spec = importlib.util.spec_from_file_location(
    "inspect_mediseg", Path(__file__).resolve().parents[1] / "scripts/inspect_mediseg.py"
)
inspection = importlib.util.module_from_spec(inspection_spec)
inspection_spec.loader.exec_module(inspection)

registry_spec = importlib.util.spec_from_file_location(
    "check_mediseg_products",
    Path(__file__).resolve().parents[1] / "scripts/check_mediseg_products.py",
)
registry = importlib.util.module_from_spec(registry_spec)
registry_spec.loader.exec_module(registry)


def archive_file(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind == tarfile.REGTYPE:
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            else:
                info.linkname = "../outside"
                archive.addfile(info)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../outside", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("C:/outside", tarfile.REGTYPE),
        ("folder\\outside", tarfile.REGTYPE),
        ("link", tarfile.SYMTYPE),
        ("link", tarfile.LNKTYPE),
    ],
)
def test_unsafe_archive_is_rejected_before_any_file_is_written(tmp_path, name, kind):
    source = tmp_path / "archive.tar.gz"
    archive_file(source, [("safe.txt", b"safe", tarfile.REGTYPE), (name, b"bad", kind)])
    root = tmp_path / "output"
    with pytest.raises(ValueError, match="Unsafe"):
        acquisition.unpack(source, root)
    assert not root.exists()


def test_duplicate_members_and_unpack_budget_are_rejected(tmp_path, monkeypatch):
    source = tmp_path / "archive.tar.gz"
    archive_file(source, [("same.txt", b"one", tarfile.REGTYPE)] * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        acquisition.unpack(source, tmp_path / "output")
    archive_file(source, [("safe.txt", b"large", tarfile.REGTYPE)])
    monkeypatch.setattr(acquisition, "MAX_UNPACKED_BYTES", 2)
    with pytest.raises(ValueError, match="unpacked size"):
        acquisition.unpack(source, tmp_path / "output")


def test_rerun_preserves_bytes_and_refuses_changed_existing_file(tmp_path):
    source = tmp_path / "archive.tar.gz"
    archive_file(source, [("folder/license.txt", b"original", tarfile.REGTYPE)])
    root = tmp_path / "output"
    assert acquisition.unpack(source, root) == 1
    assert acquisition.unpack(source, root) == 1
    target = root / "folder/license.txt"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="refusing overwrite"):
        acquisition.unpack(source, root)
    assert target.read_bytes() == b"changed"


@pytest.mark.parametrize("failure", ["truncated", "checksum", "oversized"])
def test_failed_download_publishes_no_archive(tmp_path, failure):
    content = b"original"
    record = {
        "name": "MEDISEG.tar.gz",
        "download_url": "https://example.org/archive",
        "size": len(content),
        "computed_md5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
    }
    received = {"truncated": b"short", "checksum": b"modified", "oversized": b"too long!"}[failure]
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=received))
    ) as client:
        with pytest.raises(ValueError):
            acquisition.download(client, record, tmp_path)
    assert not list(tmp_path.iterdir())


def test_existing_partial_download_is_preserved(tmp_path):
    pending = tmp_path / "MEDISEG.tar.gz.part"
    pending.write_bytes(b"another run")
    record = {"name": "MEDISEG.tar.gz", "size": 100}
    with httpx.Client() as client, pytest.raises(ValueError, match="partial download"):
        acquisition.download(client, record, tmp_path)
    assert pending.read_bytes() == b"another run"


@pytest.mark.parametrize("value", [[], [[1, 2, 3]], [[0, 0, 1, 1, float("nan"), 2]]])
def test_invalid_annotation_polygons_are_refused(value):
    with pytest.raises(ValueError):
        inspection.polygon_mask({"segmentation": value}, 64, 64)


def dataset(tmp_path):
    root = tmp_path / "MEDISEG"
    image_dir = root / "32pills/images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (160, 160), "white").save(image_dir / "pill.jpg")
    (root / "metadata.csv").write_text(
        "id,name,certificate_holder,ingredients/0,ingredients/1,url\n"
        "HK-00001,<script>Example</script>,Example holder,example,,https://example.org\n",
        encoding="utf-8-sig",
    )
    manifest = {
        "categories": [{"id": 1, "name": "HK-00001"}],
        "images": [{"id": 1, "file_name": "pill.jpg", "width": 160, "height": 160}],
        "annotations": [
            {
                "image_id": 1,
                "category_id": 1,
                "area": 100,
                "segmentation": [[40, 40, 80, 40, 80, 80, 40, 80]],
            }
        ],
    }
    (root / "32pills/annotations.json").write_text(json.dumps(manifest), encoding="utf-8")
    return image_dir / "pill.jpg"


def test_inspection_preserves_photos_and_does_not_claim_verified_identity(tmp_path):
    photo = dataset(tmp_path)
    inspection.inventory(tmp_path)
    output = tmp_path / "inspection"
    assert (output / "images/HK-00001.jpg").read_bytes() == photo.read_bytes()
    row = json.loads((output / "appearances.json").read_text())[0]
    assert row["production_product_link"] is None
    assert row["imprint_front"] is None
    assert row["identity_review_status"] == "publisher_label_only"
    assert not row["current_registration_verified"]
    assert "<script>Example" not in (output / "report.html").read_text()
    protocol = json.loads((output / "evaluation-manifest.json").read_text())
    assert protocol["scope"] == "unverified-local"
    assert protocol["cases"][0]["capture_session"] == "MEDISEG-publisher-session-unknown"


def test_corrupt_source_image_prevents_inspection_publication(tmp_path):
    photo = dataset(tmp_path)
    photo.write_bytes(b"invalid JPEG")
    with pytest.raises((ValueError, OSError)):
        inspection.inventory(tmp_path)
    assert not (tmp_path / "inspection").exists()


def test_registry_field_uses_exact_unique_label_and_decodes_entities():
    document = "<td>Product Name</td><td>:</td><td>A &amp; B</td>"
    assert registry.field(document, "Product Name") == "A & B"
    assert registry.field(document, "Registration No.") is None
    assert registry.field(document + document, "Product Name") is None
    assert registry.field("<td>Product Name</td><td>:</td><td></td>", "Product Name") is None
