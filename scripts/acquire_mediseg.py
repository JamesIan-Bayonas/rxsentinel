"""Download pinned licensed MEDISEG sources into a separate research collection."""

import argparse
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import httpx

ARTICLE_URL = "https://api.figshare.com/v2/articles/28574786/versions/2"
MAX_DOWNLOAD_BYTES = 512 * 1024**2
MAX_UNPACKED_BYTES = 3 * 1024**3
MAX_MEMBER_BYTES = 128 * 1024**2
MAX_MEMBERS = 30_000


def checksums(path):
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def download(client, record, directory):
    expected = record["size"]
    if not 0 < expected <= MAX_DOWNLOAD_BYTES:
        raise ValueError("Archive exceeds download budget")
    filename = record["name"]
    if Path(filename).name != filename or filename not in {
        "MEDISEG.tar.gz",
        "MEDISEG-Deploy.tar.gz",
    }:
        raise ValueError("Unexpected archive name")
    target = directory / filename
    if not target.exists():
        pending = target.with_suffix(target.suffix + ".part")
        if pending.exists():
            raise ValueError("A partial download already exists; refusing to replace it")
        try:
            with client.stream("GET", record["download_url"]) as response:
                response.raise_for_status()
                count, next_notice = 0, 50 * 1024**2
                with pending.open("xb") as stream:
                    for chunk in response.iter_bytes(1024**2):
                        count += len(chunk)
                        if count > expected:
                            raise ValueError("Download exceeds published size")
                        stream.write(chunk)
                        if count >= next_notice:
                            print(f"{filename}: {count}/{expected} bytes", flush=True)
                            next_notice += 50 * 1024**2
            if pending.stat().st_size != expected:
                raise ValueError("Truncated archive")
            if checksums(pending)[1] != record["computed_md5"]:
                raise ValueError("Published MD5 mismatch")
            pending.rename(target)
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    sha, md5 = checksums(target)
    if target.stat().st_size != expected or md5 != record["computed_md5"]:
        raise ValueError("Cached archive differs from published checksum")
    return {**record, "sha256": sha, "verified_md5": md5}


def safe_members(archive, root):
    members = archive.getmembers()
    if len(members) > MAX_MEMBERS:
        raise ValueError("Archive member count exceeds budget")
    total, files = 0, set()
    for member in members:
        name = PurePosixPath(member.name)
        target = root.joinpath(*name.parts).resolve()
        if (
            name.is_absolute()
            or ".." in name.parts
            or "\\" in member.name
            or ":" in member.name
            or not target.is_relative_to(root.resolve())
            or not (member.isfile() or member.isdir())
        ):
            raise ValueError("Unsafe archive path or member type")
        if member.isfile():
            if target in files or not 0 <= member.size <= MAX_MEMBER_BYTES:
                raise ValueError("Duplicate archive path or excessive member size")
            files.add(target)
            total += member.size
            if total > MAX_UNPACKED_BYTES:
                raise ValueError("Archive unpacked size exceeds budget")
    return members


def unpack(archive_path, root):
    # Inspect every member before writing; never follow archive links or execute content.
    with tarfile.open(archive_path, "r:gz") as archive:
        members = safe_members(archive, root)
        for member in members:
            target = root.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            content = archive.extractfile(member).read(member.size + 1)
            if len(content) != member.size:
                raise ValueError("Archive member length mismatch")
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError("Existing extracted file differs; refusing overwrite")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
    return sum(member.isfile() for member in members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/external/mediseg-v2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        response = client.get(ARTICLE_URL)
        response.raise_for_status()
        article = response.json()
        if article["version"] != 2 or article["license"]["name"] != "CC BY 4.0":
            raise ValueError("Unexpected source version or license")
        (args.output / "article.json").write_bytes(response.content)
        records = []
        for record in article["files"]:
            verified = download(client, record, args.output)
            verified["extracted_file_count"] = unpack(args.output / record["name"], args.output)
            records.append(verified)
            print(json.dumps(verified), flush=True)
    manifest = {
        "source": ARTICLE_URL,
        "doi": article["doi"],
        "retrieved_at": datetime.now(UTC).isoformat(),
        "article_sha256": hashlib.sha256((args.output / "article.json").read_bytes()).hexdigest(),
        "license": article["license"],
        "authors": article["authors"],
        "files": records,
        "scope": "External Hong Kong medication-image research dataset; no U.S. NDC linkage",
        "production_catalog_imported": False,
        "identity_approved": False,
        "m1_reference_gate_satisfied": False,
    }
    (args.output / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
