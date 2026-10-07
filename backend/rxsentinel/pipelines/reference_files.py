"""Bounded inspection of individual archived files; no bulk ZIP extraction."""

import hashlib
import io
import warnings
import zipfile
from pathlib import Path

import httpx
from PIL import Image

ARCHIVE_URL = "https://ftp.nlm.nih.gov/projects/pillbox/pillbox_production_images_full_202008.zip"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20_000_000


class RangeReader(io.RawIOBase):
    """Seekable ZIP reader using strict HTTP ranges and a per-run transfer budget."""

    def __init__(self, client: httpx.Client, url: str = ARCHIVE_URL):
        self.client, self.url, self.position = client, url, 0
        response = client.head(url, headers={"Accept-Encoding": "identity"})
        response.raise_for_status()
        self.size = int(response.headers.get("content-length", "0"))
        if not 22 <= self.size <= 2 * 1024**3:
            raise ValueError("Archive size is missing or outside the supported range")
        self.etag = response.headers.get("etag")
        self.modified = response.headers.get("last-modified")
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if whence not in (0, 1, 2) or not 0 <= position <= self.size:
            raise ValueError("Invalid archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        count = min(self.size - self.position, size if size >= 0 else self.size)
        if count == 0:
            return b""
        if count > 4 * 1024**2 or self.transferred + count > 20 * 1024**2:
            raise ValueError("Archive inspection transfer limit exceeded")
        start, end = self.position, self.position + count - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Match"] = self.etag
        elif self.modified:
            headers["If-Unmodified-Since"] = self.modified
        with self.client.stream("GET", self.url, headers=headers) as response:
            if response.status_code != 206:
                raise ValueError("Archive server did not honor a partial-content request")
            if response.headers.get("content-range") != f"bytes {start}-{end}/{self.size}":
                raise ValueError("Archive returned an inconsistent content range")
            if response.headers.get("content-encoding", "identity") != "identity":
                raise ValueError("Archive ranges must be transferred without content encoding")
            chunks, actual = [], 0
            for chunk in response.iter_raw():
                actual += len(chunk)
                if actual > count:
                    raise ValueError("Archive returned more bytes than requested")
                chunks.append(chunk)
            if actual != count:
                raise ValueError("Archive returned a truncated range")
        self.position += count
        self.transferred += count
        return b"".join(chunks)


def inspect_image(content: bytes) -> tuple[str, int, int]:
    try:
        return _inspect_image(content)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise ValueError("Image pixel limit exceeded") from error


def _inspect_image(content: bytes) -> tuple[str, int, int]:
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Image byte limit exceeded")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in ("JPEG", "PNG") or image.width * image.height > MAX_PIXELS:
                raise ValueError("Unsupported image format or excessive pixel count")
            kind, width, height = image.format, image.width, image.height
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            image.load()  # Verify that the pixel data actually decodes, not just the header.
    return kind, width, height


def save_image(data_dir: Path, content: bytes) -> tuple[str, str, int, int]:
    kind, width, height = inspect_image(content)
    sha = hashlib.sha256(content).hexdigest()
    relative = f"assets/pillbox-inspection/{sha}.{'jpg' if kind == 'JPEG' else 'png'}"
    target = data_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != content:
        raise ValueError("Existing content-addressed image failed its integrity check")
    target.write_bytes(content)
    return relative, sha, width, height


def archived_image(archive: zipfile.ZipFile, identifier: str) -> tuple[str, bytes]:
    # Exact basename only; archive paths are never used as filesystem destinations.
    matches = [i for i in archive.infolist() if Path(i.filename).name == f"{identifier}.jpg"]
    if len(matches) != 1:
        raise ValueError("Archived image identifier is missing or ambiguous")
    info = matches[0]
    if (
        info.is_dir()
        or info.flag_bits & 1
        or info.file_size > MAX_IMAGE_BYTES
        or info.compress_size > 4 * 1024**2
        or info.file_size > max(1, info.compress_size) * 200
    ):
        raise ValueError("Archived image exceeds supported file limits")
    with archive.open(info) as stream:
        content = stream.read(MAX_IMAGE_BYTES + 1)
    # Reading the full member verifies the ZIP CRC; decoding is checked by save_image.
    if len(content) != info.file_size:
        raise ValueError("Archived member size does not match the directory")
    return info.filename, content
