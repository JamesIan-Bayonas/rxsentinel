"""Compare publisher metadata with exact current Hong Kong registration pages."""

import argparse
import asyncio
import csv
import hashlib
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path("data/external/mediseg-v2")


def text(value):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def field(document, label):
    pattern = (
        r"<td\b[^>]*>\s*"
        + re.escape(label)
        + r"\s*</td>\s*<td\b[^>]*>\s*:\s*</td>\s*<td\b[^>]*>(.*?)</td>"
    )
    matches = re.findall(pattern, document, flags=re.IGNORECASE | re.DOTALL)
    return (text(matches[0]) or None) if len(matches) == 1 else None


async def check(client, semaphore, row, output):
    record = {
        "id": row["id"],
        "url": row["url"],
        "retrieved_at": datetime.now(UTC).isoformat(),
        "publisher_metadata": row,
        "outcome": "provider_error",
        "image_identity_verified": False,
    }
    async with semaphore:
        try:
            response = await client.get(row["url"])
            response.raise_for_status()
            if len(response.content) > 2 * 1024**2 or response.url.host != "www.drugoffice.gov.hk":
                raise ValueError("Unexpected product page size or host")
            (output / f"{row['id']}.html").write_bytes(response.content)
            record["sha256"] = hashlib.sha256(response.content).hexdigest()
            record["final_url"] = str(response.url)
            observed = {
                "id": field(response.text, "Registration No."),
                "name": field(response.text, "Product Name"),
                "certificate_holder": field(response.text, "Certificate Holder"),
            }
            if not observed["id"] or not re.fullmatch(r"HK-\d{5}", observed["id"]):
                observed["id"] = None
            table = re.findall(
                r'<table\b[^>]*id="ingredientTable"[^>]*>(.*?)</table>',
                response.text,
                re.IGNORECASE | re.DOTALL,
            )
            ingredients = []
            if len(table) == 1:
                for cells in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table[0], re.DOTALL):
                    values = re.findall(r"<td\b[^>]*>(.*?)</td>", cells, re.DOTALL)
                    if values:
                        ingredients.append(text(values[0]).casefold())
            observed["ingredients"] = sorted(ingredients)
            record["observed"] = observed
            comparisons = {
                key: "missing"
                if observed[key] is None
                else ("match" if observed[key].casefold() == row[key].casefold() else "mismatch")
                for key in ("id", "name", "certificate_holder")
            }
            expected = sorted(
                v.casefold() for v in [row["ingredients/0"], row["ingredients/1"]] if v
            )
            comparisons["ingredients"] = (
                "missing"
                if not ingredients
                else "match"
                if sorted(ingredients) == expected
                else "mismatch"
            )
            record["comparisons"] = comparisons
            outcomes = set(comparisons.values())
            record["outcome"] = (
                "not_found"
                if observed["id"] is None and observed["name"] is None
                else "mismatch"
                if "mismatch" in outcomes
                else "incomplete"
                if "missing" in outcomes
                else "consistent"
            )
        except (httpx.HTTPError, ValueError) as error:
            record["error"] = str(error)
    return record


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "registration-checks-current")
    args = parser.parse_args()
    output = args.output
    output.mkdir(exist_ok=False)
    with (ROOT / "MEDISEG/metadata.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        semaphore = asyncio.Semaphore(4)
        records = await asyncio.gather(*(check(client, semaphore, row, output) for row in rows))
    summary = {
        "product_count": len(records),
        "outcomes": {
            outcome: sum(r["outcome"] == outcome for r in records)
            for outcome in sorted({r["outcome"] for r in records})
        },
        "market": "Hong Kong",
        "image_identity_verified": False,
        "note": "Registration metadata consistency does not verify photo identity, appearance, "
        "front/back pairing, capture time or U.S. product equivalence.",
    }
    (output / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
