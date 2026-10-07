"""Search reviewed references with manual observations; no image inference or database writes."""

import argparse
import html
import tempfile
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from rxsentinel.catalog import configured_catalog
from rxsentinel.matching import MatchQuery, match_observations
from rxsentinel.pipelines.photo import bounded_read


def write_report(report, output):
    output = output.resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rxsentinel-matching-", dir=output.parent) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        (stage / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        candidates = "".join(
            f"<article><h2>{html.escape(c.appearance_id)}</h2>"
            f"<p>Catalog products: {html.escape(', '.join(p.product_id for p in c.products))}</p>"
            f"<p>Imprints: {html.escape(c.imprint_front or '')} / "
            f"{html.escape(c.imprint_back or '')}</p>"
            f"<p>Color: {html.escape(c.color or 'Unknown')}; "
            f"shape: {html.escape(c.shape or 'Unknown')}</p>"
            f"<p>Evidence points: {c.evidence_points} (not a confidence percentage).</p>"
            "<p>Unresolved observations: "
            f"{html.escape(', '.join(c.uncertainties) or 'None recorded')}</p>"
            "<p>Reference image IDs: "
            f"{html.escape(', '.join(a.asset_id for a in c.reference_images))}</p>"
            "<p>Compare references and packaging before confirming identity.</p></article>"
            for c in report.candidates
        )
        page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RxSentinel manual observation matching</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:1rem}}
article{{border-top:1px solid #aaa;padding:1rem 0}}code{{overflow-wrap:anywhere}}</style>
<h1>Manual observation matching: {report.status}</h1>
<p>{html.escape(report.reason.replace("_", " "))}.</p>
<p>Eligible appearances: {report.eligible_appearance_count}.</p>
<p>Total candidates: {report.total_candidate_count}; displayed: {len(report.candidates)}.</p>
<p>No medication identity is confirmed. This feature does not read photos or assess dose.</p>
{candidates}<p><a href="report.json">Full evidence, reference IDs, versions and exclusions</a></p>
</html>"""
        (stage / "report.html").write_text(page, encoding="utf-8")
        stage.rename(output)
    return output / "report.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True, help="MatchQuery JSON")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, help="Optional new directory for a local report")
    args = parser.parse_args()
    catalog = None
    try:
        query = MatchQuery.model_validate_json(bounded_read(args.query))
        catalog = configured_catalog(args.data_dir)
        result = match_observations(catalog, args.data_dir, query)
        if args.output_dir:
            write_report(result, args.output_dir)
        print(result.model_dump_json(indent=2))
        return 0
    except SQLAlchemyError:
        parser.exit(1, "Configured database unavailable or schema not initialized\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Matching failed: {error}\n")
    finally:
        if catalog and catalog.store:
            catalog.store.engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
