"""Fetch a bright-star subset of Gaia DR3 for Photometric Color Calibration.

One-off script; not part of the runtime pipeline. Run it to populate
`backend/app/data/gaia_bright.parquet`, commit the result, and SPCC
will use it offline without any network access.

Why Gaia DR3? Most complete all-sky photometric catalog available (1.8 B
sources, uniform calibration), CC0-licensed so we can bundle it directly.

Why a subset? Full catalog is ~600 GB. SPCC only needs:
  - Stars bright enough to show up in 10 s Seestar subs (G ≲ 12).
  - Five columns: ra, dec, phot_g_mean_mag, phot_bp_mean_mag,
    phot_rp_mean_mag.
That's ~5 M rows at G < 12 — ~30 MB as snappy-compressed Parquet.

Mirrors
-------
Two sources, tried in order:

  1. ESA Gaia TAP service (`gea.esac.esa.int`) — canonical, but
     the archive has occasional multi-hour outages.
  2. CDS Vizier TAP (`tapvizier.u-strasbg.fr`) — mirrors Gaia DR3
     as table `I/355/gaiadr3`; typically more reliable but slower
     for large result sets.

Usage
-----
    cd backend
    python scripts/fetch_gaia.py

Environment variables (all optional):
    GAIA_G_LIMIT      — magnitude cutoff (default: 12.0).
    GAIA_OUTPUT       — output path.
    GAIA_SOURCE       — "esa" | "vizier" | "auto" (default: auto).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional


_ADQL_TEMPLATE_ESA = """
SELECT
    ra,
    dec,
    phot_g_mean_mag,
    phot_bp_mean_mag,
    phot_rp_mean_mag
FROM gaiadr3.gaia_source
WHERE phot_g_mean_mag < {g_limit}
  AND phot_bp_mean_mag IS NOT NULL
  AND phot_rp_mean_mag IS NOT NULL
ORDER BY source_id
""".strip()


def fetch_esa(g_limit: float, use_async: bool = True):
    """Fetch from the ESA Gaia TAP service. Canonical source."""
    from astroquery.gaia import Gaia

    query = _ADQL_TEMPLATE_ESA.format(g_limit=f"{g_limit:.2f}")
    print(f"ESA TAP: G < {g_limit}, async={use_async} ...")
    t0 = time.time()
    if use_async:
        job = Gaia.launch_job_async(query)
    else:
        job = Gaia.launch_job(query)
    results = job.get_results()
    dt = time.time() - t0
    print(f"  received {len(results):,} rows in {dt:.1f} s")
    return results


def fetch_vizier(g_limit: float):
    """Fallback: CDS Vizier mirror of Gaia DR3 (table I/355/gaiadr3)."""
    from astroquery.vizier import Vizier

    print(f"Vizier fallback: I/355/gaiadr3, Gmag < {g_limit} ...")
    v = Vizier(
        columns=["RA_ICRS", "DE_ICRS", "Gmag", "BPmag", "RPmag"],
        row_limit=-1,
        column_filters={"Gmag": f"<{g_limit}"},
    )
    t0 = time.time()
    catalog_list = v.query_constraints(catalog="I/355/gaiadr3")
    dt = time.time() - t0
    if catalog_list is None or len(catalog_list) == 0:
        raise RuntimeError("Vizier query returned no tables")
    table = catalog_list[0]
    print(f"  received {len(table):,} rows in {dt:.1f} s")

    # Rename Vizier column names to the ESA-native schema so the rest
    # of the pipeline doesn't care which mirror fed it.
    table.rename_column("RA_ICRS", "ra")
    table.rename_column("DE_ICRS", "dec")
    table.rename_column("Gmag", "phot_g_mean_mag")
    table.rename_column("BPmag", "phot_bp_mean_mag")
    table.rename_column("RPmag", "phot_rp_mean_mag")
    return table


def fetch(g_limit: float, source: str = "auto"):
    """Try ESA first, fall back to Vizier when `source='auto'`.

    Either one is an acceptable source of Gaia DR3 photometry. The
    fallback exists because ESA's archive has occasional multi-hour
    outages and we'd rather finish a one-off script than fail on a
    transient.
    """
    if source == "esa":
        return fetch_esa(g_limit=g_limit)
    if source == "vizier":
        return fetch_vizier(g_limit=g_limit)
    if source != "auto":
        raise ValueError(f"unknown source {source!r}")
    try:
        return fetch_esa(g_limit=g_limit)
    except Exception as exc:  # noqa: BLE001
        print(f"  ESA TAP failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  falling back to Vizier", file=sys.stderr)
        return fetch_vizier(g_limit=g_limit)


def save_parquet(table, path: Path) -> None:
    """Save an astropy Table as Parquet with snappy compression."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)

    # Build an Arrow table column-by-column with compact dtypes.
    # ra/dec stay float64 (needed for arcsec-scale cross-match
    # accuracy); magnitudes downcast to float32 (Gaia quotes them to
    # ~0.01 mag, more than float32 resolution already).
    arrays = {
        "ra": pa.array(np.asarray(table["ra"], dtype=np.float64)),
        "dec": pa.array(np.asarray(table["dec"], dtype=np.float64)),
        "phot_g_mean_mag": pa.array(
            np.asarray(table["phot_g_mean_mag"], dtype=np.float32)
        ),
        "phot_bp_mean_mag": pa.array(
            np.asarray(table["phot_bp_mean_mag"], dtype=np.float32)
        ),
        "phot_rp_mean_mag": pa.array(
            np.asarray(table["phot_rp_mean_mag"], dtype=np.float32)
        ),
    }
    arrow_table = pa.Table.from_pydict(arrays)
    pq.write_table(
        arrow_table,
        str(path),
        compression="snappy",
        row_group_size=100_000,
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Wrote {path} ({size_mb:.2f} MB, {len(arrow_table):,} rows)")


def main(argv: Optional[list[str]] = None) -> int:
    g_limit = float(os.environ.get("GAIA_G_LIMIT", "12.0"))
    source = os.environ.get("GAIA_SOURCE", "auto")
    out_path = Path(
        os.environ.get(
            "GAIA_OUTPUT",
            Path(__file__).resolve().parent.parent / "app" / "data" / "gaia_bright.parquet",
        )
    )

    table = fetch(g_limit=g_limit, source=source)
    if len(table) == 0:
        print("ERROR: query returned zero rows", file=sys.stderr)
        return 1

    save_parquet(table, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
