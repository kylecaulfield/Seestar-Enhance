"""Tests for the bundled Gaia DR3 catalog (Phase 1 of SPCC).

These tests validate the schema + sanity of the bundled Parquet file
when it's present, and skip cleanly when it isn't (so CI still passes
before the data file lands, and so contributors without network access
aren't forced to fetch 30 MB of catalog data).
"""

from __future__ import annotations

import pytest
from app.data import GAIA_PARQUET, gaia_catalog_exists

_EXPECTED_COLUMNS = {
    "ra",
    "dec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
}


pytestmark = pytest.mark.skipif(
    not gaia_catalog_exists(),
    reason=(
        f"Gaia catalog Parquet not found at {GAIA_PARQUET}. "
        "Run `python backend/scripts/fetch_gaia.py` to populate it."
    ),
)


def _read_table():
    """Load the bundled Parquet with pyarrow — no pandas dependency."""
    import pyarrow.parquet as pq

    return pq.read_table(GAIA_PARQUET)


def test_schema_matches() -> None:
    """Expected columns and dtypes present."""
    import pyarrow as pa

    table = _read_table()
    assert set(table.schema.names) == _EXPECTED_COLUMNS, (
        f"schema mismatch: {set(table.schema.names)} vs {_EXPECTED_COLUMNS}"
    )
    # ra/dec may be float32 or float64. float32 is sufficient for SPCC
    # (~0.4 arcsec precision at RA=360, inside the 2-arcsec match
    # tolerance) and saves 24 MB on a 3M-row full-sky bundle.
    for coord in ("ra", "dec"):
        assert table.schema.field(coord).type in (pa.float32(), pa.float64())
    for mag in ("phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"):
        assert table.schema.field(mag).type == pa.float32()


def test_row_count_reasonable() -> None:
    """At least a minimum number of rows so SPCC has something to match.

    Even at G < 10 (the smallest plausible bundle) we'd expect ~30 000
    sources; anything below a thousand suggests the fetch went wrong."""
    table = _read_table()
    assert len(table) > 1_000, f"suspiciously few rows: {len(table)}"


def test_ra_dec_ranges() -> None:
    """RA in [0, 360), Dec in [-90, +90]."""
    import numpy as np

    table = _read_table()
    ra = table.column("ra").to_numpy()
    dec = table.column("dec").to_numpy()
    assert float(ra.min()) >= 0.0
    assert float(ra.max()) < 360.0
    assert float(dec.min()) >= -90.0
    assert float(dec.max()) <= 90.0
    assert not np.isnan(ra).any()
    assert not np.isnan(dec).any()


def test_magnitudes_reasonable() -> None:
    """Magnitudes populated and inside the physically-plausible range."""
    import numpy as np

    table = _read_table()
    for col in ("phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"):
        mag = table.column(col).to_numpy()
        assert not np.isnan(mag).any(), f"{col} has NaNs"
        # Brighter than Sirius-class is unusual; fainter than 25 is
        # outside Seestar's useful range. Sanity-only bounds.
        assert float(mag.min()) > -2.0, f"{col} min={mag.min()} below realistic floor"
        assert float(mag.max()) < 25.0, f"{col} max={mag.max()} above realistic ceiling"


def test_bp_rp_bounded() -> None:
    """BP-RP colour index should land in the usual stellar range."""
    import numpy as np

    table = _read_table()
    bp = table.column("phot_bp_mean_mag").to_numpy()
    rp = table.column("phot_rp_mean_mag").to_numpy()
    bp_rp = bp - rp
    # Typical stars fall in [-0.5, +4]; outliers exist but a median
    # outside that band would suggest the column mapping is wrong.
    median = float(np.median(bp_rp))
    assert -0.5 < median < 4.0, f"median BP-RP={median:.2f} outside plausible range"
