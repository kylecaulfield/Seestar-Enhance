"""Bundled data files (Gaia DR3 subset)."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
GAIA_PARQUET = DATA_DIR / "gaia_bright.parquet"


def gaia_catalog_path() -> Path:
    """Return the filesystem path to the bundled Gaia DR3 Parquet file."""
    return GAIA_PARQUET


def gaia_catalog_exists() -> bool:
    """True if the bundled Gaia Parquet exists locally."""
    return GAIA_PARQUET.is_file()
