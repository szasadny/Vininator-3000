"""Tests for the terroir joiner.

All offline. Each test constructs minimal climate.parquet + soil.parquet
fixtures under the `tmp_data_dir` conftest path (via the canonical
CLIMATE_SCHEMA / SOIL_SCHEMA so the join contract stays pinned to the
upstream schemas), then exercises `build_terroir_table` against them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from vininator.config import get_settings
from vininator.features.climate import CLIMATE_SCHEMA
from vininator.features.soil import SOIL_SCHEMA
from vininator.features.terroir import (
    TERROIR_SCHEMA,
    build_terroir_table,
    scan_terroir,
)

_FETCHED = datetime(2026, 5, 28, tzinfo=UTC)


def _climate_row(
    region: str, country: str | None, vintage_year: int, **overrides: Any
) -> dict[str, Any]:
    """One row matching CLIMATE_SCHEMA, with plausible Bordeaux-ish defaults."""
    row: dict[str, Any] = {
        "region": region,
        "country": country,
        "lat": 45.0,
        "lon": 3.0,
        "vintage_year": vintage_year,
        "gdd_10c": 1500.0,
        "precip_total_mm": 400.0,
        "precip_harvest_mm": 50.0,
        "heat_spike_days": 5,
        "frost_days_spring": 1,
        "diurnal_range_mean": 12.0,
        "solar_total_mj": 4500.0,
        "gdd_10c_anom": 0.0,
        "precip_total_mm_anom": 0.0,
        "precip_harvest_mm_anom": 0.0,
        "heat_spike_days_anom": 0.0,
        "frost_days_spring_anom": 0.0,
        "diurnal_range_mean_anom": 0.0,
        "solar_total_mj_anom": 0.0,
        "is_partial": False,
        "status": "ok",
        "error": None,
        "fetched_at": _FETCHED,
    }
    row.update(overrides)
    return row


def _soil_row(region: str, country: str | None, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "region": region,
        "country": country,
        "lat": 45.0,
        "lon": 3.0,
        "clay_pct": 25.0,
        "sand_pct": 40.0,
        "silt_pct": 35.0,
        "ph_h2o": 7.6,
        "soc_gkg": 20.0,
        "cec_cmolkg": 15.0,
        "bdod_kgdm3": 1.3,
        "coarse_frag_pct": 10.0,
        "elevation_m": 250.0,
        "slope_deg": 3.5,
        "drainage_class": "loamy",
        "calcareous": True,
        "status": "ok",
        "error": None,
        "fetched_at": _FETCHED,
    }
    row.update(overrides)
    return row


def _write_climate(rows: list[dict[str, Any]]) -> None:
    df = pl.DataFrame(rows, schema=CLIMATE_SCHEMA)
    df.write_parquet(get_settings().climate_parquet)


def _write_soil(rows: list[dict[str, Any]]) -> None:
    df = pl.DataFrame(rows, schema=SOIL_SCHEMA)
    df.write_parquet(get_settings().soil_parquet)


# ---------------------------------------------------------------------------
# Join correctness
# ---------------------------------------------------------------------------


def test_build_terroir_table_joins_on_region_country(tmp_data_dir: Path) -> None:
    _write_climate([_climate_row("Rioja", "Spain", 2018, gdd_10c=1450.0)])
    _write_soil([_soil_row("Rioja", "Spain", clay_pct=22.5)])

    df = pl.read_parquet(build_terroir_table())

    assert df.height == 1
    assert df["region"][0] == "Rioja"
    assert df["country"][0] == "Spain"
    assert df["vintage_year"][0] == 2018
    assert df["gdd_10c"][0] == 1450.0
    assert df["clay_pct"][0] == 22.5


def test_build_terroir_table_broadcasts_soil_across_vintages(tmp_data_dir: Path) -> None:
    _write_climate([_climate_row("Rioja", "Spain", y) for y in range(2017, 2022)])
    _write_soil([_soil_row("Rioja", "Spain", clay_pct=22.5, ph_h2o=7.8)])

    df = pl.read_parquet(build_terroir_table()).sort("vintage_year")

    assert df.height == 5
    assert df["vintage_year"].to_list() == [2017, 2018, 2019, 2020, 2021]
    assert (df["clay_pct"] == 22.5).all()
    assert (df["ph_h2o"] == 7.8).all()


def test_build_terroir_table_disambiguates_rioja_argentina_vs_spain(
    tmp_data_dir: Path,
) -> None:
    _write_climate(
        [
            _climate_row("Rioja", "Spain", 2018, gdd_10c=1450.0),
            _climate_row("Rioja", "Argentina", 2018, gdd_10c=1700.0),
        ]
    )
    _write_soil(
        [
            _soil_row("Rioja", "Spain", clay_pct=22.5),
            _soil_row("Rioja", "Argentina", clay_pct=35.0),
        ]
    )

    df = pl.read_parquet(build_terroir_table()).sort("country")

    assert df["country"].to_list() == ["Argentina", "Spain"]
    spain = df.filter(pl.col("country") == "Spain")
    argentina = df.filter(pl.col("country") == "Argentina")
    assert spain["gdd_10c"][0] == 1450.0
    assert spain["clay_pct"][0] == 22.5
    assert argentina["gdd_10c"][0] == 1700.0
    assert argentina["clay_pct"][0] == 35.0


def test_build_terroir_table_left_join_keeps_climate_when_soil_missing(
    tmp_data_dir: Path,
) -> None:
    _write_climate([_climate_row("Lonely", "X", 2018, gdd_10c=1500.0)])
    _write_soil([])

    df = pl.read_parquet(build_terroir_table())

    assert df.height == 1
    assert df["gdd_10c"][0] == 1500.0
    assert df["clay_pct"][0] is None
    assert df["soil_status"][0] is None
    assert df["soil_error"][0] is None
    assert df["soil_fetched_at"][0] is None


def test_build_terroir_table_drops_soil_when_no_climate_row(tmp_data_dir: Path) -> None:
    """Climate-driven join: a soil row without a matching climate row is gone."""
    _write_climate([])
    _write_soil([_soil_row("Orphan", "X")])

    df = pl.read_parquet(build_terroir_table())

    assert df.height == 0


# ---------------------------------------------------------------------------
# Status / provenance plumbing
# ---------------------------------------------------------------------------


def test_build_terroir_table_renames_status_columns(tmp_data_dir: Path) -> None:
    _write_climate(
        [_climate_row("R", "X", 2018, status="partial", error="frost gap")]
    )
    _write_soil([_soil_row("R", "X", status="ok", error=None)])

    df = pl.read_parquet(build_terroir_table())

    assert "status" not in df.columns
    assert "error" not in df.columns
    assert "fetched_at" not in df.columns
    assert df["climate_status"][0] == "partial"
    assert df["climate_error"][0] == "frost gap"
    assert df["soil_status"][0] == "ok"
    assert df["soil_error"][0] is None


def test_build_terroir_table_preserves_is_partial_and_anomalies(
    tmp_data_dir: Path,
) -> None:
    _write_climate(
        [
            _climate_row(
                "R",
                "X",
                2021,
                is_partial=True,
                status="partial",
                gdd_10c_anom=120.5,
                precip_total_mm_anom=-30.0,
            )
        ]
    )
    _write_soil([_soil_row("R", "X")])

    df = pl.read_parquet(build_terroir_table())

    assert bool(df["is_partial"][0]) is True
    assert df["gdd_10c_anom"][0] == 120.5
    assert df["precip_total_mm_anom"][0] == -30.0


# ---------------------------------------------------------------------------
# Schema contract + write semantics
# ---------------------------------------------------------------------------


def test_build_terroir_table_schema_matches_expected(tmp_data_dir: Path) -> None:
    _write_climate([_climate_row("R", "X", 2018)])
    _write_soil([_soil_row("R", "X")])

    df = pl.read_parquet(build_terroir_table())

    assert dict(df.schema) == TERROIR_SCHEMA
    assert list(df.columns) == list(TERROIR_SCHEMA.keys())


def test_build_terroir_table_atomic_write(tmp_data_dir: Path) -> None:
    """No `.tmp` left behind after a successful build."""
    _write_climate([_climate_row("R", "X", 2018)])
    _write_soil([_soil_row("R", "X")])

    target = get_settings().terroir_parquet
    build_terroir_table()

    tmp_files = list(target.parent.glob("*.tmp"))
    assert tmp_files == []
    assert target.exists()


def test_scan_terroir_raises_when_missing(tmp_data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="terroir"):
        scan_terroir()
