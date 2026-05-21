"""Tests for the soil + DEM feature module.

All tests are offline: the recorded JSON fixtures in `tests/data/` were captured
once by hand against the live SoilGrids and Open-Elevation APIs for the Tuscany
centroid (43.46, 11.04). The recording script lives in the project history; to
re-record, query both APIs for that point with the parameters wired into
`src/vininator/features/soil.py` and overwrite the JSON files.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from vininator.config import SOILGRIDS_BUFFER_DEG, get_settings
from vininator.data.geocode import GEOCODE_SCHEMA
from vininator.features.soil import (
    SOIL_SCHEMA,
    _aggregate_soil_properties,
    _grid_3x3,
    _horn_slope_deg,
    _property_topsoil_mean,
    build_soil_table,
    compute_soil_features,
    fetch_elevation_slope,
    fetch_soilgrids,
    region_slug,
    scan_soil,
)

FIXTURES_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Helpers — wrap recorded fixtures into the cache shape compute_* expects.
# ---------------------------------------------------------------------------


def _soil_cache_from_fixture(n_points: int = 9) -> dict:
    """A SoilGrids cache dict with `n_points` identical copies of the recorded
    Tuscany response. Identical copies mean the per-property buffer-average
    equals the single-point value — useful for testing unit conversions
    without a 3x3-of-different-values setup.
    """
    response = json.loads((FIXTURES_DIR / "soilgrids_tuscany.json").read_text())
    return {
        "region": "Tuscany",
        "country": "Italy",
        "lat": 43.46,
        "lon": 11.04,
        "buffer_deg": 0.005,
        "points": [
            {"lat": 43.46, "lon": 11.04, "response": response, "error": None}
            for _ in range(n_points)
        ],
        "status": "ok" if n_points == 9 else ("partial" if n_points > 0 else "error"),
        "fetched_at": "2026-05-21T00:00:00+00:00",
    }


def _dem_cache_from_fixture() -> dict:
    response = json.loads((FIXTURES_DIR / "open_elevation_tuscany.json").read_text())
    return {
        "region": "Tuscany",
        "country": "Italy",
        "lat": 43.46,
        "lon": 11.04,
        "grid_deg": 0.0027,
        "points": [{"lat": 0.0, "lon": 0.0} for _ in range(9)],
        "response": response,
        "status": "ok",
        "error": None,
        "fetched_at": "2026-05-21T00:00:00+00:00",
    }


def _write_fake_geocode(target: Path, rows: list[dict]) -> None:
    """Write a minimal geocode parquet so `build_soil_table` has something to
    iterate. `rows` is filled in with schema defaults where keys are missing."""
    target.parent.mkdir(parents=True, exist_ok=True)
    defaults = {
        "region": "",
        "country": None,
        "lat": 0.0,
        "lon": 0.0,
        "raw_address": None,
        "result_type": None,
        "status": "ok",
        "error": None,
        "fetched_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    pl.DataFrame(
        [{**defaults, **r} for r in rows], schema=GEOCODE_SCHEMA
    ).write_parquet(target)


# ---------------------------------------------------------------------------
# region_slug
# ---------------------------------------------------------------------------


def test_region_slug_strips_accents_and_lowercases() -> None:
    assert region_slug("Côte de Beaune", "France") == "france_cote_de_beaune"


def test_region_slug_no_country() -> None:
    assert region_slug("Napa Valley", None) == "napa_valley"


def test_region_slug_collisions_resolved_by_country() -> None:
    """Rioja (Spain) vs Rioja (Argentina) must produce distinct slugs."""
    assert region_slug("Rioja", "Spain") != region_slug("Rioja", "Argentina")


def test_region_slug_strips_punctuation() -> None:
    assert region_slug("St-Émilion!!!", "France") == "france_st_emilion"


# ---------------------------------------------------------------------------
# 3x3 grid helper
# ---------------------------------------------------------------------------


def test_grid_3x3_row_major_order() -> None:
    grid = _grid_3x3(0.0, 0.0, 1.0)
    assert len(grid) == 9
    # First row: dy=-1, dx=-1,0,1 → (-1, -1), (-1, 0), (-1, 1)
    assert grid[0] == (-1.0, -1.0)
    assert grid[4] == (0.0, 0.0)   # centre
    assert grid[8] == (1.0, 1.0)   # bottom-right


# ---------------------------------------------------------------------------
# Horn's slope algorithm
# ---------------------------------------------------------------------------


def test_horn_slope_flat_is_zero() -> None:
    assert _horn_slope_deg([100.0] * 9, lat=45.0) == 0.0


def test_horn_slope_pure_dy_tilt() -> None:
    """Elevation rises uniformly south-to-north → slope > 0, dz/dx == 0."""
    grid = [10.0, 10.0, 10.0,
            20.0, 20.0, 20.0,
            30.0, 30.0, 30.0]
    slope = _horn_slope_deg(grid, lat=0.0)
    assert slope > 0.0
    # 10 m rise over 8 * cell_y at equator (cell_y ≈ 300.6 m)
    # dzdy = ((30 + 60 + 30) - (10 + 20 + 10)) / (8 * 300.564) = 80/2404.5 ≈ 0.03327
    assert slope == pytest.approx(1.907, abs=0.05)


def test_horn_slope_matches_tuscany_fixture() -> None:
    """Concrete pinned slope angle from the recorded Open-Elevation grid."""
    response = json.loads((FIXTURES_DIR / "open_elevation_tuscany.json").read_text())
    elevs = [r["elevation"] for r in response["results"]]
    slope = _horn_slope_deg(elevs, lat=43.46)
    assert slope == pytest.approx(4.58, abs=0.01)


def test_horn_slope_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected 9 elevations"):
        _horn_slope_deg([1.0, 2.0, 3.0], lat=0.0)


# ---------------------------------------------------------------------------
# _property_topsoil_mean — depth-weighted aggregation across the 3 bands
# ---------------------------------------------------------------------------


def test_property_topsoil_mean_weighted_average() -> None:
    """Weights are 5/30, 10/30, 15/30 — verify against a hand-built response."""
    response = {
        "properties": {
            "layers": [
                {
                    "name": "phh2o",
                    "depths": [
                        {"label": "0-5cm",  "values": {"mean": 60}},
                        {"label": "5-15cm", "values": {"mean": 70}},
                        {"label": "15-30cm","values": {"mean": 80}},
                    ],
                }
            ]
        }
    }
    # 60 * 1/6 + 70 * 1/3 + 80 * 1/2 = 10 + 23.333 + 40 = 73.333
    assert _property_topsoil_mean(response, "phh2o") == pytest.approx(73.333, abs=1e-3)


def test_property_topsoil_mean_skips_missing_bands() -> None:
    response = {
        "properties": {
            "layers": [
                {
                    "name": "clay",
                    "depths": [
                        {"label": "0-5cm",  "values": {"mean": None}},
                        {"label": "5-15cm", "values": {"mean": 200}},
                        {"label": "15-30cm","values": {"mean": 300}},
                    ],
                }
            ]
        }
    }
    # Only 5-15 and 15-30 contribute. Weights 10/30 + 15/30 = 25/30.
    # weighted_sum = 200*(10/30) + 300*(15/30) = 66.667 + 150 = 216.667
    # / 25/30 = 216.667 * 30/25 = 260
    assert _property_topsoil_mean(response, "clay") == pytest.approx(260.0, abs=1e-3)


def test_property_topsoil_mean_all_null_returns_none() -> None:
    response = {
        "properties": {
            "layers": [
                {
                    "name": "soc",
                    "depths": [
                        {"label": "0-5cm",  "values": {"mean": None}},
                        {"label": "5-15cm", "values": {"mean": None}},
                        {"label": "15-30cm","values": {"mean": None}},
                    ],
                }
            ]
        }
    }
    assert _property_topsoil_mean(response, "soc") is None


def test_property_topsoil_mean_unknown_property_returns_none() -> None:
    response = {"properties": {"layers": [{"name": "phh2o", "depths": []}]}}
    assert _property_topsoil_mean(response, "clay") is None


# ---------------------------------------------------------------------------
# Unit conversions applied
# ---------------------------------------------------------------------------


def test_unit_conversions_against_tuscany_fixture() -> None:
    """The recorded Tuscany response, run through aggregation + d-factor div,
    should produce these pinned numbers. If ISRIC changes the d_factor or
    returned units, this test screams."""
    raw_soil = _soil_cache_from_fixture(n_points=9)
    features = _aggregate_soil_properties(raw_soil)
    assert features["ph_h2o"] == pytest.approx(7.78, abs=0.01)
    assert features["clay_pct"] == pytest.approx(30.47, abs=0.01)
    assert features["sand_pct"] == pytest.approx(30.53, abs=0.01)
    assert features["silt_pct"] == pytest.approx(39.00, abs=0.01)
    assert features["soc_gkg"] == pytest.approx(24.27, abs=0.01)
    assert features["cec_cmolkg"] == pytest.approx(19.28, abs=0.01)
    assert features["bdod_kgdm3"] == pytest.approx(1.36, abs=0.01)
    assert features["coarse_frag_pct"] == pytest.approx(10.45, abs=0.01)


def test_aggregate_skips_failed_points() -> None:
    """Failed (error) points don't poison the buffer average."""
    response = {
        "properties": {
            "layers": [{
                "name": "phh2o",
                "depths": [{"label": "0-5cm", "values": {"mean": 70}}],
            }]
        }
    }
    raw_soil = {
        "points": [
            {"lat": 0, "lon": 0, "response": response, "error": None},
            {"lat": 0, "lon": 0, "response": None, "error": "boom"},
            {"lat": 0, "lon": 0, "response": None, "error": "boom"},
        ]
    }
    features = _aggregate_soil_properties(raw_soil)
    # Single 0-5cm band, total_weight = 5/30. value 70 / total_weight = 70.
    # Then /10 divisor = 7.0
    assert features["ph_h2o"] == pytest.approx(7.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_soil_features — drainage_class precedence + calcareous
# ---------------------------------------------------------------------------


def test_compute_features_tuscany_is_calcareous_chalky() -> None:
    """pH 7.78 → calcareous → drainage_class 'chalky' even with 30% clay."""
    out = compute_soil_features(_soil_cache_from_fixture(), _dem_cache_from_fixture())
    assert out["calcareous"] is True
    assert out["drainage_class"] == "chalky"
    assert out["status"] == "ok"
    assert out["error"] is None
    assert out["elevation_m"] == pytest.approx(269.0, abs=1e-6)
    assert out["slope_deg"] == pytest.approx(4.58, abs=0.01)


def test_compute_features_clayey_when_not_calcareous() -> None:
    response = {
        "properties": {
            "layers": [
                {"name": "phh2o", "depths": [{"label": "5-15cm", "values": {"mean": 55}}]},  # pH 5.5
                {"name": "clay",  "depths": [{"label": "5-15cm", "values": {"mean": 500}}]},  # 50 %
                {"name": "sand",  "depths": [{"label": "5-15cm", "values": {"mean": 200}}]},  # 20 %
                {"name": "silt",  "depths": [{"label": "5-15cm", "values": {"mean": 300}}]},
                {"name": "soc",   "depths": [{"label": "5-15cm", "values": {"mean": 100}}]},
                {"name": "cec",   "depths": [{"label": "5-15cm", "values": {"mean": 100}}]},
                {"name": "bdod",  "depths": [{"label": "5-15cm", "values": {"mean": 130}}]},
                {"name": "cfvo",  "depths": [{"label": "5-15cm", "values": {"mean": 50}}]},
            ]
        }
    }
    raw_soil = {"points": [{"response": response}] * 9, "status": "ok"}
    out = compute_soil_features(raw_soil, _dem_cache_from_fixture())
    assert out["calcareous"] is False
    assert out["drainage_class"] == "clayey"


def test_compute_features_sandy_when_high_sand_low_clay() -> None:
    response = {
        "properties": {
            "layers": [
                {"name": "phh2o", "depths": [{"label": "5-15cm", "values": {"mean": 60}}]},
                {"name": "clay",  "depths": [{"label": "5-15cm", "values": {"mean": 100}}]},  # 10 %
                {"name": "sand",  "depths": [{"label": "5-15cm", "values": {"mean": 700}}]},  # 70 %
                {"name": "silt",  "depths": [{"label": "5-15cm", "values": {"mean": 200}}]},
            ]
        }
    }
    raw_soil = {"points": [{"response": response}] * 9, "status": "ok"}
    out = compute_soil_features(raw_soil, _dem_cache_from_fixture())
    assert out["drainage_class"] == "sandy"


def test_compute_features_loamy_as_default() -> None:
    response = {
        "properties": {
            "layers": [
                {"name": "phh2o", "depths": [{"label": "5-15cm", "values": {"mean": 65}}]},
                {"name": "clay",  "depths": [{"label": "5-15cm", "values": {"mean": 200}}]},  # 20 %
                {"name": "sand",  "depths": [{"label": "5-15cm", "values": {"mean": 400}}]},  # 40 %
                {"name": "silt",  "depths": [{"label": "5-15cm", "values": {"mean": 400}}]},
            ]
        }
    }
    raw_soil = {"points": [{"response": response}] * 9, "status": "ok"}
    out = compute_soil_features(raw_soil, _dem_cache_from_fixture())
    assert out["drainage_class"] == "loamy"


def test_compute_features_partial_status_when_dem_failed() -> None:
    raw_dem = {**_dem_cache_from_fixture(), "status": "error", "response": None, "error": "boom"}
    out = compute_soil_features(_soil_cache_from_fixture(), raw_dem)
    assert out["status"] == "partial"
    assert "dem=error" in out["error"]
    assert out["elevation_m"] is None
    assert out["slope_deg"] is None


def test_compute_features_full_error_when_both_failed() -> None:
    raw_soil = {"points": [], "status": "error"}
    raw_dem = {**_dem_cache_from_fixture(), "status": "error", "response": None}
    out = compute_soil_features(raw_soil, raw_dem)
    assert out["status"] == "error"
    assert out["ph_h2o"] is None
    assert out["drainage_class"] is None  # no clay/sand → no bucket


def test_compute_features_partial_when_soilgrids_returns_all_null_means() -> None:
    """SoilGrids returns HTTP 200 with `mean: null` for urban / water pixels.
    A successful response with no usable data must NOT inflate status to 'ok'.
    """
    null_response = {
        "properties": {
            "layers": [
                {"name": p, "depths": [
                    {"label": "0-5cm",  "values": {"mean": None}},
                    {"label": "5-15cm", "values": {"mean": None}},
                    {"label": "15-30cm","values": {"mean": None}},
                ]}
                for p in ("phh2o", "cec", "clay", "sand", "silt", "soc", "bdod", "cfvo")
            ]
        }
    }
    raw_soil = {
        "status": "ok",  # HTTP succeeded
        "points": [{"response": null_response} for _ in range(9)],
    }
    out = compute_soil_features(raw_soil, _dem_cache_from_fixture())
    assert out["status"] == "partial"
    assert "soil=no_data" in (out["error"] or "")
    assert out["ph_h2o"] is None
    assert out["elevation_m"] is not None  # DEM is fine


# ---------------------------------------------------------------------------
# fetch_soilgrids — cache + retry
# ---------------------------------------------------------------------------


def test_fetch_soilgrids_writes_cache_and_resumes(tmp_data_dir: Path) -> None:
    calls: list[tuple[float, float]] = []
    canned = json.loads((FIXTURES_DIR / "soilgrids_tuscany.json").read_text())

    def fake(lat: float, lon: float) -> dict:
        calls.append((lat, lon))
        return canned

    cache1 = fetch_soilgrids("Tuscany", "Italy", 43.46, 11.04,
                             fetch_fn=fake, sleep_fn=lambda _s: None)
    assert cache1["status"] == "ok"
    assert len(calls) == 9  # one per 3x3 buffer point
    cache_path = get_settings().soil_raw_dir / "italy_tuscany.json"
    assert cache_path.exists()

    calls.clear()
    cache2 = fetch_soilgrids("Tuscany", "Italy", 43.46, 11.04,
                             fetch_fn=fake, sleep_fn=lambda _s: None)
    assert calls == []  # served from cache
    assert cache2["status"] == "ok"


def test_fetch_soilgrids_partial_when_some_points_fail(tmp_data_dir: Path) -> None:
    """3 of 9 buffer points 5xx exhaustively → status='partial', 6 succeed."""
    canned = json.loads((FIXTURES_DIR / "soilgrids_tuscany.json").read_text())
    # Fail by coordinate so retries don't accidentally recover. Pick three
    # arbitrary buffer indices: top-middle, centre, bottom-middle.
    grid = _grid_3x3(0.0, 0.0, SOILGRIDS_BUFFER_DEG)
    bad_coords = {grid[1], grid[4], grid[7]}

    def fake(lat: float, lon: float) -> dict:
        if (lat, lon) in bad_coords:
            raise RuntimeError("simulated 5xx")
        return canned

    cache = fetch_soilgrids("X", "Y", 0.0, 0.0,
                            fetch_fn=fake, sleep_fn=lambda _s: None)
    assert cache["status"] == "partial"
    n_ok = sum(1 for p in cache["points"] if p["response"] is not None)
    assert n_ok == 6


def test_fetch_soilgrids_status_error_when_all_fail(tmp_data_dir: Path) -> None:
    def always_fail(lat: float, lon: float) -> dict:
        raise RuntimeError("dead")

    cache = fetch_soilgrids("X", "Y", 0.0, 0.0,
                            fetch_fn=always_fail, sleep_fn=lambda _s: None)
    assert cache["status"] == "error"
    assert all(p["response"] is None for p in cache["points"])


def test_fetch_soilgrids_force_refetches(tmp_data_dir: Path) -> None:
    canned = json.loads((FIXTURES_DIR / "soilgrids_tuscany.json").read_text())
    calls: list[int] = []

    def fake(lat: float, lon: float) -> dict:
        calls.append(1)
        return canned

    fetch_soilgrids("Tuscany", "Italy", 43.46, 11.04,
                    fetch_fn=fake, sleep_fn=lambda _s: None)
    n_first = len(calls)
    fetch_soilgrids("Tuscany", "Italy", 43.46, 11.04,
                    fetch_fn=fake, sleep_fn=lambda _s: None, force=True)
    assert len(calls) == 2 * n_first


# ---------------------------------------------------------------------------
# fetch_elevation_slope — cache
# ---------------------------------------------------------------------------


def test_fetch_dem_writes_cache_and_resumes(tmp_data_dir: Path) -> None:
    canned = json.loads((FIXTURES_DIR / "open_elevation_tuscany.json").read_text())
    calls: list[int] = []

    def fake(points: list) -> dict:
        calls.append(len(points))
        return canned

    cache1 = fetch_elevation_slope("Tuscany", "Italy", 43.46, 11.04,
                                   fetch_fn=fake, sleep_fn=lambda _s: None)
    assert cache1["status"] == "ok"
    assert calls == [9]  # 3x3 grid, one POST

    cache2 = fetch_elevation_slope("Tuscany", "Italy", 43.46, 11.04,
                                   fetch_fn=fake, sleep_fn=lambda _s: None)
    assert calls == [9]  # second call is cache hit, no new POST
    assert cache2["lat"] == 43.46


def test_fetch_dem_error_when_all_retries_fail(tmp_data_dir: Path) -> None:
    def boom(_points: list) -> dict:
        raise RuntimeError("dead")

    cache = fetch_elevation_slope("X", "Y", 0.0, 0.0,
                                  fetch_fn=boom, sleep_fn=lambda _s: None)
    assert cache["status"] == "error"
    assert cache["response"] is None
    assert "dead" in cache["error"]


# ---------------------------------------------------------------------------
# build_soil_table — orchestrator
# ---------------------------------------------------------------------------


def test_build_soil_table_writes_features(tmp_data_dir: Path) -> None:
    _write_fake_geocode(
        get_settings().geocode_parquet,
        [{"region": "Tuscany", "country": "Italy", "lat": 43.46, "lon": 11.04, "status": "ok"}],
    )
    canned_soil = json.loads((FIXTURES_DIR / "soilgrids_tuscany.json").read_text())
    canned_dem = json.loads((FIXTURES_DIR / "open_elevation_tuscany.json").read_text())

    path = build_soil_table(
        soil_fetch_fn=lambda lat, lon: canned_soil,
        dem_fetch_fn=lambda points: canned_dem,
        sleep_fn=lambda _s: None,
    )
    df = pl.read_parquet(path)
    assert df.shape == (1, len(SOIL_SCHEMA))
    row = df.row(0, named=True)
    assert row["region"] == "Tuscany"
    assert row["calcareous"] is True
    assert row["drainage_class"] == "chalky"
    assert row["status"] == "ok"
    assert row["ph_h2o"] == pytest.approx(7.78, abs=0.01)
    assert row["elevation_m"] == pytest.approx(269.0, abs=1e-6)
    assert row["slope_deg"] == pytest.approx(4.58, abs=0.01)


def test_build_soil_table_skips_failed_geocodes(tmp_data_dir: Path) -> None:
    """Rows where geocode status != 'ok' must not be queried (no coordinates)."""
    _write_fake_geocode(
        get_settings().geocode_parquet,
        [
            {"region": "Tuscany", "country": "Italy", "lat": 43.46, "lon": 11.04,
             "status": "ok"},
            {"region": "Atlantis", "country": "Nowhere", "lat": 0.0, "lon": 0.0,
             "status": "not_found"},
        ],
    )
    canned_soil = json.loads((FIXTURES_DIR / "soilgrids_tuscany.json").read_text())
    canned_dem = json.loads((FIXTURES_DIR / "open_elevation_tuscany.json").read_text())

    soil_calls: list[tuple[float, float]] = []
    dem_calls: list[int] = []

    def fake_soil(lat: float, lon: float) -> dict:
        soil_calls.append((lat, lon))
        return canned_soil

    def fake_dem(points: list) -> dict:
        dem_calls.append(len(points))
        return canned_dem

    path = build_soil_table(
        soil_fetch_fn=fake_soil, dem_fetch_fn=fake_dem, sleep_fn=lambda _s: None
    )
    df = pl.read_parquet(path)
    # Only Tuscany processed.
    assert df.shape[0] == 1
    assert df["region"].to_list() == ["Tuscany"]
    # 9 soil calls (one per buffer point), 1 DEM call. All for Tuscany.
    assert len(soil_calls) == 9
    assert dem_calls == [9]


def test_build_soil_table_resume(tmp_data_dir: Path) -> None:
    _write_fake_geocode(
        get_settings().geocode_parquet,
        [
            {"region": "Tuscany", "country": "Italy", "lat": 43.46, "lon": 11.04, "status": "ok"},
            {"region": "Bordeaux", "country": "France", "lat": 44.84, "lon": -0.58, "status": "ok"},
        ],
    )
    canned_soil = json.loads((FIXTURES_DIR / "soilgrids_tuscany.json").read_text())
    canned_dem = json.loads((FIXTURES_DIR / "open_elevation_tuscany.json").read_text())

    # First run: limit to one row.
    build_soil_table(
        limit=1,
        soil_fetch_fn=lambda lat, lon: canned_soil,
        dem_fetch_fn=lambda points: canned_dem,
        sleep_fn=lambda _s: None,
    )
    df = pl.read_parquet(get_settings().soil_parquet)
    assert df.shape[0] == 1

    # Second run: no limit. Only the missing row should be fetched.
    soil_calls: list[tuple[float, float]] = []
    build_soil_table(
        soil_fetch_fn=lambda lat, lon: (soil_calls.append((lat, lon)), canned_soil)[1],
        dem_fetch_fn=lambda points: canned_dem,
        sleep_fn=lambda _s: None,
    )
    df = pl.read_parquet(get_settings().soil_parquet)
    assert df.shape[0] == 2
    # 9 calls for the one missing region (Bordeaux's buffer).
    assert len(soil_calls) == 9


def test_build_soil_table_atomic_no_tmp_left(tmp_data_dir: Path) -> None:
    _write_fake_geocode(
        get_settings().geocode_parquet,
        [{"region": "Tuscany", "country": "Italy", "lat": 43.46, "lon": 11.04, "status": "ok"}],
    )
    canned_soil = json.loads((FIXTURES_DIR / "soilgrids_tuscany.json").read_text())
    canned_dem = json.loads((FIXTURES_DIR / "open_elevation_tuscany.json").read_text())

    build_soil_table(
        soil_fetch_fn=lambda lat, lon: canned_soil,
        dem_fetch_fn=lambda points: canned_dem,
        sleep_fn=lambda _s: None,
    )
    parquet = get_settings().soil_parquet
    assert parquet.exists()
    assert not parquet.with_suffix(parquet.suffix + ".tmp").exists()


def test_scan_soil_errors_when_missing(tmp_data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Soil parquet not found"):
        scan_soil()
