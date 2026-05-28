"""Tests for the climate feature module (NASA POWER Daily).

All tests are offline. The stub `fetch_fn` writes a hand-built NASA POWER
JSON response to the target path, exercising the full fetch→load→compute
pipeline without ever hitting the network. Pure-function tests for the
feature math operate on locally-built polars frames.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from vininator.config import (
    CLIMATE_YEAR_RANGE,
    CLIMATOLOGY_WINDOW,
    NASA_POWER_BACKOFF_SEC,
    NASA_POWER_COMMUNITY,
    NASA_POWER_DAILY_VARS,
    NASA_POWER_FILL_VALUE,
    get_settings,
)
from vininator.data.geocode import GEOCODE_SCHEMA
from vininator.features.climate import (
    CLIMATE_SCHEMA,
    NasaPowerError,
    _growing_season_window,
    build_climate_table,
    compute_climate_features,
    compute_climatology,
    fetch_nasa_power_region,
    load_nasa_power_daily,
)
from vininator.features.soil import region_slug

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _daily_frame_for_year(
    year: int,
    tmin_c: float = 12.0,
    tmean_c: float = 18.0,
    tmax_c: float = 24.0,
    precip_mm: float = 1.5,
    ssrd_mj: float = 20.0,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """Uniform daily frame covering `[start, end]` (default: full year)."""
    s = start or date(year, 1, 1)
    e = end or date(year, 12, 31)
    n_days = (e - s).days + 1
    dates = [s + timedelta(days=i) for i in range(n_days)]
    return pl.DataFrame(
        {
            "date": dates,
            "tmin_c": [tmin_c] * n_days,
            "tmean_c": [tmean_c] * n_days,
            "tmax_c": [tmax_c] * n_days,
            "precip_mm": [precip_mm] * n_days,
            "ssrd_mj": [ssrd_mj] * n_days,
        }
    )


def _make_nasa_power_payload(
    year_range: tuple[int, int],
    *,
    tmin_c: float = 12.0,
    tmean_c: float = 18.0,
    tmax_c: float = 24.0,
    precip_mm: float = 1.5,
    ssrd_mj: float = 20.0,
    per_year_tmean: dict[int, float] | None = None,
    lat: float = 45.0,
    lon: float = 3.0,
) -> dict:
    """Build a NASA POWER-shaped JSON payload covering `year_range` inclusive.

    `per_year_tmean` lets a test pin a specific tmean for individual years
    (e.g. mark 2003 as a heat-wave) while keeping the rest uniform.
    """
    start_y, end_y = year_range
    start_date = date(start_y, 1, 1)
    end_date = date(end_y, 12, 31)
    n_days = (end_date - start_date).days + 1

    t2m: dict[str, float] = {}
    t2m_min: dict[str, float] = {}
    t2m_max: dict[str, float] = {}
    prect: dict[str, float] = {}
    sw: dict[str, float] = {}
    for i in range(n_days):
        d = start_date + timedelta(days=i)
        key = d.strftime("%Y%m%d")
        t2m[key] = (
            per_year_tmean.get(d.year, tmean_c) if per_year_tmean else tmean_c
        )
        t2m_min[key] = tmin_c
        t2m_max[key] = tmax_c
        prect[key] = precip_mm
        sw[key] = ssrd_mj

    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat, 0.0]},
        "properties": {
            "parameter": {
                "T2M": t2m,
                "T2M_MIN": t2m_min,
                "T2M_MAX": t2m_max,
                "PRECTOTCORR": prect,
                "ALLSKY_SFC_SW_DWN": sw,
            }
        },
        "header": {"fill_value": NASA_POWER_FILL_VALUE},
        "parameters": {
            "T2M": {"units": "C"},
            "T2M_MIN": {"units": "C"},
            "T2M_MAX": {"units": "C"},
            "PRECTOTCORR": {"units": "mm/day"},
            "ALLSKY_SFC_SW_DWN": {"units": "MJ/m^2/day"},
        },
    }


def _stub_nasa_power_fetch(params: dict, target: Path) -> None:
    """Default stub: write a uniform NASA POWER payload covering the
    requested date range. Every region produces identical features, so
    build-table tests reduce to row presence + schema checks.
    """
    start_year = int(params["start"][:4])
    end_year = int(params["end"][:4])
    payload = _make_nasa_power_payload(
        (start_year, end_year),
        lat=float(params["latitude"]),
        lon=float(params["longitude"]),
    )
    target.write_text(json.dumps(payload), encoding="utf-8")


def _write_fake_geocode(target: Path, rows: list[dict]) -> None:
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
# Growing-season window
# ---------------------------------------------------------------------------


def test_growing_season_window_nh() -> None:
    """Northern hemisphere → April 1 – October 31 of vintage year."""
    start, end = _growing_season_window(lat=45.0, vintage_year=2018)
    assert start == date(2018, 4, 1)
    assert end == date(2018, 10, 31)


def test_growing_season_window_sh() -> None:
    """Southern hemisphere → Oct 1 of (y-1) – April 30 of y."""
    start, end = _growing_season_window(lat=-33.0, vintage_year=2018)
    assert start == date(2017, 10, 1)
    assert end == date(2018, 4, 30)


# ---------------------------------------------------------------------------
# compute_climate_features — pinned arithmetic
# ---------------------------------------------------------------------------


def test_compute_climate_features_pinned() -> None:
    """Uniform-day frame → exact GDD, precip, frost / heat day counts."""
    daily = _daily_frame_for_year(
        2018, tmin_c=12.0, tmean_c=18.0, tmax_c=24.0, precip_mm=2.0, ssrd_mj=20.0
    )
    features = compute_climate_features(daily, lat=45.0, vintage_year=2018)

    expected_days = 214  # Apr–Oct
    assert features["gdd_10c"] == pytest.approx(8.0 * expected_days)
    assert features["precip_total_mm"] == pytest.approx(2.0 * expected_days)
    assert features["precip_harvest_mm"] == pytest.approx(2.0 * 30)
    assert features["heat_spike_days"] == 0
    assert features["frost_days_spring"] == 0
    assert features["diurnal_range_mean"] == pytest.approx(12.0)
    assert features["solar_total_mj"] == pytest.approx(20.0 * expected_days)
    assert features["is_partial"] is False


def test_compute_climate_features_counts_heat_and_frost() -> None:
    """Hand-tuned per-day values to exercise heat-spike + spring-frost counting."""
    daily = _daily_frame_for_year(2018, tmin_c=10.0, tmax_c=20.0)
    rows = daily.with_columns(
        tmax_c=pl.when(
            (pl.col("date").dt.month() == 7) & (pl.col("date").dt.day().is_in([1, 2, 3, 4, 5]))
        ).then(40.0).otherwise(pl.col("tmax_c")),
        tmin_c=pl.when(
            (pl.col("date").dt.month() == 4) & (pl.col("date").dt.day().is_in([10, 11, 12]))
        ).then(-2.0).when(
            (pl.col("date").dt.month() == 2) & (pl.col("date").dt.day().is_in([1, 2]))
        ).then(-5.0).otherwise(pl.col("tmin_c")),
    )

    features = compute_climate_features(rows, lat=45.0, vintage_year=2018)
    assert features["heat_spike_days"] == 5
    assert features["frost_days_spring"] == 3  # Feb days excluded by frost window


def test_compute_climate_features_is_partial_on_missing_days() -> None:
    """A 10-day gap mid-season → is_partial=True."""
    daily = _daily_frame_for_year(2018)
    keep_mask = ~(
        (pl.col("date").dt.month() == 7) & (pl.col("date").dt.day().is_between(10, 19))
    )
    daily = daily.filter(keep_mask)
    features = compute_climate_features(daily, lat=45.0, vintage_year=2018)
    assert features["is_partial"] is True


def test_compute_climate_features_is_partial_on_null_tmean() -> None:
    """Null tmean inside the season → is_partial=True even with full date coverage.

    NASA POWER reports -999.0 for days where source data was unavailable.
    `load_nasa_power_daily` coerces that sentinel to null; this test confirms
    the downstream null-detection works the same as before.
    """
    daily = _daily_frame_for_year(2018)
    daily = daily.with_columns(
        tmean_c=pl.when(
            (pl.col("date").dt.month() == 7) & (pl.col("date").dt.day() == 15)
        ).then(None).otherwise(pl.col("tmean_c")),
    )
    features = compute_climate_features(daily, lat=45.0, vintage_year=2018)
    assert features["is_partial"] is True


def test_compute_climate_features_empty_window_returns_nulls() -> None:
    """Daily frame outside the season window → all features null, is_partial=True."""
    daily = _daily_frame_for_year(
        2018, start=date(2018, 1, 1), end=date(2018, 1, 31),
    )
    features = compute_climate_features(daily, lat=45.0, vintage_year=2018)
    assert features["gdd_10c"] is None
    assert features["is_partial"] is True


# ---------------------------------------------------------------------------
# compute_climatology
# ---------------------------------------------------------------------------


def test_compute_climatology_skips_partial_years() -> None:
    """Window has 28 years (1991–2018). Make one year partial → climatology
    still computes from the remaining 27 complete years."""
    start_y, end_y = CLIMATOLOGY_WINDOW
    daily_by_year: dict[int, pl.DataFrame] = {}
    for year in range(start_y, end_y + 1):
        if year == 2000:
            full = _daily_frame_for_year(year, tmean_c=18.0, precip_mm=2.0)
            daily_by_year[year] = full.filter(
                ~((pl.col("date").dt.month() == 10) & (pl.col("date").dt.day() > 1))
            )
        else:
            daily_by_year[year] = _daily_frame_for_year(year, tmean_c=18.0, precip_mm=2.0)

    clim = compute_climatology(daily_by_year, lat=45.0, window=CLIMATOLOGY_WINDOW)
    assert clim["gdd_10c"] == pytest.approx(1712.0)
    assert clim["precip_total_mm"] == pytest.approx(2.0 * 214)


def test_compute_climatology_empty_when_too_few_complete_years() -> None:
    """Fewer than CLIMATOLOGY_MIN_YEARS complete years → empty climatology."""
    daily_by_year = {1991: _daily_frame_for_year(1991), 1992: _daily_frame_for_year(1992)}
    clim = compute_climatology(daily_by_year, lat=45.0, window=CLIMATOLOGY_WINDOW)
    assert clim == {}


def test_anomaly_is_absolute_minus_climatology_mean(tmp_data_dir: Path) -> None:
    """End-to-end anomaly arithmetic via build_climate_table."""
    settings = get_settings()
    _write_fake_geocode(
        settings.geocode_parquet,
        [{"region": "TestRegion", "country": "Testland", "lat": 45.0, "lon": 3.0,
          "result_type": "administrative"}],
    )

    def hot_2018_fetch(params: dict, target: Path) -> None:
        """Most years tmean=18°C; 2018 is hot at tmean=20°C."""
        payload = _make_nasa_power_payload(
            CLIMATE_YEAR_RANGE, per_year_tmean={2018: 20.0},
            lat=float(params["latitude"]), lon=float(params["longitude"]),
        )
        target.write_text(json.dumps(payload), encoding="utf-8")

    build_climate_table(fetch_fn=hot_2018_fetch)

    df = pl.read_parquet(settings.climate_parquet)
    row_2018 = df.filter(pl.col("vintage_year") == 2018).row(0, named=True)
    row_2010 = df.filter(pl.col("vintage_year") == 2010).row(0, named=True)

    # Climatology mean GDD across 1991–2018 = average of 27×(8×214) + 1×(10×214).
    expected_mean_gdd = (27 * 8.0 * 214 + 1 * 10.0 * 214) / 28
    assert row_2018["gdd_10c"] == pytest.approx(10.0 * 214)
    assert row_2018["gdd_10c_anom"] == pytest.approx(10.0 * 214 - expected_mean_gdd)
    assert row_2010["gdd_10c"] == pytest.approx(8.0 * 214)
    assert row_2010["gdd_10c_anom"] == pytest.approx(8.0 * 214 - expected_mean_gdd)


# ---------------------------------------------------------------------------
# load_nasa_power_daily
# ---------------------------------------------------------------------------


def test_load_nasa_power_daily_parses_payload(tmp_path: Path) -> None:
    """JSON shape from NASA POWER → polars frame with the right columns + dtypes."""
    payload = _make_nasa_power_payload(
        (2020, 2021), tmin_c=8.0, tmean_c=15.0, tmax_c=22.0,
        precip_mm=1.0, ssrd_mj=18.0,
    )
    json_path = tmp_path / "region.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    daily = load_nasa_power_daily(json_path)
    assert daily.columns == ["date", "tmin_c", "tmean_c", "tmax_c", "precip_mm", "ssrd_mj"]
    assert daily["date"].dtype == pl.Date
    # 366 (leap) + 365 = 731 days
    assert daily.height == 731
    first = daily.row(0, named=True)
    assert first["tmin_c"] == pytest.approx(8.0)
    assert first["tmean_c"] == pytest.approx(15.0)
    assert first["tmax_c"] == pytest.approx(22.0)
    assert first["precip_mm"] == pytest.approx(1.0)
    assert first["ssrd_mj"] == pytest.approx(18.0)


def test_load_nasa_power_daily_coerces_fill_value_to_null(tmp_path: Path) -> None:
    """POWER's -999.0 fill value lands as null in the polars frame.

    Required so that null-aware downstream code (`is_partial`, climatology
    skip) sees missing days as null rather than as a wildly negative real
    measurement. Otherwise a single missing day in the growing season would
    poison every aggregate that touched it (GDD, mean diurnal, etc.).
    """
    payload = _make_nasa_power_payload((2020, 2020))
    # Knock out 2020-07-15 across all variables with POWER's fill sentinel.
    fill_day = "20200715"
    for var in ("T2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"):
        payload["properties"]["parameter"][var][fill_day] = NASA_POWER_FILL_VALUE

    json_path = tmp_path / "region.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    daily = load_nasa_power_daily(json_path)
    target = daily.filter(pl.col("date") == date(2020, 7, 15)).row(0, named=True)
    for col in ("tmin_c", "tmean_c", "tmax_c", "precip_mm", "ssrd_mj"):
        assert target[col] is None, f"{col} should be null for fill-value day"


def test_load_nasa_power_daily_errors_on_empty_payload(tmp_path: Path) -> None:
    """No properties.parameter.T2M → ValueError, not a silent empty frame."""
    json_path = tmp_path / "empty.json"
    json_path.write_text(json.dumps({"properties": {"parameter": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no properties.parameter.T2M"):
        load_nasa_power_daily(json_path)


# ---------------------------------------------------------------------------
# fetch_nasa_power_region
# ---------------------------------------------------------------------------


def test_fetch_nasa_power_region_cache_hit_skips_network(tmp_data_dir: Path) -> None:
    """A pre-existing non-empty JSON cache → fetch_fn never called."""
    settings = get_settings()
    cache = settings.nasa_power_raw_dir / f"{region_slug('TestRegion', 'Testland')}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{\"properties\": {\"parameter\": {\"T2M\": {}}}}", encoding="utf-8")

    def must_not_call(params: dict, target: Path) -> None:
        raise AssertionError("fetch_fn should not be called on cache hit")

    out = fetch_nasa_power_region(
        "TestRegion", "Testland", 45.0, 3.0, fetch_fn=must_not_call,
    )
    assert out == cache


def test_fetch_nasa_power_region_atomic_write(tmp_data_dir: Path) -> None:
    """After a successful fetch, the final JSON exists and no .tmp lingers."""
    settings = get_settings()
    cache = settings.nasa_power_raw_dir / f"{region_slug('TestRegion', 'Testland')}.json"
    tmp = cache.with_suffix(cache.suffix + ".tmp")

    def fake_fetch(params: dict, target: Path) -> None:
        assert target.name.endswith(".tmp"), f"expected tmp path, got {target}"
        _stub_nasa_power_fetch(params, target)

    out = fetch_nasa_power_region(
        "TestRegion", "Testland", 45.0, 3.0, fetch_fn=fake_fetch,
    )
    assert out == cache
    assert cache.exists()
    assert not tmp.exists()


def test_fetch_nasa_power_region_params_shape(tmp_data_dir: Path) -> None:
    """The params dict matches NASA POWER's daily point endpoint contract."""
    captured: dict = {}

    def capturing_fetch(params: dict, target: Path) -> None:
        captured.update(params)
        _stub_nasa_power_fetch(params, target)

    fetch_nasa_power_region("R", "C", 45.0, 3.0, fetch_fn=capturing_fetch)

    start_year, end_year = CLIMATE_YEAR_RANGE
    assert captured["latitude"] == 45.0
    assert captured["longitude"] == 3.0
    assert captured["start"] == f"{start_year}0101"
    assert captured["end"] == f"{end_year}1231"
    assert captured["community"] == NASA_POWER_COMMUNITY
    assert captured["format"] == "JSON"
    assert tuple(captured["parameters"].split(",")) == NASA_POWER_DAILY_VARS


def test_fetch_nasa_power_region_transient_retries(tmp_data_dir: Path) -> None:
    """A transient network error retries until success and uses backoff schedule."""
    attempts = {"n": 0}

    def flaky_fetch(params: dict, target: Path) -> None:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("network blip")
        _stub_nasa_power_fetch(params, target)

    sleeps: list[float] = []
    fetch_nasa_power_region(
        "TestRegion", "Testland", 45.0, 3.0,
        fetch_fn=flaky_fetch, sleep_fn=sleeps.append,
    )
    assert attempts["n"] == 3
    # Two failures before success → two backoff sleeps, drawn from the schedule.
    assert sleeps == [NASA_POWER_BACKOFF_SEC[0], NASA_POWER_BACKOFF_SEC[1]]


def test_fetch_nasa_power_region_persistent_failure_raises(tmp_data_dir: Path) -> None:
    """If every retry fails, NasaPowerError is raised with the underlying cause."""
    def always_fails(params: dict, target: Path) -> None:
        raise RuntimeError("network down forever")

    with pytest.raises(NasaPowerError, match="after"):
        fetch_nasa_power_region(
            "TestRegion", "Testland", 45.0, 3.0,
            fetch_fn=always_fails, sleep_fn=lambda _: None,
        )


def test_fetch_nasa_power_region_4xx_does_not_retry(tmp_data_dir: Path) -> None:
    """A NasaPowerError from the fetcher (e.g. 4xx bad params) propagates
    immediately without burning the retry budget — retrying a parameter bug
    just wastes the upstream service's time."""
    attempts = {"n": 0}

    def four_oh_four(params: dict, target: Path) -> None:
        attempts["n"] += 1
        raise NasaPowerError("HTTP 422 from POWER: bad parameter name")

    sleeps: list[float] = []
    with pytest.raises(NasaPowerError, match="422"):
        fetch_nasa_power_region(
            "TestRegion", "Testland", 45.0, 3.0,
            fetch_fn=four_oh_four, sleep_fn=sleeps.append,
        )
    assert attempts["n"] == 1
    assert sleeps == []


# ---------------------------------------------------------------------------
# build_climate_table
# ---------------------------------------------------------------------------


def test_build_climate_table_filters_via_geocode_whitelist(tmp_data_dir: Path) -> None:
    """Blacklisted result_types (e.g. bus_stop) and non-ok status produce no rows."""
    settings = get_settings()
    _write_fake_geocode(
        settings.geocode_parquet,
        [
            {"region": "GoodA", "country": "X", "lat": 45.0, "lon": 3.0,
             "result_type": "administrative"},
            {"region": "GoodB", "country": "X", "lat": 45.0, "lon": 3.0,
             "result_type": None},
            {"region": "JunkBus", "country": "X", "lat": 45.0, "lon": 3.0,
             "result_type": "bus_stop"},
            {"region": "FailedGeo", "country": "X", "lat": 0.0, "lon": 0.0,
             "result_type": "administrative", "status": "error", "error": "not found"},
        ],
    )

    build_climate_table(fetch_fn=_stub_nasa_power_fetch)

    df = pl.read_parquet(settings.climate_parquet)
    regions = set(df["region"].unique().to_list())
    assert regions == {"GoodA", "GoodB"}
    n_years = CLIMATE_YEAR_RANGE[1] - CLIMATE_YEAR_RANGE[0] + 1
    assert df.height == 2 * n_years


def test_build_climate_table_schema(tmp_data_dir: Path) -> None:
    """Schema of the written parquet matches CLIMATE_SCHEMA exactly."""
    settings = get_settings()
    _write_fake_geocode(
        settings.geocode_parquet,
        [{"region": "GoodA", "country": "X", "lat": 45.0, "lon": 3.0,
          "result_type": "administrative"}],
    )
    build_climate_table(fetch_fn=_stub_nasa_power_fetch)

    df = pl.read_parquet(settings.climate_parquet)
    assert dict(df.schema) == CLIMATE_SCHEMA


def test_build_climate_table_resumes_existing_rows(tmp_data_dir: Path) -> None:
    """A region already in climate.parquet is skipped on the next run."""
    settings = get_settings()
    _write_fake_geocode(
        settings.geocode_parquet,
        [{"region": "GoodA", "country": "X", "lat": 45.0, "lon": 3.0,
          "result_type": "administrative"}],
    )
    build_climate_table(fetch_fn=_stub_nasa_power_fetch)

    def must_not_call(params: dict, target: Path) -> None:
        raise AssertionError("resume failed — fetch_fn called for existing row")

    build_climate_table(fetch_fn=must_not_call)
