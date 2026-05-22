"""Tests for the climate feature module (Open-Meteo Historical).

All tests are offline. The stub `fetch_fn` writes a hand-built Open-Meteo
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
    OPEN_METEO_DAILY_VARS,
    get_settings,
)
from vininator.data.geocode import GEOCODE_SCHEMA
from vininator.features.climate import (
    CLIMATE_SCHEMA,
    OpenMeteoError,
    _growing_season_window,
    build_climate_table,
    compute_climate_features,
    compute_climatology,
    fetch_open_meteo_region,
    load_open_meteo_daily,
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


def _make_open_meteo_payload(
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
    """Build an Open-Meteo-shaped JSON payload covering `year_range` inclusive.

    `per_year_tmean` lets a test pin a specific tmean for individual years
    (e.g. mark 2003 as a heat-wave) while keeping the rest uniform.
    """
    start_y, end_y = year_range
    start_date = date(start_y, 1, 1)
    end_date = date(end_y, 12, 31)
    n_days = (end_date - start_date).days + 1
    dates = [(start_date + timedelta(days=i)).isoformat() for i in range(n_days)]
    tmeans = []
    for i in range(n_days):
        d = start_date + timedelta(days=i)
        tmeans.append(
            per_year_tmean.get(d.year, tmean_c) if per_year_tmean else tmean_c
        )
    return {
        "latitude": lat,
        "longitude": lon,
        "daily": {
            "time": dates,
            "temperature_2m_min": [tmin_c] * n_days,
            "temperature_2m_mean": tmeans,
            "temperature_2m_max": [tmax_c] * n_days,
            "precipitation_sum": [precip_mm] * n_days,
            "shortwave_radiation_sum": [ssrd_mj] * n_days,
        },
    }


def _stub_open_meteo_fetch(params: dict, target: Path) -> None:
    """Default stub: write a uniform Open-Meteo payload covering the
    requested date range. Every region produces identical features, so
    build-table tests reduce to row presence + schema checks.
    """
    start_year = int(params["start_date"].split("-", 1)[0])
    end_year = int(params["end_date"].split("-", 1)[0])
    payload = _make_open_meteo_payload(
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

    Open-Meteo backfills the most recent years over weeks and may return
    explicit nulls for not-yet-published days. We must catch that.
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
        payload = _make_open_meteo_payload(
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
# load_open_meteo_daily
# ---------------------------------------------------------------------------


def test_load_open_meteo_daily_parses_payload(tmp_path: Path) -> None:
    """JSON shape from Open-Meteo → polars frame with the right columns + dtypes."""
    payload = _make_open_meteo_payload(
        (2020, 2021), tmin_c=8.0, tmean_c=15.0, tmax_c=22.0,
        precip_mm=1.0, ssrd_mj=18.0,
    )
    json_path = tmp_path / "region.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    daily = load_open_meteo_daily(json_path)
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


def test_load_open_meteo_daily_errors_on_empty_payload(tmp_path: Path) -> None:
    """No daily.time array → ValueError, not a silent empty frame."""
    json_path = tmp_path / "empty.json"
    json_path.write_text(json.dumps({"latitude": 0, "longitude": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="no daily.time"):
        load_open_meteo_daily(json_path)


# ---------------------------------------------------------------------------
# fetch_open_meteo_region
# ---------------------------------------------------------------------------


def test_fetch_open_meteo_region_cache_hit_skips_network(tmp_data_dir: Path) -> None:
    """A pre-existing non-empty JSON cache → fetch_fn never called."""
    settings = get_settings()
    cache = settings.open_meteo_raw_dir / f"{region_slug('TestRegion', 'Testland')}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{\"daily\": {\"time\": []}}", encoding="utf-8")

    def must_not_call(params: dict, target: Path) -> None:
        raise AssertionError("fetch_fn should not be called on cache hit")

    out = fetch_open_meteo_region(
        "TestRegion", "Testland", 45.0, 3.0, fetch_fn=must_not_call,
    )
    assert out == cache


def test_fetch_open_meteo_region_atomic_write(tmp_data_dir: Path) -> None:
    """After a successful fetch, the final JSON exists and no .tmp lingers."""
    settings = get_settings()
    cache = settings.open_meteo_raw_dir / f"{region_slug('TestRegion', 'Testland')}.json"
    tmp = cache.with_suffix(cache.suffix + ".tmp")

    def fake_fetch(params: dict, target: Path) -> None:
        assert target.name.endswith(".tmp"), f"expected tmp path, got {target}"
        _stub_open_meteo_fetch(params, target)

    out = fetch_open_meteo_region(
        "TestRegion", "Testland", 45.0, 3.0, fetch_fn=fake_fetch,
    )
    assert out == cache
    assert cache.exists()
    assert not tmp.exists()


def test_fetch_open_meteo_region_params_shape(tmp_data_dir: Path) -> None:
    """The params dict matches Open-Meteo's archive endpoint contract."""
    captured: dict = {}

    def capturing_fetch(params: dict, target: Path) -> None:
        captured.update(params)
        _stub_open_meteo_fetch(params, target)

    fetch_open_meteo_region("R", "C", 45.0, 3.0, fetch_fn=capturing_fetch)

    start_year, end_year = CLIMATE_YEAR_RANGE
    assert captured["latitude"] == 45.0
    assert captured["longitude"] == 3.0
    assert captured["start_date"] == f"{start_year}-01-01"
    assert captured["end_date"] == f"{end_year}-12-31"
    assert captured["timezone"] == "UTC"
    assert tuple(captured["daily"].split(",")) == OPEN_METEO_DAILY_VARS


def test_fetch_open_meteo_region_transient_retries(tmp_data_dir: Path) -> None:
    """A transient 5xx retries until success."""
    attempts = {"n": 0}

    def flaky_fetch(params: dict, target: Path) -> None:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("HTTP 503 Service Unavailable")
        _stub_open_meteo_fetch(params, target)

    sleeps: list[float] = []
    fetch_open_meteo_region(
        "TestRegion", "Testland", 45.0, 3.0,
        fetch_fn=flaky_fetch, sleep_fn=sleeps.append,
    )
    assert attempts["n"] == 3
    assert len(sleeps) == 2


def test_fetch_open_meteo_region_honors_retry_after(tmp_data_dir: Path) -> None:
    """A 429 with `Retry-After: 7` makes the loop sleep exactly 7s, not the
    default backoff (2s)."""
    from vininator.features.climate import _RateLimited

    attempts = {"n": 0}

    def rate_limited_then_ok(params: dict, target: Path) -> None:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _RateLimited(retry_after_sec=7.0)
        _stub_open_meteo_fetch(params, target)

    sleeps: list[float] = []
    fetch_open_meteo_region(
        "TestRegion", "Testland", 45.0, 3.0,
        fetch_fn=rate_limited_then_ok, sleep_fn=sleeps.append,
    )
    assert attempts["n"] == 2
    assert sleeps == [7.0], "should have honored Retry-After, not used the 2s fallback"


def test_fetch_open_meteo_region_429_without_retry_after_uses_fallback(
    tmp_data_dir: Path,
) -> None:
    """A 429 with no Retry-After header falls back to OPEN_METEO_BACKOFF_SEC."""
    from vininator.features.climate import _RateLimited

    attempts = {"n": 0}

    def rate_limited_then_ok(params: dict, target: Path) -> None:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _RateLimited(retry_after_sec=None)
        _stub_open_meteo_fetch(params, target)

    sleeps: list[float] = []
    fetch_open_meteo_region(
        "TestRegion", "Testland", 45.0, 3.0,
        fetch_fn=rate_limited_then_ok, sleep_fn=sleeps.append,
    )
    assert sleeps == [2.0]  # first entry of OPEN_METEO_BACKOFF_SEC


def test_fetch_open_meteo_region_persistent_failure_raises(tmp_data_dir: Path) -> None:
    """If every retry fails, OpenMeteoError is raised with the underlying cause."""
    def always_fails(params: dict, target: Path) -> None:
        raise RuntimeError("HTTP 503 forever")

    with pytest.raises(OpenMeteoError, match="after"):
        fetch_open_meteo_region(
            "TestRegion", "Testland", 45.0, 3.0,
            fetch_fn=always_fails, sleep_fn=lambda _: None,
        )


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

    build_climate_table(fetch_fn=_stub_open_meteo_fetch)

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
    build_climate_table(fetch_fn=_stub_open_meteo_fetch)

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
    build_climate_table(fetch_fn=_stub_open_meteo_fetch)

    def must_not_call(params: dict, target: Path) -> None:
        raise AssertionError("resume failed — fetch_fn called for existing row")

    build_climate_table(fetch_fn=must_not_call)
