"""Per-region climate features for the terroir block (Open-Meteo Historical).

Phase 2 step 3. We pull the Open-Meteo Historical Weather API — a free,
JSON-only wrapper around ERA5-Land reanalysis (0.1° / ~11 km) — for every
usable region, fetching all of `CLIMATE_YEAR_RANGE` and all five daily
variables in a single HTTP request per region. The response lands as JSON
under `data/raw/open_meteo/{slug}.json`; that file *is* the resume state —
restart picks up wherever the disk says it left off.

Separation of concerns mirrors soil.py:

- `fetch_open_meteo_region`: I/O, cached, network-aware. Atomic tmp→rename.
- `load_open_meteo_daily`: pure (no network), JSON → daily polars DataFrame.
- `compute_climate_features`: pure, applies growing-season mask + derives the
  7 absolute features. Marks `is_partial=True` if growing-season days are
  missing.
- `compute_climatology`: pure, baseline mean per feature across complete
  years inside `CLIMATOLOGY_WINDOW`. Skips partial-season years.
- `build_climate_table`: orchestrator. Iterates `filter_to_usable` regions,
  fetches one JSON per region (cache-first), computes features + anomalies
  for each vintage_year in `CLIMATE_YEAR_RANGE`, writes
  `data/interim/climate.parquet`.

The same pure functions are reused at inference time by Phase 7's
`TerroirProvider`, so any change to the feature math must hold for batch AND
online — never compute features inline at serve time.

Why Open-Meteo over Copernicus CDS? Same underlying ERA5-Land data, but just
1,377 HTTP specific requests instead of 8.3k–128k chunked CDS calls that need to be aggregated. Attribution is
embedded in the parquet metadata.
"""

from __future__ import annotations

import calendar
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import requests

from vininator.config import (
    CLIMATE_ABSOLUTE_FEATURES,
    CLIMATE_YEAR_RANGE,
    CLIMATOLOGY_MIN_YEARS,
    CLIMATOLOGY_WINDOW,
    FROST_TMIN_C,
    GDD_BASE_TEMP_C,
    GROWING_SEASON_NH,
    GROWING_SEASON_SH_PREV_YEAR,
    GROWING_SEASON_SH_VINTAGE,
    HARVEST_WINDOW_DAYS,
    HEAT_SPIKE_TMAX_C,
    OPEN_METEO_ATTRIBUTION,
    OPEN_METEO_BACKOFF_SEC,
    OPEN_METEO_BASE_URL,
    OPEN_METEO_DAILY_VARS,
    OPEN_METEO_RATE_LIMIT_SEC,
    OPEN_METEO_TIMEOUT_SEC,
    SPRING_FROST_MONTHS_NH,
    SPRING_FROST_MONTHS_SH,
    get_settings,
)
from vininator.data.geocode import filter_to_usable, scan_geocode
from vininator.features.soil import region_slug

# Production fetcher takes (params dict, target path) and writes the raw JSON
# response. Tests inject a stub that synthesises a response dict and writes it.
OpenMeteoFetchFn = Callable[[dict[str, Any], Path], None]
ProgressFn = Callable[[int, int], None]
NotifyFn = Callable[[str], None]
SleepFn = Callable[[float], None]


CLIMATE_SCHEMA: dict[str, pl.DataType] = {
    "region": pl.String(),
    "country": pl.String(),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "vintage_year": pl.Int64(),
    "gdd_10c": pl.Float64(),
    "precip_total_mm": pl.Float64(),
    "precip_harvest_mm": pl.Float64(),
    "heat_spike_days": pl.Int64(),
    "frost_days_spring": pl.Int64(),
    "diurnal_range_mean": pl.Float64(),
    "solar_total_mj": pl.Float64(),
    "gdd_10c_anom": pl.Float64(),
    "precip_total_mm_anom": pl.Float64(),
    "precip_harvest_mm_anom": pl.Float64(),
    "heat_spike_days_anom": pl.Float64(),
    "frost_days_spring_anom": pl.Float64(),
    "diurnal_range_mean_anom": pl.Float64(),
    "solar_total_mj_anom": pl.Float64(),
    "is_partial": pl.Boolean(),
    "status": pl.String(),
    "error": pl.String(),
    "fetched_at": pl.Datetime("us", "UTC"),
}

CLIMATOLOGY_SCHEMA: dict[str, pl.DataType] = {
    "region": pl.String(),
    "country": pl.String(),
    "feature": pl.String(),
    "mean": pl.Float64(),
    "window_start": pl.Int64(),
    "window_end": pl.Int64(),
    "n_years": pl.Int64(),
}


class OpenMeteoError(RuntimeError):
    """Raised when an Open-Meteo request fails permanently (after retries)."""


class _RateLimited(Exception):  # internal, never escapes the retry loop
    """429 from Open-Meteo. Carries the server's `Retry-After` hint if present."""

    def __init__(self, retry_after_sec: float | None) -> None:
        self.retry_after_sec = retry_after_sec
        super().__init__(f"rate limited (Retry-After={retry_after_sec})")


# ---------------------------------------------------------------------------
# Pure helpers — growing season + frost windows
# ---------------------------------------------------------------------------


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _growing_season_window(lat: float, vintage_year: int) -> tuple[date, date]:
    """Growing season for `vintage_year` at latitude `lat`.

    Northern hemisphere (lat >= 0): April 1 – October 31 of vintage_year.
    Southern hemisphere (lat < 0): October 1 of (vintage_year − 1) – April 30
    of vintage_year. Vintage year = harvest year for both hemispheres.
    """
    if lat >= 0:
        start_month, end_month = GROWING_SEASON_NH
        return (
            date(vintage_year, start_month, 1),
            date(vintage_year, end_month, _last_day_of_month(vintage_year, end_month)),
        )
    prev_start_month = GROWING_SEASON_SH_PREV_YEAR[0]
    vintage_end_month = GROWING_SEASON_SH_VINTAGE[1]
    return (
        date(vintage_year - 1, prev_start_month, 1),
        date(
            vintage_year,
            vintage_end_month,
            _last_day_of_month(vintage_year, vintage_end_month),
        ),
    )


def _spring_frost_window(lat: float, vintage_year: int) -> tuple[date, date]:
    """Spring-frost window for `vintage_year` at latitude `lat`.

    NH: April 1 – May 31 of vintage_year.
    SH: October 1 – November 30 of (vintage_year − 1).
    """
    if lat >= 0:
        start_month, end_month = SPRING_FROST_MONTHS_NH
        return (
            date(vintage_year, start_month, 1),
            date(vintage_year, end_month, _last_day_of_month(vintage_year, end_month)),
        )
    start_month, end_month = SPRING_FROST_MONTHS_SH
    return (
        date(vintage_year - 1, start_month, 1),
        date(
            vintage_year - 1,
            end_month,
            _last_day_of_month(vintage_year - 1, end_month),
        ),
    )


# ---------------------------------------------------------------------------
# JSON → daily DataFrame (pure)
# ---------------------------------------------------------------------------


def load_open_meteo_daily(json_path: Path) -> pl.DataFrame:
    """Parse an Open-Meteo JSON cache file into a daily polars frame.

    Output schema: `date, tmin_c, tmean_c, tmax_c, precip_mm, ssrd_mj`.

    Open-Meteo's response shape:
        {
          "latitude": ..., "longitude": ...,
          "daily": {
            "time": ["1991-01-01", ...],
            "temperature_2m_min": [...], "temperature_2m_mean": [...],
            "temperature_2m_max": [...], "precipitation_sum": [...],
            "shortwave_radiation_sum": [...]
          }
        }

    Units already match what `compute_climate_features` expects (°C, mm, MJ/m²).
    Nulls in any array land as nulls in the polars frame — `compute_climate_features`
    will then surface them as `is_partial=True` if they fall in the growing season.

    Pure — no network, no global state.
    """
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    daily = payload.get("daily")
    if not daily or not daily.get("time"):
        raise ValueError(f"Open-Meteo JSON at {json_path} has no daily.time array")

    return (
        pl.DataFrame(
            {
                "date": daily["time"],
                "tmin_c": daily.get("temperature_2m_min", []),
                "tmean_c": daily.get("temperature_2m_mean", []),
                "tmax_c": daily.get("temperature_2m_max", []),
                "precip_mm": daily.get("precipitation_sum", []),
                "ssrd_mj": daily.get("shortwave_radiation_sum", []),
            }
        )
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        .sort("date")
    )


# ---------------------------------------------------------------------------
# Feature computation (pure)
# ---------------------------------------------------------------------------


def _filter_to_window(
    daily: pl.DataFrame, start: date, end: date
) -> pl.DataFrame:
    return daily.filter((pl.col("date") >= start) & (pl.col("date") <= end))


def _is_partial(season: pl.DataFrame, start: date, end: date) -> bool:
    """True if any expected day is missing OR has a null tmean (the GDD anchor)."""
    expected = (end - start).days + 1
    if season.height < expected:
        return True
    return season.select(pl.col("tmean_c").is_null().any()).item()


def compute_climate_features(
    daily: pl.DataFrame, lat: float, vintage_year: int,
) -> dict[str, Any]:
    """Apply the growing-season mask to `daily` and derive the 7 absolute
    features + an `is_partial` flag.

    Pure — used identically in batch training and online inference.
    """
    start, end = _growing_season_window(lat, vintage_year)
    season = _filter_to_window(daily, start, end)

    if season.is_empty():
        return {
            "gdd_10c": None,
            "precip_total_mm": None,
            "precip_harvest_mm": None,
            "heat_spike_days": None,
            "frost_days_spring": None,
            "diurnal_range_mean": None,
            "solar_total_mj": None,
            "is_partial": True,
        }

    harvest_start = end - timedelta(days=HARVEST_WINDOW_DAYS - 1)
    harvest = _filter_to_window(season, harvest_start, end)

    frost_start, frost_end = _spring_frost_window(lat, vintage_year)
    frost_window = _filter_to_window(daily, frost_start, frost_end)

    aggs = season.select(
        (
            pl.when(pl.col("tmean_c") > GDD_BASE_TEMP_C)
            .then(pl.col("tmean_c") - GDD_BASE_TEMP_C)
            .otherwise(0.0)
            .sum()
            .alias("gdd_10c")
        ),
        pl.col("precip_mm").sum().alias("precip_total_mm"),
        (pl.col("tmax_c") > HEAT_SPIKE_TMAX_C).sum().alias("heat_spike_days"),
        (pl.col("tmax_c") - pl.col("tmin_c")).mean().alias("diurnal_range_mean"),
        pl.col("ssrd_mj").sum().alias("solar_total_mj"),
    ).row(0, named=True)

    precip_harvest_mm = (
        float(harvest.select(pl.col("precip_mm").sum()).item())
        if not harvest.is_empty()
        else 0.0
    )
    frost_days_spring = (
        int(frost_window.select((pl.col("tmin_c") < FROST_TMIN_C).sum()).item())
        if not frost_window.is_empty()
        else 0
    )

    return {
        "gdd_10c": float(aggs["gdd_10c"]) if aggs["gdd_10c"] is not None else None,
        "precip_total_mm": (
            float(aggs["precip_total_mm"]) if aggs["precip_total_mm"] is not None else None
        ),
        "precip_harvest_mm": precip_harvest_mm,
        "heat_spike_days": int(aggs["heat_spike_days"]),
        "frost_days_spring": frost_days_spring,
        "diurnal_range_mean": (
            float(aggs["diurnal_range_mean"])
            if aggs["diurnal_range_mean"] is not None
            else None
        ),
        "solar_total_mj": (
            float(aggs["solar_total_mj"]) if aggs["solar_total_mj"] is not None else None
        ),
        "is_partial": _is_partial(season, start, end),
    }


def compute_climatology(
    daily_by_year: dict[int, pl.DataFrame],
    lat: float,
    window: tuple[int, int],
) -> dict[str, float]:
    """Per-feature mean across years inside `window` that have a complete season.

    Partial-season years are skipped. If fewer than `CLIMATOLOGY_MIN_YEARS`
    complete years are available, the climatology is empty (callers will see
    null anomalies for that region).
    """
    start_year, end_year = window
    per_year: dict[str, list[float]] = {f: [] for f in CLIMATE_ABSOLUTE_FEATURES}

    for year in range(start_year, end_year + 1):
        daily = daily_by_year.get(year)
        if daily is None or daily.is_empty():
            continue
        features = compute_climate_features(daily, lat, year)
        if features["is_partial"]:
            continue
        for name in CLIMATE_ABSOLUTE_FEATURES:
            value = features.get(name)
            if value is None:
                continue
            per_year[name].append(float(value))

    n_years = min(len(v) for v in per_year.values()) if per_year else 0
    if n_years < CLIMATOLOGY_MIN_YEARS:
        return {}
    return {name: sum(values) / len(values) for name, values in per_year.items() if values}


# ---------------------------------------------------------------------------
# Fetcher (I/O, cached)
# ---------------------------------------------------------------------------


def _open_meteo_params(lat: float, lon: float) -> dict[str, Any]:
    """Build the query-string dict for one (lat, lon) region request."""
    start_year, end_year = CLIMATE_YEAR_RANGE
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": ",".join(OPEN_METEO_DAILY_VARS),
        "timezone": "UTC",
    }


def fetch_open_meteo_region(
    region: str,
    country: str | None,
    lat: float,
    lon: float,
    *,
    force: bool = False,
    fetch_fn: OpenMeteoFetchFn | None = None,
    notify_fn: NotifyFn | None = None,
    sleep_fn: SleepFn = time.sleep,
) -> Path:
    """Fetch all years for one region from Open-Meteo, cache the JSON, return its path.

    Cache lives at `data/raw/open_meteo/{slug}.json`. On hit (file exists,
    non-empty, `force=False`) the network is never touched — that's the resume
    mechanism. Atomic writes via `tmp → rename` so a Ctrl+C mid-download never
    leaves a half-written `.json` behind.

    Transient failures (network blips, 429, 5xx) retry with exponential
    backoff per `OPEN_METEO_BACKOFF_SEC`. Persistent failures raise
    `OpenMeteoError`.
    """
    slug = region_slug(region, country)
    cache_path = get_settings().open_meteo_raw_dir / f"{slug}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    if fetch_fn is None:
        fetch_fn = _default_open_meteo_fetch_fn

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    params = _open_meteo_params(lat, lon)

    def _notify(msg: str) -> None:
        if notify_fn is not None:
            notify_fn(msg)

    last_exc: Exception | None = None
    for attempt in range(len(OPEN_METEO_BACKOFF_SEC) + 1):
        try:
            _notify(f"... Open-Meteo {region!r} attempt {attempt + 1}")
            fetch_fn(params, tmp_path)
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError(f"Open-Meteo produced empty file at {tmp_path}")
            os.replace(tmp_path, cache_path)
            return cache_path
        except OpenMeteoError:
            _cleanup_tmp(tmp_path)
            raise
        except _RateLimited as exc:
            last_exc = exc
            _cleanup_tmp(tmp_path)
            if attempt < len(OPEN_METEO_BACKOFF_SEC):
                # Prefer the server's hint; fall back to our schedule if absent.
                delay = (
                    exc.retry_after_sec
                    if exc.retry_after_sec is not None
                    else OPEN_METEO_BACKOFF_SEC[attempt]
                )
                _notify(
                    f"... Open-Meteo 429 {region!r}; "
                    f"sleeping {delay:.1f}s "
                    f"({'Retry-After' if exc.retry_after_sec is not None else 'fallback'})"
                )
                sleep_fn(delay)
        except Exception as exc:  # noqa: BLE001 — retry policy is the catch-all
            last_exc = exc
            _cleanup_tmp(tmp_path)
            if attempt < len(OPEN_METEO_BACKOFF_SEC):
                delay = OPEN_METEO_BACKOFF_SEC[attempt]
                _notify(f"... Open-Meteo error {exc!r}; retrying in {delay:.1f}s")
                sleep_fn(delay)
    assert last_exc is not None
    raise OpenMeteoError(
        f"Open-Meteo failed for {region!r} after {len(OPEN_METEO_BACKOFF_SEC) + 1} "
        f"attempts. Last error: {last_exc}"
    ) from last_exc


def _cleanup_tmp(tmp_path: Path) -> None:
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except OSError:
        pass


def _default_open_meteo_fetch_fn(params: dict[str, Any], target: Path) -> None:
    """Production fetcher: GET the archive endpoint, write the JSON to `target`.

    On HTTP 429 we lift the response's `Retry-After` header into a
    `_RateLimited` exception so the retry loop can honor the server's specific
    wait time instead of guessing with our fallback backoff.
    """
    response = requests.get(
        OPEN_METEO_BASE_URL,
        params=params,
        timeout=OPEN_METEO_TIMEOUT_SEC,
        headers={"User-Agent": "vininator-3000/0.1"},
    )
    if response.status_code == 429:
        raise _RateLimited(_parse_retry_after(response.headers.get("Retry-After")))
    response.raise_for_status()
    _write_json_atomic_text(response.text, target)


def _parse_retry_after(value: str | None) -> float | None:
    """Open-Meteo sends `Retry-After` as integer seconds. RFC 7231 also allows
    HTTP-dates but we don't see those from this API; if we ever do, fall back
    to the default backoff schedule by returning None.
    """
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _write_json_atomic_text(text: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Build orchestrator
# ---------------------------------------------------------------------------


def _split_daily_by_year(daily: pl.DataFrame) -> dict[int, pl.DataFrame]:
    """Partition a multi-year daily frame into one frame per calendar year."""
    if daily.is_empty():
        return {}
    years = daily.select(pl.col("date").dt.year().alias("y"))["y"].unique().to_list()
    return {int(y): daily.filter(pl.col("date").dt.year() == y) for y in years}


def _empty_climate_row(
    region: str, country: str | None, lat: float, lon: float, vintage_year: int,
    *, status: str, error: str | None,
) -> dict[str, Any]:
    """Schema-aligned row for the error / no-data path."""
    row: dict[str, Any] = {
        "region": region, "country": country, "lat": lat, "lon": lon,
        "vintage_year": vintage_year, "is_partial": True,
        "status": status, "error": error,
        "fetched_at": datetime.now(UTC),
    }
    for name in CLIMATE_ABSOLUTE_FEATURES:
        row[name] = None
        row[f"{name}_anom"] = None
    return row


def _row_from_features(
    region: str, country: str | None, lat: float, lon: float, vintage_year: int,
    features: dict[str, Any], climatology: dict[str, float],
) -> dict[str, Any]:
    """Glue feature dict + climatology into a schema-aligned parquet row."""
    is_partial = bool(features["is_partial"])
    has_data = any(features.get(name) is not None for name in CLIMATE_ABSOLUTE_FEATURES)
    if not has_data:
        status = "error"
        error: str | None = "no_data"
    elif is_partial:
        status = "partial"
        error = "season_window_incomplete"
    else:
        status = "ok"
        error = None

    row: dict[str, Any] = {
        "region": region, "country": country, "lat": lat, "lon": lon,
        "vintage_year": vintage_year, "is_partial": is_partial,
        "status": status, "error": error,
        "fetched_at": datetime.now(UTC),
    }
    for name in CLIMATE_ABSOLUTE_FEATURES:
        value = features.get(name)
        row[name] = value
        baseline = climatology.get(name)
        row[f"{name}_anom"] = (
            float(value) - baseline if value is not None and baseline is not None
            else None
        )
    return row


def build_climate_table(
    force: bool = False,
    limit: int | None = None,
    fetch_fn: OpenMeteoFetchFn | None = None,
    progress_fn: ProgressFn | None = None,
    notify_fn: NotifyFn | None = None,
    sleep_fn: SleepFn = time.sleep,
) -> Path:
    """Build `data/interim/climate.parquet` for every usable region × vintage.

    Iterates `filter_to_usable(geocode)` rows, fetches one Open-Meteo JSON per
    region via `fetch_open_meteo_region` (cache-first), computes climatology
    over `CLIMATOLOGY_WINDOW`, then writes one row per `(region, vintage_year)`
    in `CLIMATE_YEAR_RANGE`. Resume-aware: regions whose rows already exist
    in the parquet are skipped via an anti-join on `(region, country)`.

    `--force` re-fetches every JSON and recomputes every row; without it,
    rows from prior partial runs are kept and only missing regions are
    processed.

    Sequential. Open-Meteo's free tier allows up to 10k requests/day with no
    per-second hard cap; we politely space requests by `OPEN_METEO_RATE_LIMIT_SEC`.
    A future PR can lift this to a thread pool if wall-time matters.

    `OpenMeteoError` propagates immediately — it indicates persistent API
    failure that will hit every other region the same way.
    """
    settings = get_settings()
    settings.ensure_dirs()
    climate_path = settings.climate_parquet

    geocoded = filter_to_usable(scan_geocode().collect()).select(
        ["region", "country", "lat", "lon"]
    )

    if force or not climate_path.exists():
        existing = pl.DataFrame(schema=CLIMATE_SCHEMA)
    else:
        existing = pl.read_parquet(climate_path)

    todo = geocoded.join(
        existing.select(["region", "country"]).unique(),
        on=["region", "country"],
        how="anti",
    )
    if limit is not None:
        todo = todo.head(limit)

    if todo.is_empty():
        return climate_path

    total = todo.height
    years = list(range(CLIMATE_YEAR_RANGE[0], CLIMATE_YEAR_RANGE[1] + 1))

    def _notify(msg: str) -> None:
        if notify_fn is not None:
            notify_fn(msg)

    for i, row in enumerate(todo.iter_rows(named=True), start=1):
        # Polite per-request spacing; cache hits skip the sleep so an all-cached
        # re-run is instant.
        cache_path = (
            settings.open_meteo_raw_dir
            / f"{region_slug(row['region'], row['country'])}.json"
        )
        cache_hit = cache_path.exists() and cache_path.stat().st_size > 0
        if i > 1 and not cache_hit:
            sleep_fn(OPEN_METEO_RATE_LIMIT_SEC)

        _notify(f"... climate {i}/{total}: {row['region']!r}, {row['country']!r}")
        region_rows: list[dict[str, Any]] = []
        try:
            json_path = fetch_open_meteo_region(
                row["region"], row["country"], row["lat"], row["lon"],
                force=force, fetch_fn=fetch_fn, notify_fn=notify_fn,
                sleep_fn=sleep_fn,
            )
            daily = load_open_meteo_daily(json_path)
            daily_by_year = _split_daily_by_year(daily)
            climatology = compute_climatology(
                daily_by_year, row["lat"], CLIMATOLOGY_WINDOW
            )
            for year in years:
                features = compute_climate_features(daily, row["lat"], year)
                region_rows.append(
                    _row_from_features(
                        row["region"], row["country"], row["lat"], row["lon"],
                        year, features, climatology,
                    )
                )
        except OpenMeteoError:
            raise
        except Exception as exc:  # noqa: BLE001 — bad region shouldn't kill the run
            _notify(f"!!! climate {row['region']!r} failed: {exc!r}")
            error_msg = str(exc)
            for year in years:
                region_rows.append(
                    _empty_climate_row(
                        row["region"], row["country"], row["lat"], row["lon"],
                        year, status="error", error=error_msg,
                    )
                )

        new_df = pl.DataFrame(region_rows, schema=CLIMATE_SCHEMA)
        existing = pl.concat([existing, new_df], how="vertical")
        _write_parquet_atomic(existing, climate_path)

        if progress_fn is not None:
            progress_fn(i, total)

    return climate_path


def scan_climate() -> pl.LazyFrame:
    """Lazy frame over the climate cache."""
    path = get_settings().climate_parquet
    if not path.exists():
        raise FileNotFoundError(
            f"Climate parquet not found at {path}. "
            "Run `uv run vininator features climate` first."
        )
    return pl.scan_parquet(path)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def _write_parquet_atomic(df: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.write_parquet(
        tmp,
        # Embed the Open-Meteo attribution in the parquet metadata.
        metadata={"source": OPEN_METEO_ATTRIBUTION},
    )
    tmp.replace(target)
