"""Per-region soil + terrain features for the terroir block.

Phase 2 step 2. Soil composition (texture, pH, organic carbon) and terrain
(elevation, slope) are time-invariant on the wine-vintage timescale: one pull
per region is enough forever.

This module deliberately splits **fetching** (I/O, cached, retry-aware) from
**computing features** (pure, no I/O). The pure half is reused at inference
time by `api/terroir_provider.py` (Phase 7) so the model sees identical
features in batch and online — anything that goes through `compute_soil_features`
in training must go through it at serve time too.

Caveat: SoilGrids does NOT expose calcium carbonate. We approximate the
"chalky" / calcareous signal via `ph_h2o >= 7.5`, which captures the same
high-pH soils that limestone bedrock produces (Champagne, Chablis, Jerez).
This is documented for downstream readers; a literal CaCO3 dataset would be
nicer but isn't worth the extra source.
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl
import requests

from vininator.config import (
    CALCAREOUS_PH_THRESHOLD,
    DEM_GRID_DEG,
    DEM_LAT_METERS_PER_DEG,
    DRAINAGE_CLAY_PCT,
    DRAINAGE_SAND_PCT,
    OPEN_ELEVATION_TIMEOUT_SEC,
    OPEN_ELEVATION_URL,
    SOILGRIDS_BACKOFF_SEC,
    SOILGRIDS_BASE_URL,
    SOILGRIDS_BUFFER_DEG,
    SOILGRIDS_DEPTH_WEIGHTS,
    SOILGRIDS_DEPTHS,
    SOILGRIDS_PROPERTIES,
    SOILGRIDS_RATE_LIMIT_SEC,
    SOILGRIDS_TIMEOUT_SEC,
    SOILGRIDS_UNIT_CONVERSIONS,
    get_settings,
)
from vininator.data.geocode import filter_to_usable, scan_geocode

SoilFetchFn = Callable[[float, float], dict]
DemFetchFn = Callable[[list[tuple[float, float]]], dict]
ProgressFn = Callable[[int, int], None]
NotifyFn = Callable[[str], None]
SleepFn = Callable[[float], None]

SOIL_SCHEMA: dict[str, pl.DataType] = {
    "region": pl.String(),
    "country": pl.String(),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "clay_pct": pl.Float64(),
    "sand_pct": pl.Float64(),
    "silt_pct": pl.Float64(),
    "ph_h2o": pl.Float64(),
    "soc_gkg": pl.Float64(),
    "cec_cmolkg": pl.Float64(),
    "bdod_kgdm3": pl.Float64(),
    "coarse_frag_pct": pl.Float64(),
    "elevation_m": pl.Float64(),
    "slope_deg": pl.Float64(),
    "drainage_class": pl.String(),
    "calcareous": pl.Boolean(),
    "status": pl.String(),
    "error": pl.String(),
    "fetched_at": pl.Datetime("us", "UTC"),
}

# Map SoilGrids property identifier -> our parquet column name.
_SOIL_COLUMN_NAMES: dict[str, str] = {
    "phh2o": "ph_h2o",
    "clay":  "clay_pct",
    "sand":  "sand_pct",
    "silt":  "silt_pct",
    "soc":   "soc_gkg",
    "cec":   "cec_cmolkg",
    "bdod":  "bdod_kgdm3",
    "cfvo":  "coarse_frag_pct",
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def region_slug(region: str, country: str | None) -> str:
    """Stable, filesystem-safe slug for cache filenames.

    NFKD-fold then strip non-alphanumerics. Country is prefixed when present so
    collisions like Rioja (Spain) vs Rioja (Argentina) get distinct slugs.

    >>> region_slug("Côte de Beaune", "France")
    'france_cote_de_beaune'
    """
    parts = [country, region] if country else [region]
    text = " ".join(p for p in parts if p)
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "_", folded).strip("_").lower()


def _grid_3x3(lat: float, lon: float, deg: float) -> list[tuple[float, float]]:
    """3x3 grid centred on (lat, lon), spaced `deg` apart, row-major order."""
    return [
        (lat + dy * deg, lon + dx * deg)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ]


def _horn_slope_deg(elevations: list[float], lat: float) -> float:
    """Slope angle (degrees) from a 3x3 elevation grid via Horn's algorithm.

    Elevations are in row-major order:

        a b c
        d e f
        g h i

    `lat` is the centre latitude: longitudinal cell size shrinks with
    cos(lat), so a 0.0027° step is ~300 m at the equator but ~200 m at 50° N.
    """
    if len(elevations) != 9:
        raise ValueError(f"expected 9 elevations, got {len(elevations)}")
    a, b, c, d, _, f, g, h, i = elevations
    cell_y_m = DEM_GRID_DEG * DEM_LAT_METERS_PER_DEG
    cos_lat = math.cos(math.radians(lat))
    # Near the poles cos(lat) → 0; fall back to square cells to avoid div-by-zero.
    cell_x_m = cell_y_m * cos_lat if abs(cos_lat) > 1e-6 else cell_y_m
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cell_x_m)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cell_y_m)
    return math.degrees(math.atan(math.sqrt(dzdx * dzdx + dzdy * dzdy)))


def _property_topsoil_mean(response: dict, prop: str) -> float | None:
    """Depth-weighted topsoil mean for one property at one buffer point.

    SoilGrids ships separate `0-5cm`, `5-15cm`, `15-30cm` bands; we collapse
    them into a single 0-30cm value by weighting each band's `mean` by its
    thickness (5/30, 10/30, 15/30). Returns None if every band is missing.
    """
    layers = response.get("properties", {}).get("layers", [])
    for layer in layers:
        if layer.get("name") != prop:
            continue
        weighted_sum = 0.0
        total_weight = 0.0
        for d in layer.get("depths", []):
            label = d.get("label")
            weight = SOILGRIDS_DEPTH_WEIGHTS.get(label)
            if weight is None:
                continue
            v = d.get("values", {}).get("mean")
            if v is None:
                continue
            weighted_sum += float(v) * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else None
    return None


def _aggregate_soil_properties(raw_soil: dict) -> dict[str, float | None]:
    """Per property, average the topsoil-weighted mean across the 9 buffer
    points, then apply the d-factor + unit conversion."""
    out: dict[str, float | None] = {col: None for col in _SOIL_COLUMN_NAMES.values()}
    points = raw_soil.get("points", [])
    for prop in SOILGRIDS_PROPERTIES:
        values: list[float] = []
        for p in points:
            resp = p.get("response")
            if resp is None:
                continue
            v = _property_topsoil_mean(resp, prop)
            if v is not None:
                values.append(v)
        if values:
            divisor, _ = SOILGRIDS_UNIT_CONVERSIONS[prop]
            out[_SOIL_COLUMN_NAMES[prop]] = sum(values) / len(values) / divisor
    return out


def _calcareous_from_ph(ph: float | None) -> bool | None:
    if ph is None:
        return None
    return ph >= CALCAREOUS_PH_THRESHOLD


def _drainage_class(
    clay_pct: float | None, sand_pct: float | None, calcareous: bool | None
) -> str | None:
    """Coarse bucket. `chalky` wins over everything when calcareous is true."""
    if calcareous:
        return "chalky"
    if clay_pct is None or sand_pct is None:
        return None
    if clay_pct >= DRAINAGE_CLAY_PCT:
        return "clayey"
    if sand_pct >= DRAINAGE_SAND_PCT:
        return "sandy"
    return "loamy"


def _terrain_from_dem(raw_dem: dict) -> tuple[float | None, float | None]:
    """Centre elevation (m) + slope (degrees) from a DEM 3x3 cache entry."""
    if raw_dem.get("status") != "ok":
        return (None, None)
    response = raw_dem.get("response") or {}
    results = response.get("results", [])
    if len(results) != 9:
        return (None, None)
    try:
        elevations = [float(r["elevation"]) for r in results]
    except (KeyError, TypeError, ValueError):
        return (None, None)
    centre = elevations[4]
    lat = float(raw_dem.get("lat", 0.0))
    slope = _horn_slope_deg(elevations, lat)
    return (centre, slope)


def compute_soil_features(raw_soil: dict, raw_dem: dict) -> dict[str, Any]:
    """Pure aggregator: 9-point SoilGrids buffer + 3x3 DEM grid → one feature row.

    The two `raw_*` dicts are the on-disk cache shape produced by
    `fetch_soilgrids` and `fetch_elevation_slope`. Output is keyed by parquet
    column name (excluding `region`, `country`, `lat`, `lon`, `fetched_at`,
    which the caller fills in).
    """
    soil_features = _aggregate_soil_properties(raw_soil)
    ph = soil_features.get("ph_h2o")
    clay = soil_features.get("clay_pct")
    sand = soil_features.get("sand_pct")

    calcareous = _calcareous_from_ph(ph)
    drainage = _drainage_class(clay, sand, calcareous)
    elevation_m, slope_deg = _terrain_from_dem(raw_dem)

    # A successful HTTP response doesn't always mean usable data — SoilGrids
    # returns `mean: null` for urban / water / sparse-coverage pixels even when
    # the request succeeds. Downgrade the effective status to reflect real
    # data presence so downstream code can trust the `status` column.
    soil_has_data = any(v is not None for v in soil_features.values())
    dem_has_data = elevation_m is not None
    soil_status = raw_soil.get("status", "error")
    dem_status = raw_dem.get("status", "error")
    effective_soil = soil_status if soil_has_data else "error"
    effective_dem = dem_status if dem_has_data else "error"

    if effective_soil == "ok" and effective_dem == "ok":
        status = "ok"
    elif effective_soil == "error" and effective_dem == "error":
        status = "error"
    else:
        status = "partial"

    error_parts = []
    if effective_soil != "ok":
        why = soil_status if soil_status != "ok" else "no_data"
        error_parts.append(f"soil={why}")
    if effective_dem != "ok":
        why = dem_status if dem_status != "ok" else "no_data"
        error_parts.append(f"dem={why}")
    error = "; ".join(error_parts) if error_parts else None

    return {
        **soil_features,
        "elevation_m": elevation_m,
        "slope_deg": slope_deg,
        "drainage_class": drainage,
        "calcareous": calcareous,
        "status": status,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Fetchers (I/O, cached)
# ---------------------------------------------------------------------------


def _with_retries(
    fn: Callable[[], Any],
    backoffs: tuple[float, ...],
    sleep_fn: SleepFn,
) -> Any:
    """Run `fn`, retrying up to `len(backoffs)` times with the listed sleeps
    between attempts. Re-raises the last exception on total failure.
    """
    last_exc: Exception | None = None
    for attempt in range(len(backoffs) + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — retry policy is intentional catch-all
            last_exc = exc
            if attempt < len(backoffs):
                sleep_fn(backoffs[attempt])
    assert last_exc is not None
    raise last_exc


def fetch_soilgrids(
    region: str,
    country: str | None,
    lat: float,
    lon: float,
    *,
    force: bool = False,
    fetch_fn: SoilFetchFn | None = None,
    sleep_fn: SleepFn = time.sleep,
) -> dict:
    """Fetch the SoilGrids 0–30 cm profile over a 3x3 buffer around (lat, lon).

    One HTTP call per buffer point, retried per `SOILGRIDS_BACKOFF_SEC`. Cache
    is one JSON per region under `data/interim/soil_raw/{slug}.json`. Resume-
    aware: if the cache exists and `force=False`, returns it without network I/O.

    Status semantics:
    - `ok` — all 9 points returned a response
    - `partial` — 1–8 succeeded
    - `error` — all 9 failed
    """
    slug = region_slug(region, country)
    cache_path = get_settings().soil_raw_dir / f"{slug}.json"

    if not force and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if fetch_fn is None:
        fetch_fn = _default_soil_fetch_fn

    grid = _grid_3x3(lat, lon, SOILGRIDS_BUFFER_DEG)
    points: list[dict[str, Any]] = []
    for j, (plat, plon) in enumerate(grid):
        if j > 0:
            sleep_fn(SOILGRIDS_RATE_LIMIT_SEC)
        try:
            resp = _with_retries(partial(fetch_fn, plat, plon), SOILGRIDS_BACKOFF_SEC, sleep_fn)
            points.append({"lat": plat, "lon": plon, "response": resp, "error": None})
        except Exception as exc:  # noqa: BLE001
            points.append({"lat": plat, "lon": plon, "response": None, "error": str(exc)})

    n_ok = sum(1 for p in points if p["response"] is not None)
    if n_ok == 9:
        status = "ok"
    elif n_ok == 0:
        status = "error"
    else:
        status = "partial"

    cache = {
        "region": region,
        "country": country,
        "lat": lat,
        "lon": lon,
        "buffer_deg": SOILGRIDS_BUFFER_DEG,
        "points": points,
        "status": status,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(cache, cache_path)
    return cache


def fetch_elevation_slope(
    region: str,
    country: str | None,
    lat: float,
    lon: float,
    *,
    force: bool = False,
    fetch_fn: DemFetchFn | None = None,
    sleep_fn: SleepFn = time.sleep,
) -> dict:
    """Fetch elevations for a 3x3 grid around (lat, lon).

    Single POST returns all 9 elevations. Cached as one JSON per region under
    `data/interim/dem/{slug}.json`.
    """
    slug = region_slug(region, country)
    cache_path = get_settings().dem_raw_dir / f"{slug}.json"

    if not force and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if fetch_fn is None:
        fetch_fn = _default_dem_fetch_fn

    grid = _grid_3x3(lat, lon, DEM_GRID_DEG)
    try:
        response = _with_retries(partial(fetch_fn, grid), SOILGRIDS_BACKOFF_SEC, sleep_fn)
        status = "ok"
        error: str | None = None
    except Exception as exc:  # noqa: BLE001
        response = None
        status = "error"
        error = str(exc)

    cache = {
        "region": region,
        "country": country,
        "lat": lat,
        "lon": lon,
        "grid_deg": DEM_GRID_DEG,
        "points": [{"lat": pl_, "lon": po_} for pl_, po_ in grid],
        "response": response,
        "status": status,
        "error": error,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(cache, cache_path)
    return cache


def _default_soil_fetch_fn(lat: float, lon: float) -> dict:
    """One SoilGrids point query. Repeated `property` and `depth` query
    params are how the ISRIC v2 API accepts multiple values."""
    params: list[tuple[str, Any]] = [("lat", lat), ("lon", lon), ("value", "mean")]
    params.extend(("depth", d) for d in SOILGRIDS_DEPTHS)
    params.extend(("property", p) for p in SOILGRIDS_PROPERTIES)
    resp = requests.get(SOILGRIDS_BASE_URL, params=params, timeout=SOILGRIDS_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()


def _default_dem_fetch_fn(points: list[tuple[float, float]]) -> dict:
    """One Open-Elevation POST for the full 3x3 grid."""
    body = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in points]}
    resp = requests.post(OPEN_ELEVATION_URL, json=body, timeout=OPEN_ELEVATION_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Build orchestrator
# ---------------------------------------------------------------------------


def build_soil_table(
    force: bool = False,
    limit: int | None = None,
    soil_fetch_fn: SoilFetchFn | None = None,
    dem_fetch_fn: DemFetchFn | None = None,
    progress_fn: ProgressFn | None = None,
    notify_fn: NotifyFn | None = None,
    sleep_fn: SleepFn = time.sleep,
) -> Path:
    """Build `data/interim/soil.parquet` from successfully-geocoded regions.

    Iterates over `geocode_parquet` rows with `status='ok'`, calls
    `fetch_soilgrids` + `fetch_elevation_slope` (both cache-first), then
    `compute_soil_features`. Resume-aware via anti-join on (region, country).

    Errors from either fetcher land in the parquet as `status != 'ok'` rows;
    re-run with `--force` to retry them.
    """
    settings = get_settings()
    settings.ensure_dirs()
    soil_path = settings.soil_parquet

    geocoded = filter_to_usable(scan_geocode().collect()).select(
        ["region", "country", "lat", "lon"]
    )

    if force or not soil_path.exists():
        existing = pl.DataFrame(schema=SOIL_SCHEMA)
    else:
        existing = pl.read_parquet(soil_path)

    todo = geocoded.join(
        existing.select(["region", "country"]),
        on=["region", "country"],
        how="anti",
    )
    if limit is not None:
        todo = todo.head(limit)

    if todo.is_empty():
        return soil_path

    total = todo.height
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal existing, pending
        if not pending:
            return
        new_df = pl.DataFrame(pending, schema=SOIL_SCHEMA)
        existing = pl.concat([existing, new_df], how="vertical")
        _write_parquet_atomic(existing, soil_path)
        pending = []

    def notify(msg: str) -> None:
        if notify_fn is not None:
            notify_fn(msg)

    try:
        for i, row in enumerate(todo.iter_rows(named=True), start=1):
            notify(f"... soil {i}/{total}: {row['region']!r}, {row['country']!r}")
            raw_soil = fetch_soilgrids(
                row["region"], row["country"], row["lat"], row["lon"],
                fetch_fn=soil_fetch_fn, sleep_fn=sleep_fn,
            )
            raw_dem = fetch_elevation_slope(
                row["region"], row["country"], row["lat"], row["lon"],
                fetch_fn=dem_fetch_fn, sleep_fn=sleep_fn,
            )
            features = compute_soil_features(raw_soil, raw_dem)
            features["region"] = row["region"]
            features["country"] = row["country"]
            features["lat"] = row["lat"]
            features["lon"] = row["lon"]
            features["fetched_at"] = datetime.now(UTC)
            pending.append(features)
            # Each row costs ~10 HTTP calls; flushing every row keeps Ctrl+C
            # losses bounded to the in-flight one.
            flush()
            if progress_fn is not None:
                progress_fn(i, total)
    finally:
        flush()

    return soil_path


def scan_soil() -> pl.LazyFrame:
    """Lazy frame over the soil cache."""
    path = get_settings().soil_parquet
    if not path.exists():
        raise FileNotFoundError(
            f"Soil parquet not found at {path}. "
            "Run `uv run vininator features soil` first."
        )
    return pl.scan_parquet(path)


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------


def _write_parquet_atomic(df: pl.DataFrame, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(target)


def _write_json_atomic(data: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
