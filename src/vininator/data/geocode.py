"""Geocode unique `RegionName` values to `(lat, lon)` via Nominatim.

Phase 2 step 1. Every downstream terroir feature is keyed on `(lat, lon)`,
so we need one geocode per unique region. Nominatim is free, rate-limited
to 1 req/sec by their TOS, and noisy on ambiguous strings (e.g. "Bordeaux"
can resolve to the city rather than the wine region). We accept the
lossiness: appellation centroids are accurate enough for NASA POWER's
~55 km grid and SoilGrids' 250 m pixels.

The cache (`data/interim/geocode.parquet`) is the resume state. Re-running
processes only `(region, country)` pairs not already present. Failed lookups
stay in the cache with `status != "ok"` — downstream code decides whether to
skip or fall back to a country centroid; we never silently drop a region.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from geopy.exc import GeocoderRateLimited, GeocoderServiceError, GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.location import Location

from vininator.config import (
    GEOCODE_BACKOFF_BASE_SEC,
    GEOCODE_BACKOFF_CAP_SEC,
    GEOCODE_BAD_RESULT_TYPES,
    GEOCODE_CHECKPOINT_EVERY,
    NOMINATIM_RATE_LIMIT_SEC,
    NOMINATIM_TIMEOUT_SEC,
    get_settings,
)
from vininator.data.load import scan_xwines_wines

GeocodeFn = Callable[[str], Location | None]
ProgressFn = Callable[[int, int], None]
NotifyFn = Callable[[str], None]
SleepFn = Callable[[float], None]

GEOCODE_SCHEMA: dict[str, pl.DataType] = {
    "region": pl.String(),
    "country": pl.String(),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "raw_address": pl.String(),
    "result_type": pl.String(),
    "status": pl.String(),
    "error": pl.String(),
    "fetched_at": pl.Datetime("us", "UTC"),
}


def load_unique_regions() -> pl.DataFrame:
    """Return deduplicated `(region, country)` pairs from the wines parquet.

    Light canonicalization only: strip whitespace, drop rows where region is
    null or empty. No lowercasing — Nominatim is case-insensitive and X-Wines
    casing is usually meaningful (e.g. "Côte de Beaune").
    """
    return (
        scan_xwines_wines()
        .select(
            pl.col("RegionName").str.strip_chars().alias("region"),
            pl.col("Country").str.strip_chars().alias("country"),
        )
        .filter(pl.col("region").is_not_null() & (pl.col("region") != ""))
        .unique()
        .sort(["country", "region"])
        .collect()
    )


def geocode_regions(
    force: bool = False,
    limit: int | None = None,
    geocode_fn: GeocodeFn | None = None,
    progress_fn: ProgressFn | None = None,
    notify_fn: NotifyFn | None = None,
    sleep_fn: SleepFn = time.sleep,
    checkpoint_every: int = GEOCODE_CHECKPOINT_EVERY,
    backoff_base_sec: float = GEOCODE_BACKOFF_BASE_SEC,
    backoff_cap_sec: float = GEOCODE_BACKOFF_CAP_SEC,
) -> Path:
    """Geocode every unique region pair, resumable across runs.

    `force=True` discards the existing cache and re-geocodes everything.
    `limit` caps work per run (still resumable — subsequent runs cover the rest).
    `checkpoint_every` flushes partial results to disk every N rows, so a Ctrl+C
    or process kill mid-run loses at most that many rows. The final partial
    batch is always flushed, including when an exception aborts the loop.

    When Nominatim returns 429 (rate limit), the row is **not** appended to the
    cache — it stays in `todo` via the anti-join so the next run retries it.
    Between such rows we sleep `max(retry_after, base * 2^(consecutive-1))`
    seconds, capped at `backoff_cap_sec`. Counter resets on any successful row.

    `geocode_fn`, `progress_fn`, `notify_fn`, and `sleep_fn` exist so tests (and
    the CLI) can inject offline doubles, progress sinks, status logs, and a
    deterministic clock. In production `geocode_fn` defaults to a Nominatim
    geocoder wrapped in `geopy.extra.rate_limiter.RateLimiter`.
    """
    settings = get_settings()
    settings.ensure_dirs()
    cache_path = settings.geocode_parquet

    regions = load_unique_regions()

    if force or not cache_path.exists():
        existing = pl.DataFrame(schema=GEOCODE_SCHEMA)
    else:
        existing = pl.read_parquet(cache_path)

    todo = regions.join(
        existing.select(["region", "country"]),
        on=["region", "country"],
        how="anti",
    )
    if limit is not None:
        todo = todo.head(limit)

    if todo.is_empty():
        return cache_path

    if geocode_fn is None:
        geocode_fn = _default_geocode_fn()

    total = todo.height
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal existing, pending
        if not pending:
            return
        new_df = pl.DataFrame(pending, schema=GEOCODE_SCHEMA)
        existing = pl.concat([existing, new_df], how="vertical")
        _write_parquet_atomic(existing, cache_path)
        pending = []

    def notify(msg: str) -> None:
        if notify_fn is not None:
            notify_fn(msg)

    completed = 0
    consecutive_rate_limited = 0
    try:
        for i, row in enumerate(todo.iter_rows(named=True), start=1):
            try:
                result_row = geocode_one(row["region"], row["country"], geocode_fn)
            except GeocoderRateLimited as exc:
                consecutive_rate_limited += 1
                retry_after = getattr(exc, "retry_after", None) or 0.0
                backoff = backoff_base_sec * (2 ** (consecutive_rate_limited - 1))
                sleep_sec = min(max(float(retry_after), backoff), backoff_cap_sec)
                notify(
                    f"rate-limited at row {i}/{total} on "
                    f"{row['region']!r}, {row['country']!r}; "
                    f"sleeping {sleep_sec:.0f}s "
                    f"(consecutive={consecutive_rate_limited})"
                )
                # Persist anything in-flight before the sleep so a Ctrl+C
                # during the wait doesn't lose work that already succeeded.
                flush()
                sleep_fn(sleep_sec)
                continue

            pending.append(result_row)
            completed = i
            consecutive_rate_limited = 0
            if i % checkpoint_every == 0:
                flush()
                if progress_fn is not None:
                    progress_fn(i, total)
    finally:
        flush()
        if progress_fn is not None:
            progress_fn(completed, total)

    return cache_path


def geocode_one(region: str, country: str | None, geocode_fn: GeocodeFn) -> dict[str, Any]:
    """Single Nominatim lookup.

    Returns a dict matching `GEOCODE_SCHEMA` for any non-rate-limit outcome.
    The `status` field — not `lat is None` — is the source of truth for whether
    the lookup succeeded.

    Raises `GeocoderRateLimited` so the outer loop can apply exponential
    backoff. Rate-limited rows must not be cached (they'd masquerade as a
    permanent failure and skip future retries via the anti-join).
    """
    query = f"{region}, {country}" if country else region
    base: dict[str, Any] = {
        "region": region,
        "country": country,
        "lat": None,
        "lon": None,
        "raw_address": None,
        "result_type": None,
        "status": "error",
        "error": None,
        "fetched_at": datetime.now(UTC),
    }
    try:
        result = geocode_fn(query)
    except GeocoderRateLimited:
        raise
    except (GeocoderServiceError, GeocoderTimedOut) as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    if result is None:
        base["status"] = "not_found"
        return base

    raw = getattr(result, "raw", {}) or {}
    base.update(
        {
            "lat": float(result.latitude),
            "lon": float(result.longitude),
            "raw_address": result.address,
            "result_type": raw.get("type"),
            "status": "ok",
        }
    )
    return base


def scan_geocode() -> pl.LazyFrame:
    """Lazy frame over the geocode cache."""
    path = get_settings().geocode_parquet
    if not path.exists():
        raise FileNotFoundError(
            f"Geocode parquet not found at {path}. "
            "Run `uv run vininator features geocode` first."
        )
    return pl.scan_parquet(path)


def filter_to_usable(geocode_df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows that shouldn't be used as wine-region coordinates.

    Keeps rows with `status='ok'` AND `result_type` not in
    `GEOCODE_BAD_RESULT_TYPES`. This is a blacklist (not whitelist) because
    Nominatim tags real wine regions with a wildly varied set of types —
    a tight whitelist drops hundreds of real appellations. See the bad-types
    block in `config.py` for the reasoning.

    Filter is applied at read time; the parquet itself stays untouched so
    audits remain reproducible.
    """
    return geocode_df.filter(
        (pl.col("status") == "ok")
        & (
            pl.col("result_type").is_null()
            | ~pl.col("result_type").is_in(list(GEOCODE_BAD_RESULT_TYPES))
        )
    )


def result_type_distribution(geocode_df: pl.DataFrame) -> pl.DataFrame:
    """Value-count distribution of `result_type` over `status='ok'` rows, with
    one example region per type. Used by the `geocode-audit` CLI to decide
    whether the blacklist needs refining.
    """
    ok = geocode_df.filter(pl.col("status") == "ok")
    counts = (
        ok.group_by("result_type")
        .len()
        .rename({"len": "n"})
        .sort("n", descending=True)
    )
    examples = (
        ok.group_by("result_type")
        .agg(pl.col("region").first().alias("example_region"))
    )
    return counts.join(examples, on="result_type", how="left").with_columns(
        pl.col("result_type")
        .is_in(list(GEOCODE_BAD_RESULT_TYPES))
        .alias("blacklisted")
    )


def _default_geocode_fn() -> GeocodeFn:
    # swallow_exceptions=False so `GeocoderRateLimited` actually surfaces here,
    # instead of being turned into a silent `None` (which would then be cached
    # as `status="not_found"` and never retried). The outer loop's exponential
    # backoff handles 429s correctly only when they propagate.
    user_agent = get_settings().nominatim_user_agent
    geolocator = Nominatim(user_agent=user_agent, timeout=NOMINATIM_TIMEOUT_SEC)
    return RateLimiter(
        geolocator.geocode,
        min_delay_seconds=NOMINATIM_RATE_LIMIT_SEC,
        swallow_exceptions=False,
    )


def _write_parquet_atomic(df: pl.DataFrame, target: Path) -> None:
    """Write via tmp + rename so a crash mid-write doesn't corrupt the cache."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(target)
