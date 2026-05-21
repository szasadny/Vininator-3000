# TerroirProvider blueprint

Implementation reference for `src/vininator/api/terroir_provider.py`. This is the only component that hits ERA5 or SoilGrids at request time.

## Why the layering matters

The model is trained on terroir features produced by the **batch pipeline** in `src/vininator/features/`. If `terroir_provider.py` computes features even slightly differently — different unit conversion, different growing-season window, different climatology window — every prediction at serve time is on slightly skewed inputs. The model has no defense against that; it'll just be subtly wrong.

So: `terroir_provider.py` does **not** reimplement feature math. It imports `fetch_daily_era5`, `compute_climate_features`, `fetch_soilgrids`, `fetch_elevation_slope`, `compute_soil_features` from `features/` and orchestrates caching around them. Anything else is a feature-skew bug.

## Cache tier design

Four tiers, each strictly faster and dumber than the next:

| Tier | Storage | Lookup | Latency | Survives |
| ---: | --- | --- | --- | --- |
| 1 | `functools.lru_cache` (in-process) | by key | µs | process lifetime |
| 2 | SQLite at `${VININATOR_CACHE_DIR}/terroir.sqlite` | by key | sub-ms | VM restart, if on a volume |
| 3 | R2 / B2 object | by key | ~50ms | forever |
| 4 | Live ERA5 + SoilGrids fetch | computed | 30s–60s | — |

A miss propagates downward; a hit propagates the result back upward through every cache tier above it. That way the same key is fast on every subsequent request from any worker in any process.

## Public surface

```python
@dataclass
class TerroirFeatures:
    # climate (region, vintage_year)
    gdd_10c: float | None
    precip_total_mm: float | None
    precip_harvest_mm: float | None
    heat_spike_days: int | None
    frost_days_spring: int | None
    diurnal_range_mean: float | None
    # anomalies
    gdd_10c_anom: float | None
    # ... one per climate feature
    # soil (region)
    ph_h2o: float | None
    clay_pct: float | None
    sand_pct: float | None
    silt_pct: float | None
    soc_gkg: float | None
    cec_cmolkg: float | None
    bdod_kgdm3: float | None
    coarse_frag_pct: float | None
    calcareous: bool | None
    drainage_class: str | None
    # terrain (region)
    elevation_m: float | None
    slope_deg: float | None
    # metadata
    is_partial: bool
    source: str  # which tier served this — debug only

class TerroirProvider:
    async def get(self, region: str, vintage_year: int) -> TerroirFeatures: ...
```

The API service holds a singleton `TerroirProvider` at app startup. Every `/predict` call goes through `provider.get(region, year)`.

## Lookup flow

```python
async def get(self, region: str, vintage_year: int) -> TerroirFeatures:
    key = (canonicalize(region), vintage_year)

    # Tier 1
    cached = self._lru.get(key)
    if cached is not None and not self._expired(cached):
        return cached._replace(source="lru")

    # Tier 2
    row = self._sqlite.get(key)
    if row is not None and not self._expired(row):
        self._lru[key] = row
        return row._replace(source="sqlite")

    # Tier 3
    blob = await self._r2.get(f"terroir/{slug(region)}/{vintage_year}.json")
    if blob is not None and not self._expired(blob):
        features = TerroirFeatures(**blob)
        self._sqlite.put(key, features)
        self._lru[key] = features
        return features._replace(source="r2")

    # Tier 4 — live fetch via the same code as the batch pipeline
    features = await self._fetch_live(region, vintage_year)
    await self._r2.put(...)
    self._sqlite.put(key, features)
    self._lru[key] = features
    return features._replace(source="live")
```

`_expired` consults the TTL policy below. Soil-only entries are never expired; the region-keyed soil cache lives in a separate SQLite table.

## TTL policy

- **Soil + terrain entries**: no TTL. The earth doesn't move on a human timescale at our resolution.
- **Climate entries for closed vintages** (vintage_year + harvest is in the past, ERA5 has had ≥6 months to backfill): 90-day soft TTL. ERA5 occasionally publishes corrections; a refresh every 90 days catches them at negligible cost.
- **Climate entries with `is_partial=True`**: 7-day soft TTL. The whole point is that the data is still being produced.

`_expired` is a single function reading `fetched_at` and `is_partial` off the cached row. Centralize the policy here, not at the call sites.

## SQLite schema

```sql
CREATE TABLE IF NOT EXISTS climate_cache (
    region_slug TEXT NOT NULL,
    vintage_year INTEGER NOT NULL,
    payload TEXT NOT NULL,        -- JSON of climate features
    is_partial INTEGER NOT NULL,
    fetched_at INTEGER NOT NULL,  -- unix epoch
    PRIMARY KEY (region_slug, vintage_year)
);

CREATE TABLE IF NOT EXISTS soil_cache (
    region_slug TEXT PRIMARY KEY,
    payload TEXT NOT NULL,        -- JSON of soil + terrain features
    fetched_at INTEGER NOT NULL
);
```

Reads use `PRAGMA journal_mode=WAL;` to allow concurrent reads from multiple uvicorn workers. Writes go through a single asyncio Lock to serialize SQLite writes from async code (SQLite handles concurrent writers but is happier with one writer at a time when wrapped in asyncio).

## R2 / B2 layout

S3-compatible. Keys:

- `terroir/{region_slug}/climate/{year}.json` — `{"features": {...}, "is_partial": false, "fetched_at": 1731000000}`.
- `terroir/{region_slug}/soil.json` — `{"features": {...}, "fetched_at": 1731000000}`.

Use the bucket's lifecycle rule to expire `is_partial=true` blobs after 30 days as a hygiene net (SQLite TTL catches them sooner, but the bucket policy is the backstop). Don't expire `is_partial=false` blobs ever — let the 90-day app-level refresh handle them.

R2 credentials live in environment variables (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`, `R2_ENDPOINT`). Read once at provider construction, never re-read.

## Partial-season fallback

The provider, **not** the pure feature function, owns this:

```python
async def _fetch_live_climate(self, region: str, year: int) -> dict:
    lat, lon = await self._geocode.resolve(region)
    daily = await fetch_daily_era5(lat, lon, year)  # may be partial

    # Detect partial coverage relative to the hemisphere's growing season
    expected = 214 if lat > 0 else 212
    is_partial = daily.height < expected - 5

    if is_partial:
        # Fill missing days with climatology daily means
        clim_daily = await self._fetch_climatology_daily(lat, lon)
        daily = self._merge_with_climatology(daily, clim_daily)

    features = compute_climate_features(daily, lat)
    climatology = await self._fetch_climatology_summary(lat, lon)
    features = with_anomalies(features, climatology)
    features["is_partial"] = is_partial
    return features
```

`_fetch_climatology_daily` returns a synthetic 365-day "typical year" assembled from the climatology window. Merging is a left-join on day-of-year: if the day exists in `daily`, use the observed value; otherwise use the climatology mean for that day-of-year. This means GDD and precipitation sums degrade gracefully toward typical-year values as the season progresses, rather than dropping to zero.

The `is_partial=True` flag rides along on the response so the API can include it in the prediction payload — the frontend can render "based on partial vintage data" as a caveat when needed.

## Cache warmer

A small script at `deploy/warm_cache.py` runs on a daily cron. For each `(top_100_regions, current_year)` and `(top_100_regions, current_year - 1)`, it calls `provider.get(region, year)`. The first call per cell does the slow ERA5 fetch; subsequent user requests for the same cell are sub-millisecond. The warmer also refreshes any entry whose `fetched_at` is older than the TTL.

Run it on Fly's scheduled machines or as a HuggingFace Spaces cron. Output goes to a single log file — the warmer doesn't need its own observability stack.

## Concurrency note

ERA5 queues serialize across the CDS user account anyway, so making the provider fetch ERA5 concurrently buys nothing. SoilGrids is fine to call in parallel up to ~5 concurrent requests. The straightforward implementation — a single asyncio Lock per `(region, year)` key so duplicate concurrent requests for the same cell wait on one fetch rather than triggering N — is enough for v1. A `dict[key, asyncio.Future]` of in-flight requests is the standard pattern; reuse one if it's already in the codebase.

## Tests

- Unit test using a mock `_fetch_live` that returns a deterministic payload; verify LRU → SQLite → R2 promotion across consecutive calls.
- TTL test: insert a row with `fetched_at` 100 days ago, assert `_expired` is True for climate but False for soil.
- Partial-season test: mock `fetch_daily_era5` to return a 100-day frame; assert `is_partial=True` and that climatology-filled features are within a sensible band.
- Geocode failure test: when geocode returns `status=not_found`, the provider returns a `TerroirFeatures` with all-null fields and does not call ERA5 / SoilGrids.
