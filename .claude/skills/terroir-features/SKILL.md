---
name: terroir-features
description: Expert guidance for building the Vininator 3000 terroir feature pipeline — ERA5 climate reanalysis, SoilGrids soil + DEM terrain, region geocoding, and the cache-first TerroirProvider that serves features for both training (region × vintage_year) and live inference. Trigger this whenever an agent touches src/vininator/features/{climate,soil,terroir,build}.py, src/vininator/data/geocode.py, src/vininator/api/terroir_provider.py, or discusses Phase 2/3/7 of PROJECT.md. Also trigger on any prompt mentioning ERA5, GDD, growing degree days, growing season, climatology, anomaly features, SoilGrids, CaCO3 / calcareous, drainage class, region centroid, geocoding wine regions, terroir cache, vintage-year features, or "feature pipeline for wine prediction". Pull this skill *before* writing the first line of code in those modules — the design rules (split keys, leakage isolation, climatology baseline, partial-vintage fallback) are easier to get right up front than to retrofit.
---

# Terroir feature engineering for Vininator 3000

You are designing or extending the **terroir feature block** — the part of the system that turns `(region_string, vintage_year)` into a vector of climate + soil + terrain numbers that a CatBoost model can learn from. This is the distinctive ML contribution of the project; everything else (grape, producer, price) is table-stakes baseline.

Before writing code, read **`PROJECT.md` §3, §4 (Phase 2 + Phase 7), §7**, then this skill, then the relevant reference file(s). Re-read `.claude/CLAUDE.md` for the layering rules — terroir code lives under `features/` and must not import from `models/` or `api/`.

## The shape of the problem

You are building **one batch pipeline and one online provider that share the same code**.

```
                ┌──────────────────────────────┐
  region str ──►│  geocode → (lat, lon)        │  per-region, cached forever
                └──────────────────────────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼                                ▼
   (lat, lon, vintage_year)              (lat, lon)
              │                                │
              ▼                                ▼
   ┌────────────────────┐          ┌────────────────────┐
   │ ERA5 daily pull    │          │ SoilGrids + DEM    │
   │ → growing season   │          │ → 0–30 cm profile  │
   │ → GDD, precip,     │          │ → CaCO3, pH,       │
   │   heat days,       │          │   texture, SOC,    │
   │   anomalies        │          │   elevation, slope │
   └────────────────────┘          └────────────────────┘
              │                                │
              └──────────┬─────────────────────┘
                         ▼
              data/interim/terroir.parquet  ← batch
                         │
                         ▼
                CatBoost feature table
```

The **same pure functions** `compute_climate_features(lat, lon, year)` and `compute_soil_features(lat, lon)` are reused by `api/terroir_provider.py` at inference — that's the layering rule that lets one codebase serve both. Don't write a separate "online" version; you'll drift and the model will see different features than the API.

## Required decisions before coding

Confirm with the user (or note explicitly in the module docstring if unambiguous) before proceeding:

1. **CDS API key available?** ERA5 needs `~/.cdsapirc` with a valid Copernicus account. If absent, stop and ask — don't write fetch code that silently fails.
2. **Which `cdsapi` dataset?** Default to `reanalysis-era5-land` (9 km grid, land-only, finer than ERA5 single levels for viticulture). Fall back to `reanalysis-era5-single-levels` only if ERA5-Land is missing for the region (rare; happens for tiny islands).
3. **Climatology window.** Default **1991–2020** WMO normal. If the user has a strong reason for a different window (e.g., 1981–2010 for older vintages), confirm — but never compute climatology on a window that overlaps the wines being predicted.
4. **Cache backend.** Parquet for climate (one file per `(region, year)`) and JSON for raw SoilGrids responses. SQLite for the online provider's warm cache (small, single-file, atomic). See `references/provider_blueprint.md`.

## Step-by-step build order

Build in this order. Each step produces a parquet that the next step joins to — no upstream step depends on the next.

### 1. Geocoding — `src/vininator/data/geocode.py`

- One row per **unique** region string from WineSensed. Expect 1k–5k uniques after light canonicalization (strip, lowercase, collapse whitespace, remove trailing country if duplicated).
- Use `geopy.geocoders.Nominatim` with a descriptive `user_agent="vininator/0.1 (<contact>)"` (Nominatim enforces this in their TOS).
- **Rate limit 1 req/sec** via `geopy.extra.rate_limiter.RateLimiter`. Anything faster will get the IP blocked.
- Cache to `data/interim/geocode.parquet` with columns `region, lat, lon, raw_address, status, fetched_at`. Resume by left-joining against this on `region` and only geocoding the missing rows.
- For ambiguous strings like `"Bordeaux"` (which Nominatim may resolve to the city), pass `country_codes` hints when the region row carries a country, and **prefer the result whose `type` is `administrative` or `region`**. Accept lossiness; a 50 km centroid offset is fine for wine regions of that size.
- Failures: write a `status` field (`ok` | `not_found` | `ambiguous` | `error`). Never drop the row — downstream code can choose whether to skip or fall back to country centroid.

### 2. Climate features — `src/vininator/features/climate.py`

Long version: **`references/era5_recipes.md`**. Quick rules:

- Public surface is two pure functions:
  ```python
  def fetch_daily_era5(lat: float, lon: float, year: int) -> pl.DataFrame: ...
  def compute_climate_features(daily: pl.DataFrame, lat: float) -> dict[str, float]: ...
  ```
  `fetch_daily_era5` handles caching; `compute_climate_features` is pure (same input → same output, no I/O). This split is what lets the same code serve batch + online and stay testable.

- **Growing season window depends on hemisphere** (decided by `lat`):
  - Northern (`lat > 0`): **April 1 – October 31** of `vintage_year`.
  - Southern (`lat < 0`): **October 1 of (vintage_year − 1) – April 30 of vintage_year**.
  - Vintage year convention for southern hemisphere wines = harvest year; this is industry standard and what WineSensed records.

- **Variables to request from ERA5-Land**:
  - `2m_temperature` (hourly, K → °C) — aggregate to daily min/mean/max.
  - `total_precipitation` (hourly, m → mm) — sum to daily.
  - Optional: `surface_solar_radiation_downwards` (J/m² → MJ/m²/day) — proxy for sunshine, useful but not critical.

- **Features to compute** (all over the growing-season window):
  - `gdd_10c` — `sum(max(daily_mean − 10°C, 0))`. The canonical viticulture index.
  - `precip_total_mm` — sum of daily precip.
  - `precip_harvest_mm` — sum over the last 30 days of the window (harvest precip; rain at harvest is the strongest single quality signal in many regions).
  - `heat_spike_days` — count of days `Tmax > 35°C`.
  - `frost_days_spring` — for the Northern hemisphere, count April–May days `Tmin < 0`. For Southern, October–November.
  - `diurnal_range_mean` — mean of `(Tmax − Tmin)` over the window.
  - `solar_total_mj` (optional) — sum of daily SSRD if pulled.

- **Anomalies are computed against a 30-year climatology per (lat, lon)**. For each feature, also emit `<feature>_anom` = `value − climatology_mean(feature)`. The climatology itself is cached at `data/interim/climatology.parquet` keyed on `(lat, lon, feature)`. The hot year that matters is the one that's hot *for that place*; absolute GDD compares Bordeaux to Mendoza, anomaly compares Bordeaux 2003 to a typical Bordeaux. Anomaly features routinely beat absolute values in tree models trained across regions.

- **Cache shape**: one parquet per `(region, year)` under `data/interim/era5_daily/{region_slug}/{year}.parquet` holding the daily aggregates. The expensive thing is the CDS request; once the daily parquet exists, `compute_climate_features` is sub-millisecond.

- **Resume strategy**: before any CDS call, list existing parquets; emit a request only for missing `(region, year)`. The CDS queue can stall for hours — make this restartable from any point. See `references/era5_recipes.md` for the exact request body, retry/backoff, and how to batch multi-year requests to reduce queue overhead.

- **Partial-season handling** (live serving only — batch never sees this): if the user asks for a vintage whose growing season is still in progress or has only partially been ingested by ERA5 (ERA5 has a ~5-day publication lag), `compute_climate_features` returns climatology values for the missing portion and sets `is_partial=True`. The provider exposes this flag. The CatBoost model is trained with `is_partial=False` rows only, but the column exists so live serving can communicate uncertainty. See `references/provider_blueprint.md`.

### 3. Soil + terrain features — `src/vininator/features/soil.py`

Long version: **`references/soilgrids_recipes.md`**. Quick rules:

- **Public surface**:
  ```python
  def fetch_soilgrids(lat: float, lon: float) -> dict: ...      # cached
  def fetch_elevation_slope(lat: float, lon: float) -> dict: ...  # cached
  def compute_soil_features(raw_soil: dict, raw_dem: dict) -> dict[str, float | str | bool]: ...
  ```

- **SoilGrids endpoint**: `https://rest.isric.org/soilgrids/v2.0/properties/query`. Free, no auth, but flaky — single-pixel queries occasionally 5xx, and noise at the 250 m grid is real.
  - **Buffer**: query a small grid around the centroid (e.g. ±0.005° in lat/lon ≈ ~500 m, 3×3 = 9 points) and average the `mean` values. This kills single-pixel noise.
  - **Depth band**: `0-30cm` (topsoil) for `phh2o`, `cec`, `clay`, `sand`, `silt`, `soc`, `bdod`. For `cfvo` (coarse fragments) use the same band.
  - **CaCO3**: SoilGrids does **not** expose calcium carbonate directly. Use **`phh2o > 7.5`** as a proxy for calcareous soils (chalky / limestone), and treat the proxy as approximate — flag this clearly in the docstring so a future reader knows it's not the literal CaCO3 measurement that PROJECT.md mentions.
  - **Retry**: 3 attempts, exponential backoff (1s, 4s, 16s). After that, write a `status=error` row to the cache and continue — don't crash the batch.

- **Features**:
  - `clay_pct`, `sand_pct`, `silt_pct`, `soc_gkg`, `cec_cmolkg`, `bdod_kgdm3`, `ph_h2o`, `coarse_frag_pct` (all averages over the 3×3 buffer).
  - `drainage_class` (categorical): bucket from `(clay_pct, sand_pct)` — `sandy` (sand ≥ 60), `loamy` (otherwise), `clayey` (clay ≥ 40), `chalky` (when `calcareous=True`). Precedence: chalky > clayey > sandy > loamy when multiple bucket boundaries fire.
  - `calcareous` (bool): `ph_h2o ≥ 7.5`.

- **Terrain**: DEM via the `elevation` package (SRTM 30 m) or Open-Elevation REST. The `elevation` package builds GDAL VRTs locally — heavier setup but offline-capable. For ≤5k centroids, Open-Elevation REST is easier; rate-limit to ~1 req/sec and cache. Features:
  - `elevation_m` — point elevation.
  - `slope_deg` — compute from a 3×3 elevation grid around the centroid (Horn's algorithm). Open-Elevation lets you POST a small lat/lon array per request, so one request per centroid suffices.
  - `aspect_deg` (optional) — orientation of the slope (south-facing matters in cool regions). Only add if you confirm it's worth the complexity.

- **Cache**: one JSON per region under `data/interim/soil_raw/{region_slug}.json` (SoilGrids response) and one per region under `data/interim/dem/{region_slug}.json`. Soil is effectively immutable — soil entries never expire.

### 4. The joiner — `src/vininator/features/terroir.py`

Tiny module. One function:

```python
def build_terroir_table(
    geocode: pl.DataFrame,
    climate: pl.DataFrame,
    soil: pl.DataFrame,
) -> pl.DataFrame:
    """Outer-join the three blocks on region (for soil) and (region, year) (for climate).

    Soil values are region-only and broadcast across vintages. Output is keyed on
    (region, vintage_year) with one row per cell. Rows where geocoding failed are
    kept with null climate/soil — the model can learn from the missingness pattern,
    and dropping them would silently shrink the dataset.
    """
```

Write the result to `data/interim/terroir.parquet`. This is the artifact `features/build.py` joins to the wine table.

### 5. Feature assembly — `src/vininator/features/build.py`

This is where leakage gets introduced if you're careless. Read **`references/leakage_rules.md`** before adding any aggregate or text-derived feature. The hard rules:

- Soil is region-only — safe to compute on the full dataset.
- Climate is `(region, vintage_year)` — also safe.
- **Producer aggregates** (mean rating, std, n_reviews per producer) — compute on the **training fold only**, then left-join onto test/val. Never on the full dataset. Same applies to any `(grape, region)` mean-rating fallback used for price imputation.
- **Text-derived body/acidity/tannin** parsed from review text — aggregate per wine (majority vote across that wine's reviews). Because the split is by `wine_id`, a wine is entirely in train or entirely in test, so per-wine aggregation does **not** leak. But never aggregate across wines (e.g., "average body for this region") — that pulls test labels into train.

### 6. The online provider — `src/vininator/api/terroir_provider.py`

This is the thing that lets the API answer for any `(region, vintage_year)` including unseen current vintages. Read **`references/provider_blueprint.md`** for the full design. Summary:

- Tier 1: in-process `functools.lru_cache` (size 1024) keyed on `(region, year)`.
- Tier 2: SQLite at `${VININATOR_CACHE_DIR}/terroir.sqlite`, one row per `(region, year)` holding a JSON blob of features + `fetched_at` + `is_partial`.
- Tier 3: R2 / B2 object at `terroir/{region_slug}/{year}.json`. Shared across deployments — essential because Fly.io's scale-to-zero loses the SQLite cache on restart unless persisted to a volume.
- Tier 4: live ERA5 + SoilGrids fetch via the same `features/climate.py` and `features/soil.py` functions used in batch. On miss, write upward through every cache tier.

TTL: weather entries refresh after 90 days; soil entries never expire.

The provider also owns the **partial-season fallback** logic — when ERA5 hasn't published the full growing season yet, fill the remainder with climatology and set `is_partial=True`.

## Anti-patterns to avoid

These are the failure modes that are easy to fall into. If you catch yourself doing one, stop and reconsider.

- **Recomputing climatology every run.** The 30-year baseline per `(lat, lon)` is expensive (decades of daily ERA5). Compute it once, cache to `data/interim/climatology.parquet`, treat it as immutable. Bump the cache only when you change the climatology window deliberately.
- **Skipping the buffer on SoilGrids.** Single-pixel queries are noisy enough to flip `drainage_class` in adjacent runs and create spurious "feature drift." Always average the 3×3.
- **Letting the API code import from `data/` or `notebooks/`.** The provider must only use `features/climate.py` and `features/soil.py`. If you need a data loader at serve time, you've broken the layering.
- **Adding "just one" feature in `models/` instead of `features/`.** Every numerical feature must come out of the batch pipeline so it shows up identically at inference. Features defined in the model training script are an inference-skew bug waiting to happen.
- **Using `vintage_year` as both a feature and a split column.** It's fine as a feature, but the future-vintage holdout (2019–2021) means the model must extrapolate; don't add a feature that's equivalent to year (e.g., a per-year random effect).
- **Forgetting that ERA5-Land returns Kelvin and meters.** Convert at ingest. Storing native units in interim parquets ensures the bug shows up immediately, but every reader needs to remember — easier to convert once at fetch time.

## What "done" looks like for this phase

- `data/interim/geocode.parquet`, `climate.parquet` (one parquet per region-year aggregated to one summary file), `soil.parquet`, `terroir.parquet` all exist and are reproducible from the CLI: `vininator features build-terroir`.
- `tests/features/test_climate.py` includes a frozen-input test where a synthetic daily dataframe goes through `compute_climate_features` and produces the documented numbers — this catches future refactors that silently change feature definitions.
- `tests/features/test_soil.py` includes a recorded SoilGrids JSON response (committed to the repo) being parsed end-to-end, no network.
- `src/vininator/api/terroir_provider.py` has unit tests using a mock fetch function, verifying tier-by-tier promotion and TTL expiry.
- Climatology cache exists and the climatology window is logged in any experiment that uses anomaly features.

## Reference files

Read the relevant file in full when you start the corresponding step. They are written for someone implementing the module, not for skim reading.

- **`references/era5_recipes.md`** — exact `cdsapi` request body, NetCDF → polars conversion, GDD math, anomaly + climatology computation, retry/backoff, partial-season fallback.
- **`references/soilgrids_recipes.md`** — REST endpoint, buffer averaging, depth bands, retry policy, the CaCO3 proxy decision, DEM/slope computation.
- **`references/leakage_rules.md`** — split-by-`wine_id` semantics, producer aggregate computation order, future-vintage holdout, what's safe to compute on the full dataset.
- **`references/provider_blueprint.md`** — `TerroirProvider` class outline, tier-by-tier promotion, SQLite schema, R2 key layout, TTL policy, partial-season detection, cache warmer cron.
