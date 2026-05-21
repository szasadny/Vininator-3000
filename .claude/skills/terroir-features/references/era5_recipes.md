# ERA5 recipes

Implementation reference for `src/vininator/features/climate.py`. Read in full before writing the module.

## Why ERA5-Land over ERA5 single-levels

ERA5-Land (`reanalysis-era5-land`) is on a 9 km grid and is explicitly land-focused, so its surface variables (2 m temperature, total precipitation) are tuned for terrestrial use cases. ERA5 single-levels is 31 km — too coarse for distinct viticultural appellations (e.g., Côte de Beaune and Côte de Nuits are 25 km apart). Use ERA5-Land by default. The only time to fall back to single-levels is when a coordinate is over the ocean per the land-sea mask, which can happen for islands like Santorini — handle by sliding the centroid inland 0.05° toward the nearest land pixel before refetching.

## CDS API setup

The user supplies `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <UID>:<api-key>
```

If the file is missing, fail loudly at module import time of any function that calls `cdsapi.Client()`. Do not write a fallback that silently produces empty features.

## Request body

For one `(lat, lon, year)` and the Northern hemisphere growing season:

```python
client.retrieve(
    "reanalysis-era5-land",
    {
        "variable": ["2m_temperature", "total_precipitation"],
        "year": [str(year)],
        "month": ["04", "05", "06", "07", "08", "09", "10"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "format": "netcdf",
        "area": [lat + 0.05, lon - 0.05, lat - 0.05, lon + 0.05],  # N W S E, ~5 km box
    },
    target=str(target_path),
)
```

For the Southern hemisphere, the year spans two calendar years; issue **two requests** (Oct–Dec of `year-1`, Jan–Apr of `year`) and concatenate the resulting hourly arrays before aggregating to daily. Do not try to do this in a single CDS request — the year field is rigidly per-calendar-year.

The 0.05° box trick: CDS returns the **smallest enclosing grid box** that contains the requested area. Asking for a tiny box around the centroid pulls one or two grid cells, which is what you want — single-pixel feature parity between batch and serving. The buffer averaging happens on SoilGrids, not ERA5, because ERA5's 9 km grid already smooths local noise.

## NetCDF → polars

```python
import xarray as xr
import polars as pl

ds = xr.open_dataset(target_path)
# Variables are named 't2m' and 'tp' in the NetCDF.
hourly = ds[["t2m", "tp"]].to_dataframe().reset_index()
# Drop the lat/lon multi-index columns; if multiple pixels in the box, mean across them
hourly = hourly.groupby("time").mean(numeric_only=True).reset_index()
df = pl.from_pandas(hourly)
# Unit conversion: K → °C, m → mm
df = df.with_columns(
    (pl.col("t2m") - 273.15).alias("t2m_c"),
    (pl.col("tp") * 1000.0).alias("precip_mm"),
)
# Daily aggregation
daily = df.group_by(pl.col("time").dt.date().alias("date")).agg(
    pl.col("t2m_c").min().alias("tmin_c"),
    pl.col("t2m_c").mean().alias("tmean_c"),
    pl.col("t2m_c").max().alias("tmax_c"),
    pl.col("precip_mm").sum().alias("precip_mm"),
)
```

Persist `daily` to `data/interim/era5_daily/{region_slug}/{year}.parquet`. That's the unit of caching; everything downstream is recomputable from this file alone.

## GDD and the rest of the features

All features operate on the `daily` parquet. Pure function, no I/O:

```python
def compute_climate_features(daily: pl.DataFrame, lat: float) -> dict[str, float]:
    # Window the growing season
    is_north = lat > 0
    if is_north:
        season = daily.filter(pl.col("date").dt.month().is_between(4, 10))
        spring = daily.filter(pl.col("date").dt.month().is_between(4, 5))
    else:
        # Caller has already concatenated Oct(y-1)..Apr(y); season is the whole frame
        season = daily
        spring = daily.filter(pl.col("date").dt.month().is_between(10, 11))

    # GDD base 10°C — the textbook viticulture index. Some literature uses base 50°F (10°C);
    # don't switch to other bases (5°C, 15°C) without explicit reason — comparability across
    # appellations is the point of using the standard.
    gdd = season.with_columns(
        pl.max_horizontal(pl.col("tmean_c") - 10.0, 0.0).alias("gdd_day")
    )["gdd_day"].sum()

    # Harvest precip: last 30 days of season
    harvest_cutoff = season["date"].max() - timedelta(days=30)
    harvest_precip = season.filter(pl.col("date") > harvest_cutoff)["precip_mm"].sum()

    return {
        "gdd_10c": gdd,
        "precip_total_mm": season["precip_mm"].sum(),
        "precip_harvest_mm": harvest_precip,
        "heat_spike_days": (season["tmax_c"] > 35).sum(),
        "frost_days_spring": (spring["tmin_c"] < 0).sum(),
        "diurnal_range_mean": (season["tmax_c"] - season["tmin_c"]).mean(),
    }
```

Tested with a frozen synthetic input that produces known numbers. The test is what prevents silent feature drift in refactors.

## Climatology and anomalies

The climatology is the per-location mean of each feature over a 30-year window (default 1991–2020).

```python
def build_climatology(lat: float, lon: float, years: range) -> dict[str, float]:
    rows = []
    for y in years:
        daily = fetch_daily_era5(lat, lon, y)  # uses the same cache
        rows.append(compute_climate_features(daily, lat))
    df = pl.DataFrame(rows)
    return {f"{c}_clim": df[c].mean() for c in df.columns}
```

Cache the resulting dict to `data/interim/climatology.parquet` keyed on `(lat_round, lon_round)` where you round to 0.1° (≈10 km) — adjacent vineyards share a climatology, no need to fetch 30 years of ERA5 per centroid. Log the climatology window in MLflow params for every training run.

Anomaly features are subtraction:

```python
def with_anomalies(features: dict, climatology: dict) -> dict:
    out = dict(features)
    for k, v in features.items():
        ck = f"{k}_clim"
        if ck in climatology:
            out[f"{k}_anom"] = v - climatology[ck]
    return out
```

Anomaly columns join the absolute columns in the final terroir parquet. The reason anomalies outperform absolute values in tree models trained across regions: a tree split like `gdd_10c > 1800` separates Bordeaux from Mendoza, not good vintages from bad. A split like `gdd_10c_anom > 200` separates *hot for here* from *typical for here*, which is what actually drives wine quality at first order.

## Retry, backoff, resume

CDS is queued. A single request can sit for minutes to hours. The pipeline must survive process restarts.

- Wrap `client.retrieve` in a retry with exponential backoff (initial 30s, factor 2, max 4 attempts).
- Before issuing a request, check whether the target parquet already exists; if yes, skip. This is the resume primitive.
- Run with a worker pool of **at most 4 concurrent requests** — CDS limits per-user concurrency, and exceeding it just queues longer.
- Log every request to `data/interim/era5_requests.log` with timestamp, key, status, duration. When the user inevitably says "is the pull stuck?", this log is the answer.

## Partial-season fallback (online only)

ERA5 has ~5-day publication lag, and the latest year's growing season may be incomplete when a user queries for "this year's Burgundy."

In `compute_climate_features`, detect partial coverage:

```python
expected_days = 214 if is_north else 212  # Apr 1 – Oct 31 = 214; Oct 1 – Apr 30 = 212
actual_days = season.height
is_partial = actual_days < expected_days - 5  # tolerate a few missing days
```

When `is_partial=True`, the **online provider** (not the batch pipeline) fills the remainder of the season with the climatology daily means before computing features. This is documented in `references/provider_blueprint.md` — keep the partial-handling logic in the provider, not in `compute_climate_features`, so the pure function stays pure and the batch pipeline never returns climatology-filled values.

Set `is_partial=True` as an extra feature column in the provider's output. The CatBoost training set has `is_partial=False` everywhere (we only train on completed vintages), so at inference the model sees the column at its training-time value for full vintages and at `True` for in-progress ones. Whether to feed `is_partial` to the model or hide it from training depends on whether you want uncertainty signal — see Phase 4 discussion in PROJECT.md.
