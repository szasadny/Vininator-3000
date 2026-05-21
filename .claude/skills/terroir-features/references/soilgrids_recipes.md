# SoilGrids + DEM recipes

Implementation reference for `src/vininator/features/soil.py`. SoilGrids is free, no auth, but flaky. Plan for failure.

## SoilGrids v2 endpoint

```
GET https://rest.isric.org/soilgrids/v2.0/properties/query
  ?lat=<lat>&lon=<lon>
  &property=phh2o&property=cec&property=clay&property=sand&property=silt
  &property=soc&property=bdod&property=cfvo
  &depth=0-30cm
  &value=mean
```

Response shape (excerpt):

```json
{
  "properties": {
    "layers": [
      {"name": "phh2o", "depths": [{"label": "0-30cm", "values": {"mean": 71}}]},
      ...
    ]
  }
}
```

Note: `mean` is **scaled**. Every SoilGrids property has a `d_factor` — pH is reported × 10 (so `71` means pH 7.1), CEC × 10, clay/sand/silt × 10, SOC × 10, bdod × 100, cfvo × 10. The unit metadata is in the layer description but it's easier to hardcode the divisors per property because they don't change:

```python
SCALE = {
    "phh2o": 10.0,
    "cec": 10.0,
    "clay": 10.0,
    "sand": 10.0,
    "silt": 10.0,
    "soc": 10.0,
    "bdod": 100.0,
    "cfvo": 10.0,
}
```

Wrong unit handling is the most common SoilGrids bug. Write a test that asserts `pH ∈ [3, 10]` and `clay+sand+silt ≈ 100` for at least one known region — both fail loudly if you forget to divide.

## Buffer averaging

A single 250 m pixel is noisy. Query a 3×3 grid around the centroid and average. The grid spacing of 0.005° ≈ 555 m covers a ~1.6 km × 1.6 km area, which lines up reasonably with the scale of an appellation parcel:

```python
def buffered_query(lat: float, lon: float) -> dict[str, float]:
    offsets = [-0.005, 0.0, 0.005]
    samples = []
    for dlat in offsets:
        for dlon in offsets:
            samples.append(query_soilgrids(lat + dlat, lon + dlon))
    # Each sample is {"phh2o": 71, "cec": 152, ...} (raw, scaled)
    keys = samples[0].keys()
    return {k: mean(s[k] for s in samples if s.get(k) is not None) for k in keys}
```

Average **before** unscaling — they're scaled by the same factor, so order doesn't matter mathematically, but doing it in this order keeps the unscaled raw cache identical across regions and makes debugging easier.

Cache raw responses (one per region centroid, not per buffer point) to `data/interim/soil_raw/{region_slug}.json` — the JSON contains all 9 sub-samples. If you decide to change the buffer policy later, you can re-derive without re-hitting the API.

## Retry policy

Targeted: 3 attempts with exponential backoff (1s, 4s, 16s), only on 5xx and `requests.exceptions.ConnectionError`. **Do not retry on 4xx** — those are real client errors (bad coords, malformed query) and retrying just spams the API.

```python
@retry(
    retry=retry_if_exception_type((RequestException, HTTPError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    reraise=True,
)
def query_soilgrids(lat: float, lon: float) -> dict: ...
```

After three failures, write a `status="error"` row to the cache rather than crashing the batch. Some regions (mid-ocean, polar) genuinely have no soil data; the model should learn that null soil is a real signal.

## Deriving features

```python
def compute_soil_features(raw_buffered: dict[str, float]) -> dict:
    # Unscale
    ph = raw_buffered["phh2o"] / 10.0
    clay = raw_buffered["clay"] / 10.0
    sand = raw_buffered["sand"] / 10.0
    silt = raw_buffered["silt"] / 10.0
    soc = raw_buffered["soc"] / 10.0
    cec = raw_buffered["cec"] / 10.0
    bdod = raw_buffered["bdod"] / 100.0
    cfvo = raw_buffered["cfvo"] / 10.0

    calcareous = ph >= 7.5

    if calcareous:
        drainage = "chalky"
    elif clay >= 40:
        drainage = "clayey"
    elif sand >= 60:
        drainage = "sandy"
    else:
        drainage = "loamy"

    return {
        "ph_h2o": ph,
        "clay_pct": clay,
        "sand_pct": sand,
        "silt_pct": silt,
        "soc_gkg": soc,
        "cec_cmolkg": cec,
        "bdod_kgdm3": bdod,
        "coarse_frag_pct": cfvo,
        "calcareous": calcareous,
        "drainage_class": drainage,
    }
```

### About the CaCO3 proxy

PROJECT.md asks for `CaCO3` as a feature. SoilGrids v2 does **not** ship calcium carbonate as a queryable property — it offers pH, organic carbon, texture, CEC, bulk density, coarse fragments, and that's it. The practical proxy is **soil pH ≥ 7.5**, which captures calcareous parent material at coarse resolution (chalk, limestone, marl all push topsoil basic).

Document this in the module docstring so a future reader who looks for `CaCO3` understands the substitution. If a higher-fidelity carbonate signal becomes critical (e.g., for Champagne / Chablis / Jerez separation), candidates are:

1. **HWSD2** (Harmonized World Soil Database v2) — has `caco3` at 30 arc-second resolution. Heavier to query (file-based), but worth it if the pH proxy underperforms.
2. **WoSIS** point data — high quality but sparse, doesn't cover much of the world.

For v1, ship the pH proxy with a `calcareous` boolean alongside the raw pH and revisit if SHAP shows it's a load-bearing feature.

## DEM — elevation and slope

Two viable backends:

**Option A — Open-Elevation REST** (simpler, online-only):

```python
POST https://api.open-elevation.com/api/v1/lookup
{"locations": [{"latitude": lat0, "longitude": lon0}, ...]}
```

POST a 3×3 grid for each centroid (same buffer as SoilGrids), compute slope from the resulting 9 elevations using Horn's algorithm. Rate-limit to ~1 req/sec, cache per region.

**Option B — `elevation` Python package + SRTM tiles** (heavier, offline):

```python
import elevation
elevation.clip(bounds=(west, south, east, north), output=tile_path)
```

Downloads SRTM 30 m tiles to local disk via GDAL. ~25 MB per 1° × 1° tile. For a few thousand centroids spread globally, you may end up downloading 50–200 tiles totaling a few GB. Worth it if you want offline reproducibility.

Default to A unless the user has explicit reasons to want B (no network at inference, large-scale per-vineyard analysis).

### Horn's algorithm for slope

```python
def horn_slope(z: list[float], cell_m: float = 555.0) -> float:
    """z is a 3x3 elevation grid flattened row-major: [NW, N, NE, W, C, E, SW, S, SE].
    cell_m is the spacing in meters between adjacent grid points (~555 for 0.005°)."""
    dzdx = ((z[2] + 2*z[5] + z[8]) - (z[0] + 2*z[3] + z[6])) / (8 * cell_m)
    dzdy = ((z[6] + 2*z[7] + z[8]) - (z[0] + 2*z[1] + z[2])) / (8 * cell_m)
    return math.degrees(math.atan(math.sqrt(dzdx**2 + dzdy**2)))
```

Aspect (degrees from north, optional) uses `atan2(dzdy, -dzdx)`. Most wine regions are too gently sloped for aspect to matter at the centroid scale; revisit if SHAP says otherwise.

## Caching policy

- Per region, write **one** JSON for SoilGrids and **one** JSON for DEM into `data/interim/soil_raw/` and `data/interim/dem/`.
- The derived features parquet `data/interim/soil.parquet` is a function of these raw caches; recompute it cheap from cache on every pipeline run so changes to `compute_soil_features` propagate without re-hitting the APIs.
- Soil never expires. DEM never expires (it's measuring rocks). Only ERA5 has a TTL.

## Tests

- A frozen SoilGrids response in `tests/features/data/soilgrids_burgundy.json` plus a frozen DEM grid; assert `drainage_class == "chalky"` and `calcareous == True` for that input.
- A round-trip test that confirms `clay_pct + sand_pct + silt_pct ≈ 100 ± 1` on at least three diverse regions (sandy Mendoza, chalky Chablis, clayey Pomerol).
- A retry test using `responses` or `pytest-httpx` that simulates a 5xx-then-200 sequence.
