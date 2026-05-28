# Vininator 3000 — Project Plan

A wine rating + tasting-note predictor trained on the X-Wines dataset, with a terroir feature pipeline combining NASA POWER daily climate (MERRA-2 + CERES SYN1DEG) with SoilGrids soil and terrain properties.

---

## 1. Goals

Three prediction targets, in increasing order of difficulty:

1. **Rating** — regression on the 1–5 X-Wines score
2. **Structured profile** — body, acidity (X-Wines ships these as labels) + alcohol % (`ABV`)
3. **Tasting notes** — retrieve the most similar real-review text and a multi-label set of flavor descriptors (X-Wines ships food pairings as the `Harmonize` column)

**Inputs:** grape variety/blend, region (hierarchical), per-rating vintage year, producer (`WineryID`), `age_at_review`, plus a derived **terroir feature block** that combines `region × vintage` weather from NASA POWER (MERRA-2 + CERES SYN1DEG) with `region`-level soil and terrain properties from SoilGrids.

**Core objective:** build a predictive model that estimates a wine's sensory profile — including rating, structure, and tasting characteristics — from structured wine metadata, the bottle age at the moment of rating, and vintage-specific terroir conditions.

---

## 2. Dataset — X-Wines

- **Source:** [`rogerioxavier/X-Wines`](https://github.com/rogerioxavier/X-Wines) on GitHub. The `test` variant ships in-repo (auto-fetched by the loader); `slim` and `full` are on the X-Wines Google Drive (manual download).
- **License:** **CC0 1.0** (public domain). No restrictions.
- **Variants:**
  - `test` — 100 wines / 1,000 ratings (smoke-test only).
  - `slim` — 1,007 wines / 150,000 ratings (good for iteration without paying the full-data cost).
  - `full` — **100,646 wines / 21,013,536 ratings**, 1,056,079 users, 30,510 wineries, 2,160 regions, 62 countries, collected 2012–2021.
- **Wines schema:** `WineID, WineName, Type, Elaborate, Grapes (list[str]), Harmonize (list[str], food pairings), ABV, Body, Acidity, Code, Country, RegionID, RegionName, WineryID, WineryName, Website, Vintages (list[int])`.
- **Ratings schema:** `RatingID, UserID, WineID, Vintage (int), Rating (1–5, half-steps), Date (ISO timestamp)`. The loader derives `age_at_review = year(Date) - Vintage` during normalization.
- **Why this dataset:** every rating ships with an exact timestamp and the rated `Vintage`, so `age_at_review` is a real per-row feature (the headline reason we dropped WineSensed, which has no review timestamps). Full structured attributes are on every wine, not just ~5%.
- **No images** are included — this dataset is metadata-only.

---

## 3. Architecture overview

```
X-Wines CSVs (test | slim | full)
        │
        ▼
  data/raw/xwines_wines.parquet
  data/raw/xwines_ratings.parquet   (with derived `age_at_review`)
        │
        ├─► geocoded regions ──┬─► NASA POWER weather pull ──► climate.parquet
        │                       │                                │
        │                       └─► SoilGrids pull   ──► soil.parquet
        │                                                        │
        │                                          ──► terroir.parquet (climate ⨝ soil)
        │                                                        │
        ▼                                                        ▼
  feature assembly (joins wine + terroir + parsed text features)
        │
        ▼
  data/processed/train.parquet
        │
        ├─► CatBoost rating model
        ├─► CatBoost flavor-tag multilabel
        └─► retrieval-based tasting-notes model
        │
        ▼
  FastAPI backend ──► React + Vite frontend (web UI)
                         │
                         └─► TerroirProvider (cache-first NASA POWER + SoilGrids
                                              for vintages newer than training)
```

The ML side is the primary deliverable. The web UI is a thin wrapper around the prediction API: enter year + year of opening + region + brand + composition, get back predicted rating, profile, and retrieved tasting notes.

---

## 4. Phases

### Phase 1 — Data acquisition & EDA

- Pull X-Wines via `vininator data download` (test variant from GitHub; slim/full are dropped into `data/raw/` after a Google Drive download).
- Loader normalizes both CSVs into parquets, parses `Grapes` and `Vintages` list columns, and derives `age_at_review = year(Date) - Vintage`.
- EDA notebook: schema + missingness on both tables, `age_at_review` distribution (sanity-check negative ages and outliers), rating distribution, ratings-per-year coverage, grape / region / country / winery cardinality, ratings-per-wine distribution, `(RegionName, Vintage)` cells with `>= 5` wines.

**Deliverable:** `notebooks/01_eda.ipynb` + a written summary of what's actually usable.

### Phase 2 — Terroir feature pipeline

**Region → coordinates**
- Geocode each unique region string to lat/lon. Use Nominatim (OSM) or GeoNames. Rate-limit politely (1 req/sec for Nominatim).
- Cache aggressively — geocode each region once, ever. A few thousand unique regions, all cached to `data/interim/geocode.parquet`.
- Imprecise names ("Bordeaux") → use appellation centroid. Accept the lossiness.
- The same `(lat, lon)` feeds both the climate and the soil pipeline below.
- **Audit `result_type` before downstream use.** Nominatim returns the OSM entity type ("administrative", "region", "city", "village", "house", "monument", …). Some X-Wines region strings resolve to the *wrong* entity — e.g. "Buenos Aires, Argentina" → city centre, "Scanderbeg, Albania" → a public square named after the historical figure. These rows pass `status='ok'` but their coordinates don't land in vineyard country, so SoilGrids returns null and the terroir signal is junk. Before any model trains on these features, filter the geocode parquet to keep only `result_type ∈ {administrative, region, county, state, locality, suburb, isolated_dwelling, hamlet, village}` (or a similar whitelist refined against the actual distribution). The soil pull is unaffected — CatBoost handles the nulls — but the filter must happen before training so the bad rows don't enter the training set.

**Weather (NASA POWER Daily API)**

**Strategy: one HTTP request per region.** [NASA POWER](https://power.larc.nasa.gov/) is a free, public-domain REST/JSON endpoint serving daily MERRA-2 meteorology and CERES SYN1DEG solar radiation at ~0.5° / ~55 km resolution, going back to 1981. A single request fetches all of `CLIMATE_YEAR_RANGE` × all five daily variables for one (lat, lon). After the PR2.5 geocode blacklist there are **~1,377 requests total**, cached as one JSON per region under `data/raw/nasa_power/{slug}.json`. No account, no token, no NetCDF parsing.

**Why NASA POWER over Open-Meteo / ERA5-Land?** We initially pivoted *to* Open-Meteo (an ERA5-Land wrapper, 0.1° / ~11 km) because it avoided Copernicus CDS chunking, then pivoted *off* it when the free tier turned out to bill per data point — 30 years × 5 variables × one coordinate exceeded the daily quota in a single region pull. NASA POWER's coarser resolution is the cost of staying free; for the growing-degree-day / heat-spike / harvest-precip aggregates we actually compute, the ~5× resolution downgrade is negligible compared to the centroid-vs-vineyard error already baked into region geocoding. POWER's temperature data is itself derived from MERRA-2 (sibling to ERA5-Land in lineage), so the underlying science isn't a step down — just the grid.

**Variables pulled per region** (NASA POWER variable IDs → our feature math, units verified via live API metadata):

| NASA POWER variable | Units | Used for |
| --- | --- | --- |
| `T2M_MIN` | °C | Spring frost days, diurnal range |
| `T2M` | °C | GDD, climatology baseline, anomalies |
| `T2M_MAX` | °C | Heat-spike days, diurnal range |
| `PRECTOTCORR` | mm/day | Growing-season + harvest-month precip |
| `ALLSKY_SFC_SW_DWN` | MJ/m²/day | Sunshine / cloud-cover proxy |

Units land in exactly the form `compute_climate_features` expects — no unit conversion in the loader. POWER returns `-999.0` for days where the source product was unavailable; `load_nasa_power_daily` coerces that sentinel to null so downstream null-detection (`is_partial`, climatology skip) works the same as it did with Open-Meteo's explicit nulls.

**Growing season mask** (applied at feature time, not at fetch time, so we can recompute thresholds without re-fetching):

- Northern hemisphere (lat ≥ 0): **April–October** of vintage_year.
- Southern hemisphere (lat < 0): **October of (vintage_year − 1) – April** of vintage_year.

**Features computed per `(region, vintage)`:**

- **Growing Degree Days (GDD)** — `Σ max(Tmean_daily − 10°C, 0)` across the season.
- **Total growing-season precipitation** (mm).
- **Harvest-month precipitation** (last 30 days of the growing-season window).
- **Heat spike days** — count of days `Tmax > 35°C`.
- **Spring frost days** — count of April–May (or Oct–Nov SH) days `Tmin < 0°C`.
- **Mean diurnal temperature range** — average of `(Tmax − Tmin)` across the season.
- **Total growing-season solar radiation** (MJ/m²).
- **Anomaly vs. 1991–2018 regional climatology** for each of the above. The baseline window deliberately ends at the training cutoff (2018) rather than the WMO-standard 1991–2020, so the anomaly column carries **zero information leakage** into the 2019–2021 future-vintage holdout. If the climatology used 2019–2020 it would peek at the held-out years and overstate the model's out-of-sample skill on vintages we haven't trained on. The climatology is computed once per region from the same per-region pull (no extra HTTP calls). A hot year in Bordeaux means something different from a hot year in Mendoza, so the anomaly often predicts better than the absolute value.

**Partial-vintage handling.** A growing-season window may extend outside the pulled year range (e.g. a Southern-hemisphere 1991 vintage's October–December 1990 leg falls before 1991) or carry nulls inside the season (POWER's `-999.0` sentinel after parsing, for days where the source product was unavailable). The batch path computes features on whatever days are present and sets `is_partial=True` on the row — both for missing dates and for null `tmean` inside the season. Rows are kept; the downstream model can decide whether to use them. The Phase 7 `TerroirProvider` is the only path that *substitutes* climatology for the partial season — batch never does.

**NASA POWER compliance & politeness**

- **Public domain data.** POWER data are released without licence restrictions. Vininator embeds an acknowledgement of LaRC and the underlying MERRA-2 / CERES SYN1DEG sources in parquet metadata — required courtesy, not a legal blocker.
- **Polite request spacing.** POWER doesn't publish a hard rate limit but the docs warn that hammering the same coordinate triggers opaque blocking. 1 s between submissions keeps the 1,377-region pull at ~23 minutes wall time and well clear of "hammering" territory.
- **Backoff on transient failures.** Exponential backoff (2s, 8s, 30s, 60s, 120s, 300s) on network blips and 5xx. POWER returns no structured `Retry-After`, so every retry uses the same schedule. Persistent failure after all retries raises `NasaPowerError` and stops the run.
- **Resume from disk, not memory.** Each region's JSON is written atomically (`tmp` → `rename`). On restart, the loader skips any region whose `.json` already exists and is non-empty. No central state file — the cache *is* the state.
- **User-Agent** identifies the project: `vininator-3000/0.1`. POWER doesn't require a contactable operator, but identifying ourselves is good manners.
- **Attribution.** Every derived artifact (`climate.parquet`, trained models) gets the POWER attribution embedded in its metadata: *"Weather data from the NASA Langley Research Center POWER Project, funded through the NASA Earth Science Directorate Applied Science Program. Underlying sources: MERRA-2 (meteorology) and CERES SYN1DEG (solar radiation)."*

**Sizing (one-time, end-to-end):** 1,377 HTTP requests, each returning a multi-megabyte JSON spanning 31 years. Aggregated `climate.parquet` is small (one row per `(region, vintage_year)` × 31 years × ~1.4k regions = ~43k rows).

**Soil & terrain (SoilGrids 250m via ISRIC REST API)**

Soil composition is a defining component of terroir and is largely time-invariant — pulled per `region`, not per `(region, vintage)`. Free, no auth required, well-documented.

- For each unique region centroid, query SoilGrids for the topsoil (0–30 cm) profile. Use a small spatial neighbourhood (e.g. 1 km buffer, mean aggregation) so single-pixel noise is smoothed out.
- Extract:
  - **Calcium carbonate content (`CaCO3`, "kalkgehalte")** — the classic chalky-soil signal (Champagne, Chablis, Jerez).
  - **pH (in H2O)** — acidity of the soil itself; correlates with grape acid retention.
  - **Soil texture** — clay, sand, silt percentages. Together they determine drainage and heat retention.
  - **Soil organic carbon (`SOC`)** — proxy for fertility; vines on rich soils tend to over-crop and underperform.
  - **Cation exchange capacity (`CEC`)** — nutrient-holding capacity.
  - **Bulk density** — proxy for compaction and rooting depth.
- Derive a few interpretable composites:
  - **Drainage class** — coarse bucket from sand% and clay% (`sandy`, `loamy`, `clayey`, `chalky` when CaCO3 is dominant).
  - **Calcareous flag** — boolean, CaCO3 above a region-relative threshold.
- Pull **elevation and slope** from a DEM (SRTM 30m via `elevation` or Open-Elevation) at the same centroid. Slope and aspect influence sun exposure and frost drainage; elevation correlates with diurnal range.
- Cache raw SoilGrids responses per region to `data/interim/soil_raw/`. The API is friendly but flaky — resume must work.

**Deliverable:** `data/interim/climate.parquet` keyed by `(region, vintage_year)`, `data/interim/soil.parquet` keyed by `region`, and a joined `data/interim/terroir.parquet`. Source files: `src/vininator/features/climate.py` (NASA POWER), `src/vininator/features/soil.py`, `src/vininator/features/terroir.py` (joiner).

### Phase 3 — Feature engineering & dataset assembly

Build the final training table with these blocks:

**Wine identity**
- Grape variety (high cardinality; for blends, multi-hot encode top ~50 grapes + "other")
- Producer/winery (very high cardinality — let CatBoost handle natively, or target-encode)
- Country, region, sub-region, appellation (hierarchical categoricals)
- Wine type (red/white/rosé/sparkling/dessert/fortified)

**Wine context**
- Vintage year (int + binned decade for sparser data)
- Price (log-transform; impute missing with median per `region × grape`)
- Alcohol %

**Climate block** (from Phase 2) — six absolute metrics + six anomalies = ~12 numerical features, joined on `(region, vintage_year)`.

**Soil & terrain block** (from Phase 2) — CaCO3, pH, clay/sand/silt %, organic carbon, CEC, bulk density, elevation, slope + derived `drainage_class` (categorical) and `calcareous` (boolean), joined on `region` only. Treat as static per region.

**Text-derived features** (from review aggregation)
- Body, acidity, tannin parsed from standardized phrasing ("med+ acidity", "low tannins"). Regex pass + light cleanup; aggregate per-wine by majority vote across its reviews.
- Review aggregation must be computed after train/test splitting to avoid leakage between wines appearing in multiple reviews.

**Producer aggregates** (carefully, to avoid leakage)
- Producer mean rating, std, n_reviews — computed **on the training fold only**.

**Canonicalization phase**
- Canonicalize producer, grape, and region names to reduce duplication and improve grouping consistency.

**Sample weight:** `log(1 + n_ratings)` per wine.

**Splitting strategy (critical):**

- Split by **`WineID`**, not by review. Same wine in train and test is leakage.
- Held-out test: 15% of wines.
- Additional "future vintage" test: train on vintages ≤ 2018, test on 2019–2021. Reveals whether the model learned terroir or just memorized region averages.
- Use grouped cross-validation by `WineID` during development and hyperparameter tuning.
- Optional: grouped validation by producer or region to test generalization beyond memorized winery/region effects.

**Deliverable:** `data/processed/train.parquet`, `data/processed/test.parquet`, `data/processed/future_vintage_test.parquet`.

### Phase 4 — Modeling

**Rating model — CatBoost regression.** Handles high-cardinality categoricals natively, trains on CPU, is the boring correct choice. Use `age_at_review` as a first-class numerical feature.

**Baselines to beat** (measured on the full X-Wines variant, in-sample / optimistic; out-of-sample on a `WineID` split will be looser):

1. Global mean — **RMSE ~0.74** (= `std(Rating)`, since predicting the mean for everyone gives an RMSE equal to the rating standard deviation; the global mean itself is **3.89**, not 3.5).
2. Per-`WineID` mean — **RMSE ~0.65** in-sample. Unavailable at test time under a wine-level split — replace with per-`WineryID` mean (a leakage-safe proxy) for a held-out baseline.
3. Per-`(WineID, Vintage)` mean — **RMSE ~0.63** in-sample. Same wine-split caveat.
4. Per-`(RegionName, Vintage)` mean — leakage-safe under both wine and future-vintage splits. The number we actually need to beat in production.
5. Per-`(GrapeMajority, RegionName)` mean — same as (4) but stratified by the dominant grape.

The headline experiment is "rating with terroir block" vs. "rating without terroir block", anchored to baselines (1) and (4-5).

Additional modeling:
- Quantile regression for prediction intervals / confidence bands.

**Profile model — CatBoost multi-class.** Classifiers trained against the X-Wines `Body` and `Acidity` labels (no `Tannin` ships in X-Wines — derive from review text in Phase 3 if we want it).

- `Body` — 5 ordinal classes: `{Very light-bodied, Light-bodied, Medium-bodied, Full-bodied, Very full-bodied}`. Heavy skew: 44% Full, 34% Medium, 11% Very Full, 10% Light, 1% Very Light.
- `Acidity` — 3 ordinal classes: `{Low, Medium, High}`. **Very heavy skew: 79% High, 18% Medium, 3% Low** — single-class baseline already gets ~79% accuracy; report macro-F1, not accuracy, and consider class weights.

Same feature set as rating.

**Tasting notes model**

1. **Multi-label flavor-tag prediction.** Build a vocabulary of ~150 flavor descriptors (cherry, oak, leather, citrus, tobacco, vanilla, minerality, …) from review-text frequency. Per wine, aggregate descriptors mentioned across its reviews into a binary label vector. Train multi-label CatBoost or small MLP on categorical embeddings. Evaluate per-label F1 + Jaccard similarity.
2. **Retrieval.** Embed all real review texts with `sentence-transformers/all-MiniLM-L6-v2`. Train a model to predict mean embedding from features. At inference, return k-nearest real reviews. Retrieval-based notes are preferred because they preserve realistic wine language and avoid hallucinated flavor descriptions.

### Phase 5 — Evaluation & analysis

- Rating: RMSE + MAE on held-out wines and held-out future vintages.
- Flavor tags: per-label F1, especially on rarer tokens.
- **SHAP / feature importance on the rating model** — what actually matters?
- **Ablations:** drop terroir, drop producer, drop price. Quantify each block's contribution.
- **Qualitative sanity check:** pick 10 wines you personally know, predict ratings + notes, eyeball it.

### Phase 6 — Web UI

Thin wrapper around the model. Backend serves predictions; frontend lets users explore.

**Backend (FastAPI + uvicorn)**

- `POST /predict` — body = `{vintage_year, opening_year, region, brand, composition}`, returns `{rating, profile, flavor_tags, retrieved_notes}`.
  - `composition` is a list of `{grape, share}` objects so blends are first-class.
  - `opening_year` is used to derive `bottle_age = opening_year - vintage_year`, applied at inference as an aging-adjustment feature.
- `GET /regions`, `GET /grapes`, `GET /brands?q=…` — autocomplete endpoints backed by the dataset.
- Models loaded once at startup, kept in memory. Wrap in a `PredictionService`.
- The service resolves the terroir block for the requested `(region, vintage_year)` via a `TerroirProvider` (Phase 7) that hits cache first, fetches live NASA POWER + SoilGrids only on miss.
- No DB needed for v1 — the dataset is read-only, served from parquet.

**Frontend (React + Vite + TypeScript + Tailwind)**

- One-page form with five inputs: **vintage year**, **year of opening**, **region**, **brand**, **composition** (grape varieties + percentage shares).
- Result panel: predicted rating (with confidence band), profile bars (body/acidity/tannin), flavor-tag chips, 3–5 retrieved real reviews from similar wines.
- Optional: map showing the region with that vintage's climate + soil summary.

### Phase 7 — Deployment & live-data cache

The model is trained on historical vintages. A user can — and will — ask about a vintage newer than anything in training. The backend must therefore be able to pull and cache **current** terroir data for any `(region, vintage_year)` it has never seen, without making every cold request slow or expensive.

**Hosting target — cheap and scale-to-zero**

| Layer | Choice | Why |
| --- | --- | --- |
| Frontend | Cloudflare Pages (or Vercel/Netlify) static | Free tier, global CDN, no servers |
| Backend | Fly.io tiny VM with scale-to-zero, **or** HuggingFace Spaces (free CPU) | Stateful enough to hold the CatBoost models in RAM; free or near-free |
| Cache store | Cloudflare R2 / Backblaze B2 (S3-compatible) | Pennies per GB; the terroir cache is small |
| Hot-path cache | In-process LRU + a tiny SQLite on the VM | Avoid hitting object storage on every request |

Picking serverless functions instead is viable, but cold-starting CatBoost on every invocation is more expensive *and* slower than a single small always-warm-when-needed VM. Default to the small-VM path.

**Live terroir data flow**

```
client → /predict
            │
            ▼
     TerroirProvider.get(region, vintage_year)
            │
            ├─ in-process LRU hit? ──► return
            │
            ├─ SQLite cache hit? ────► return (warm LRU)
            │
            ├─ R2 object hit? ───────► return (warm SQLite + LRU)
            │
            └─ MISS: fetch NASA POWER + SoilGrids + DEM
                       ──► write R2, SQLite, LRU
                       ──► return
```

- The same feature code used in Phase 2 is reused — `features/climate.py` and `features/soil.py` expose pure functions over `(lat, lon, year)` that don't care whether they're called from a batch notebook or from a request handler.
- A request for an unseen `(region, vintage)` pays one NASA POWER round-trip (sub-second for a single vintage year); every subsequent request for the same key is milliseconds.
- A background warmer can pre-populate the cache for popular regions × the latest closed vintages on a daily cron — keeps the user-facing miss rate near zero.

**Operational notes**

- NASA POWER's MERRA-2 inputs have a multi-week publication lag and the latest year's growing season may be incomplete — the provider must detect this (null `tmean` inside the season window, after the `-999.0` → null coercion) and fall back to climatology + partial-season anomalies rather than returning nothing.
- SoilGrids is essentially time-invariant; one fetch per region is enough forever.
- Soft TTL: weather entries refresh after 90 days (in case POWER backfills MERRA-2/CERES corrections), soil entries effectively never expire.
- No API keys needed for NASA POWER — public domain, anonymous access. SoilGrids is also auth-free.

**Deliverables:** `src/vininator/api/terroir_provider.py` (cache-first NASA POWER + SoilGrids fetcher), deployment configs (`fly.toml` or `Dockerfile` + Spaces config), a Cloudflare Pages / Vercel project for the frontend, and a small cron entrypoint for cache warming.

---

## 5. Project structure

```
vininator/
├── pyproject.toml          # uv for deps; ruff + pytest configured
├── README.md
├── PROJECT.md              # this file
├── CLAUDE.md               # operating rules for Claude Code
├── data/
│   ├── raw/                # X-Wines CSVs + parquets, NASA POWER JSON pulls, SoilGrids responses
│   ├── interim/            # geocoded regions, climate.parquet, soil.parquet, terroir.parquet
│   └── processed/          # final feature parquets (train/test/future_vintage)
├── src/vininator/
│   ├── data/
│   │   ├── load.py         # X-Wines loader
│   │   └── geocode.py      # region → lat/lon (cached)
│   ├── features/
│   │   ├── climate.py      # NASA POWER → GDD, precip, anomalies
│   │   ├── soil.py         # SoilGrids + DEM → CaCO3, pH, texture, slope, ...
│   │   ├── terroir.py      # join climate + soil into the terroir block
│   │   ├── text.py         # parse body/acidity/tannin/flavors from notes
│   │   └── build.py        # assemble final feature table
│   ├── models/
│   │   ├── rating.py       # CatBoost rating regressor
│   │   ├── profile.py      # body/acidity/tannin classifiers
│   │   ├── tags.py         # multi-label flavor tags
│   │   └── notes.py        # retrieval-based tasting notes
│   ├── eval/
│   │   ├── metrics.py
│   │   └── ablations.py
│   ├── api/                # FastAPI app
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   ├── service.py      # wraps trained models, single source of inference
│   │   └── terroir_provider.py  # cache-first live NASA POWER + SoilGrids fetcher
│   └── cli.py              # typer CLI: vininator train rating, etc.
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── lib/            # api client, constants
│       ├── types/          # shared TS types (mirror API schemas)
│       ├── components/
│       └── pages/
├── notebooks/              # 01_eda, 02_climate, 03_soil, 04_rating, 05_tags, 06_ablations
├── configs/                # yaml per experiment (rating_v1.yaml, etc.)
├── deploy/                 # fly.toml / Dockerfile / Pages config / cron warmer
└── tests/
```

---

## 6. Stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python 3.12 | Standard for ML |
| Env / deps | `uv` | Fast, modern, lockfiles work |
| Data wrangling | `polars` | Faster than pandas at 800k rows; lazy is nice |
| Modeling | `catboost` | Native categorical support; no manual encoding |
| Text | `sentence-transformers`, `re` | MiniLM for embeddings; regex for body/acidity parsing |
| Weather | NASA POWER Daily API | MERRA-2 + CERES SYN1DEG via clean JSON REST, no auth |
| Soil | SoilGrids REST API (ISRIC) | Free, no auth, global 250 m coverage |
| Terrain | SRTM 30 m via `elevation` or Open-Elevation | Free; elevation + slope per centroid |
| Geocoding | `geopy` (Nominatim) | Free; respect rate limits |
| Tracking | `mlflow` *or* `wandb` | Choose one; track from day one |
| API | FastAPI + uvicorn | Async, auto-OpenAPI, easy frontend integration |
| Frontend | React + Vite + TS + Tailwind | Familiar; quick to ship |
| Hosting | Cloudflare Pages (frontend) + Fly.io / HF Spaces (backend) + R2 (cache) | Cheap, scale-to-zero, generous free tiers |
| Lint/test | ruff + pytest + pytest-asyncio | Standard |

---

## 7. Realistic things to know

- **NASA POWER pulls take ~23 minutes sequentially.** One HTTP request per region (~1,377 total after the geocode blacklist), polite 1s spacing, exponential backoff on 5xx, resume-from-disk per JSON file. The cache *is* the state — a half-finished run restarts cleanly. Attribution must be embedded in derived artifacts (LaRC POWER + MERRA-2 + CERES SYN1DEG lineage). Public domain, no auth, no quota. We pivoted off Open-Meteo's ERA5-Land wrapper after discovering its free tier bills per data point — 30 years × 5 vars × one coordinate exceeded the daily quota in a single region pull. POWER's ~55 km cells are coarser than ERA5-Land's ~11 km, but for growing-season aggregates the difference is negligible next to the centroid-vs-vineyard error.
- **Climatology window is 1991–2018, not the WMO-standard 1991–2020.** The baseline used to compute climate anomalies ends at the training cutoff so the anomaly column contains zero information about the 2019–2021 future-vintage holdout. Reporting that "the model generalizes to future vintages" would be a lie if the anomalies it trained on already peeked at those vintages.
- **SoilGrids is fast but flaky.** Single-pixel queries can be noisy and the endpoint occasionally 5xxs. Always buffer-and-average, always retry with backoff, always cache.
- **Geocoding has rate limits.** Nominatim asks for 1 req/sec. A few thousand regions is fine, just plan for it.
- **Geocode `result_type` lies sometimes.** Nominatim happily resolves a wine-region string to a city, monument, or random POI when the appellation isn't in OSM under that exact name. The resulting `status='ok'` row points at the wrong place, and SoilGrids returns null on top of that wrong location. Audit the `result_type` distribution after the geocode pull and filter to a known-good whitelist (administrative / region / locality / hamlet / etc.) before training. Examples encountered in PR2 smoke: "Buenos Aires" → `city`, "Scanderbeg" → `square`.
- **The full X-Wines variant is 21M ratings.** Plenty of data, but for iteration always work off the `slim` variant (150k ratings) — full is for the final training run.
- **Producer (`WineryID`) will dominate everything.** Be ready for the result that terroir adds a few percent RMSE improvement on top of producer + region + grape + price. That's still a real, interesting result — just not the headline "weather predicts wine" story. Frame the project honestly around this from the start.
- **Coverage is skewed toward popular regions.** The model will be best at well-represented regions and worse at obscure ones. Check and report this explicitly.
- **Splits matter.** Split by `WineID`, not by rating. Future-vintage split reveals real terroir learning vs. memorization.
- **`age_at_review` is real but bounded.** `Date` covers 2012-2021, so ratings of pre-2012 vintages are over-represented at high ages; ratings of recent vintages are absent at high ages. Treat `age_at_review` as a feature, not a target.
- **License.** X-Wines is CC0 (public domain). No restrictions on intermediate artifacts or trained models.
- **Live serving needs a fetch-and-cache layer.** The model is trained on historical wines, but the UI must answer for any `(region, vintage_year)` — including this year's. The `TerroirProvider` in Phase 7 must own that fetch-on-miss path so the rest of the codebase can stay batch-flavoured.
