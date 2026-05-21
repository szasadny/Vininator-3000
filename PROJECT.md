# Vininator 3000 — Project Plan

A wine rating + tasting-note predictor trained on the X-Wines dataset, with a terroir feature pipeline combining ERA5 climate reanalysis with SoilGrids soil and terrain properties.

---

## 1. Goals

Three prediction targets, in increasing order of difficulty:

1. **Rating** — regression on the 1–5 X-Wines score
2. **Structured profile** — body, acidity (X-Wines ships these as labels) + alcohol % (`ABV`)
3. **Tasting notes** — retrieve the most similar real-review text and a multi-label set of flavor descriptors (X-Wines ships food pairings as the `Harmonize` column)

**Inputs:** grape variety/blend, region (hierarchical), per-rating vintage year, producer (`WineryID`), `age_at_review`, plus a derived **terroir feature block** that combines `region × vintage` weather from ERA5 with `region`-level soil and terrain properties from SoilGrids.

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
        ├─► geocoded regions ──┬─► ERA5 weather pull ──► climate.parquet
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
                         └─► TerroirProvider (cache-first ERA5 + SoilGrids
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

**Weather (ERA5 reanalysis via Copernicus CDS)**

**Strategy: per-region multi-year daily pull.** One CDS request per region covering all vintages (1950–2021) × growing season × daily resolution. **2,160 requests total** (one per unique `RegionName`), one-time, cached to disk. Working set after the eligible-cell filter (`>= 5 wines per (region, vintage)`) is 18,478 cells — same data, just sliced differently downstream.

**Variables pulled per region:**

| CDS variable | Used for |
| --- | --- |
| `2m_temperature` (daily mean) | GDD, anomalies |
| `2m_temperature` (daily max) | Heat-spike days, diurnal range |
| `2m_temperature` (daily min) | Spring frost days, diurnal range |
| `total_precipitation` (daily sum, mm) | Growing-season + harvest-month precip |
| `surface_solar_radiation_downwards` (daily sum, J/m²) | Sunshine / cloud-cover proxy |

Use the derived dataset `reanalysis-era5-single-levels-daily-statistics` (it ships daily min/max/mean directly — no client-side hourly→daily aggregation). Fall back to `reanalysis-era5-land` for high-altitude or coastal regions if the parent grid is too coarse.

**Growing season mask** (applied after pulling the full year so we can recompute thresholds later without re-fetching):

- Northern hemisphere (Country in Europe/USA/Canada/Asia): **April–October**.
- Southern hemisphere (Country in Argentina/Chile/Australia/New Zealand/South Africa/Brazil): **October (prev. year) – April**.

**Features computed per `(region, vintage)`:**

- **Growing Degree Days (GDD)** — `Σ max(Tmean_daily − 10°C, 0)` across the season.
- **Total growing-season precipitation** (mm).
- **Harvest-month precipitation** (last 30 days of the growing-season window).
- **Heat spike days** — count of days `Tmax > 35°C`.
- **Spring frost days** — count of April–May (or Oct–Nov SH) days `Tmin < 0°C`.
- **Mean diurnal temperature range** — average of `(Tmax − Tmin)` across the season.
- **Mean daily solar radiation** — season-average J/m².
- **Anomaly vs. 30-year regional climatology** for each of the above. The 30-year baseline per region is computed once from the same per-region pull (no extra CDS calls). A hot year in Bordeaux means something different from a hot year in Mendoza, so the anomaly often predicts better than the absolute value.

**CDS compliance & politeness (non-negotiable)**

- **License acceptance.** The CDS account must accept the ERA5 product licence in the web UI once before the API will serve data — the loader prints the licence URL on first 403 and stops, rather than retrying blindly.
- **Concurrency cap = 8 in-flight requests.** CDS allows up to ~20 but caps at the operator's discretion; staying at 8 leaves headroom for other users and avoids triggering anti-abuse throttling.
- **Inter-submission delay = 250 ms.** Don't fire 8 requests in the same millisecond; stagger submissions.
- **Backoff on 429/5xx.** Exponential backoff (2s, 4s, 8s, …, cap 5 min) with full jitter. **Never retry on 401/403** — those are credential/licence problems, fail loudly.
- **Resume from disk, not memory.** Each region's raw NetCDF is written atomically to `data/raw/era5/<region_id>.nc` (`tmp` → `rename`). On restart, the loader skips any region whose `.nc` already exists and is non-empty. No central state file — the cache *is* the state.
- **Single user agent** identifying the project: `vininator-3000/0.1 ({contact})`, where `{contact}` is supplied via the `VININATOR_NOMINATIM_CONTACT` env var (defaults to the project's public GitHub URL). Helps CDS operators contact us before they ban us. **Never hardcode a personal email here** — this file is committed.
- **Attribution.** Every derived artifact (`climate.parquet`, trained models) gets the Copernicus attribution embedded in its metadata: *"Generated using Copernicus Climate Change Service information [2026]; neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains."*
- **No bulk redistribution.** We ship aggregated features, not the raw NetCDF tiles. (X-Wines is CC0, but ERA5 is CC-BY 4.0 with a redistribution-restricted bulk-data clause — aggregated derivatives are fine, raw tiles are not.)

**Sizing (one-time, end-to-end):** ~2,160 requests × ~5–20 min CDS queue median; at concurrency 8 that's **~3–6 hours wall time** single-machine, resumable. Raw download is ~1 GB NetCDF; aggregated `climate.parquet` lands at ~200 MB full-resolution, ~5 MB after the eligible-cell filter.

Credentials live in `~/.cdsapirc` (the `cdsapi` library's default location); the loader also accepts `CDS_API_URL` / `CDS_API_KEY` env vars for ephemeral / CI contexts.

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

**Deliverable:** `data/interim/climate.parquet` keyed by `(region, vintage_year)`, `data/interim/soil.parquet` keyed by `region`, and a joined `data/interim/terroir.parquet`. Source files: `src/vininator/features/climate.py`, `src/vininator/features/soil.py`, `src/vininator/features/terroir.py` (joiner).

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
- The service resolves the terroir block for the requested `(region, vintage_year)` via a `TerroirProvider` (Phase 7) that hits cache first, fetches live data only on miss.
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
            └─ MISS: fetch ERA5 + SoilGrids + DEM
                       ──► write R2, SQLite, LRU
                       ──► return
```

- The same feature code used in Phase 2 is reused — `features/climate.py` and `features/soil.py` expose pure functions over `(lat, lon, year)` that don't care whether they're called from a batch notebook or from a request handler.
- A request for an unseen `(region, vintage)` pays one slow first call (tens of seconds to a minute for ERA5); every subsequent request for the same key is milliseconds.
- A background warmer can pre-populate the cache for popular regions × the latest closed vintages on a daily cron — keeps the user-facing miss rate near zero.

**Operational notes**

- ERA5 has a ~5-day publication lag and the latest year's growing season may be incomplete — the provider must detect this and fall back to climatology + partial-season anomalies rather than returning nothing.
- SoilGrids is essentially time-invariant; one fetch per region is enough forever.
- Soft TTL: weather entries refresh after 90 days (in case ERA5 backfills), soil entries effectively never expire.
- API keys (CDS) live in the host's secret store, never in the image.

**Deliverables:** `src/vininator/api/terroir_provider.py`, deployment configs (`fly.toml` or `Dockerfile` + Spaces config), a Cloudflare Pages / Vercel project for the frontend, and a small cron entrypoint for cache warming.

---

## 5. Project structure

```
vininator/
├── pyproject.toml          # uv for deps; ruff + pytest configured
├── README.md
├── PROJECT.md              # this file
├── CLAUDE.md               # operating rules for Claude Code
├── data/
│   ├── raw/                # X-Wines CSVs + parquets, ERA5 daily pulls, SoilGrids responses
│   ├── interim/            # geocoded regions, climate.parquet, soil.parquet, terroir.parquet
│   └── processed/          # final feature parquets (train/test/future_vintage)
├── src/vininator/
│   ├── data/
│   │   ├── load.py         # X-Wines loader
│   │   └── geocode.py      # region → lat/lon (cached)
│   ├── features/
│   │   ├── climate.py      # ERA5 → GDD, precip, anomalies
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
│   │   └── terroir_provider.py  # cache-first live ERA5 + SoilGrids fetcher
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
| Weather | `cdsapi` (Copernicus CDS) | Free ERA5 access |
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

- **ERA5 pulls take ~3–6 hours.** Per-region multi-year daily, concurrency capped at 8, exponential backoff on 429/5xx, resume-from-disk. The cache *is* the state — a half-finished run restarts cleanly. ERA5 attribution must be embedded in derived artifacts; raw NetCDF tiles are not for redistribution.
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
