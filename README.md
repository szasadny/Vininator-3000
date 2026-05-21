# 🍷 Vininator 3000

> Predict wine ratings, structure, and tasting characteristics from wine metadata and vintage-specific terroir conditions.

A machine learning project that combines the [X-Wines](https://github.com/rogerioxavier/X-Wines) dataset with [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) growing-season weather data and [SoilGrids](https://soilgrids.org/) soil and terrain data to predict:

- **Rating** — the 1–5 star score a wine is likely to receive
- **Profile** — body, acidity, and tannin levels (Body / Acidity columns ship in X-Wines)
- **Tasting notes** — retrieved similar real reviews + flavor descriptors (food pairings ship as the `Harmonize` column)

The project evaluates how much vintage-specific climate features and region-level soil composition contribute to predicting a wine's sensory profile beyond producer, region, grape, price, and `age_at_review`.

---

## Status

🚧 **In development.** Phase 2. See [PROJECT.md](./PROJECT.md) for the full plan and current phase.

---

## Quickstart

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- A [Copernicus CDS](https://cds.climate.copernicus.eu/) account + API key (free; needed for ERA5 weather pulls in Phase 2)
- Node 20+ and pnpm (only if you want to run the web UI)

### Setup

```bash
# Clone and install Python deps
git clone <repo-url> vininator
cd vininator
uv sync --extra notebook

# Configure
cp .env.example .env
# Edit .env if you want a non-default data dir or X-Wines variant

# Download X-Wines (test variant = 100 wines / 1k ratings, auto-fetched from GitHub)
uv run vininator data download

# Run the EDA notebook
uv run jupyter lab notebooks/01_eda.ipynb
```

To use a larger variant, download from the [X-Wines Google Drive](https://github.com/rogerioxavier/X-Wines#-availability):

```bash
# Drop XWines_Slim_1K_wines.csv and XWines_Slim_150K_ratings.csv into data/raw/
echo "VININATOR_XWINES_VARIANT=slim" >> .env
uv run vininator data download
```

### Running the full pipeline

```bash
# 1. Geocode regions (cached; first run ~30min for the full variant)
uv run vininator features geocode

# 2. Pull ERA5 weather (cached, resumable; first run can take hours-to-a-day)
uv run vininator features climate

# 3. Pull SoilGrids soil & terrain (cached, resumable; fast)
uv run vininator features soil

# 4. Assemble the training table
uv run vininator features build

# 5. Train the rating model
uv run vininator train rating --config configs/rating_v1.yaml

# 6. Measure each feature block's contribution by retraining with it dropped
uv run vininator eval ablations
```

### Web UI (local)

```bash
# Backend
uv run vininator api serve   # http://localhost:8000

# Frontend (in another terminal)
cd frontend
pnpm install
pnpm dev                     # http://localhost:5173
```

---

## Project layout

```text
src/vininator/
  data/         X-Wines loader, geocoding (cached)
  features/     Climate (ERA5), soil & terrain (SoilGrids + DEM), terroir joiner, text parsing, feature assembly
  models/       CatBoost rating + profile, flavor-tag multi-label, retrieval-based tasting notes
  eval/         Metrics, ablations, SHAP
  api/          FastAPI service + cache-first terroir provider for live vintages
  cli.py        Typer CLI entrypoint

frontend/        React + Vite + TypeScript + Tailwind
data/            raw → interim → processed (never edited after write)
notebooks/       01_eda, 02_climate, 03_soil, 04_rating, 05_tags, 06_ablations
configs/         One YAML per experiment
deploy/          fly.toml / Dockerfile / Pages config / cron warmer
tests/           pytest suite
```

Full structure and rationale: [PROJECT.md](./PROJECT.md). Working conventions: [CLAUDE.md](./CLAUDE.md).

---

## Data

**Primary dataset:** [X-Wines](https://github.com/rogerioxavier/X-Wines) (Xavier 2023, MDPI BDCC). The full variant covers **100,646 wines** and **21,013,536 ratings** from 2012–2021, spanning 62 wine-producing countries.

Three variants are supported via the `VININATOR_XWINES_VARIANT` env var:

| variant | wines | ratings | source |
| --- | ---: | ---: | --- |
| `test` | 100 | 1,000 | GitHub raw (auto-fetched) |
| `slim` | 1,007 | 150,000 | Google Drive (manual drop) |
| `full` | 100,646 | 21,013,536 | Google Drive (manual drop) |

Each rating ships with **`Date`** (ISO timestamp) plus the rated **`Vintage`**, so the loader derives a per-row **`age_at_review = year(Date) - Vintage`** during normalization.

**License:** **CC0 1.0** — public domain dedication. No usage restrictions.

**Weather (Phase 2):** ERA5 daily reanalysis pulled per `(region, vintage_year)` via the Copernicus CDS API. Free, requires registration.

**Soil & terrain (Phase 2):** [SoilGrids](https://soilgrids.org/) (ISRIC) for topsoil composition — calcium carbonate (kalkgehalte), pH, texture, organic carbon, CEC, bulk density — plus SRTM elevation and derived slope. Pulled per region centroid, no auth required.

---

## Key design choices

- **Split by `WineID`, not by rating.** Same wine in train and test is leakage.
- **Future-vintage holdout** (train ≤ 2018, test 2019–2021) tests whether the model genuinely learned terroir vs. memorized region averages.
- **`age_at_review`** is a real per-row feature, derived from `Date` and `Vintage` at load time.
- **Producer (`WineryID`) aggregates computed on training fold only.** Standard target-leakage prevention.
- **CatBoost over manual encoding.** High-cardinality categoricals (`WineryID`, `RegionName`) handled natively.
- **Retrieval only for tasting notes.** Notes are surfaced by retrieving the most similar real reviews — no generative model, no hallucinated flavor descriptions.
- **Cache-first live serving.** The model is trained on historical vintages, but the API can answer for any `(region, vintage_year)` — including this year's — by transparently fetching and caching the missing terroir data.

---

## Evaluation

- **Rating:** RMSE + MAE on held-out wines and on the future-vintage split.
- **Profile:** per-attribute accuracy and macro-F1 against the X-Wines Body / Acidity labels.
- **Flavor tags:** per-label F1 + Jaccard similarity (with attention to rarer tokens, not just "cherry" and "oak").
- **Ablations:** drop terroir, drop producer, drop `age_at_review` — quantify each block's marginal contribution.
- **SHAP** on the rating model to understand what's actually doing the work.

---

## Deployment

The hosted web app is built for cheap, scale-to-zero operation:

| Layer | Service | Notes |
| --- | --- | --- |
| Frontend | Cloudflare Pages (or Vercel / Netlify) | Static React build, global CDN, free tier |
| Backend | Fly.io tiny VM **or** HuggingFace Spaces (free CPU) | Keeps the CatBoost models in RAM, scales to zero when idle |
| Terroir cache | Cloudflare R2 / Backblaze B2 + on-VM SQLite + in-process LRU | Three-tier cache so live requests hit hot storage |

The user-facing form takes **vintage year**, **year of opening**, **region**, **brand**, and **composition** (grape varieties + percentage shares). For any `(region, vintage_year)` the trained model has not seen, the backend's `TerroirProvider` fetches ERA5 + SoilGrids on demand, stores the result in R2/SQLite/LRU, and serves every subsequent request from cache. A small daily cron warms the cache for popular regions and the latest closed vintages.

---

## Acknowledgements

- **X-Wines dataset** — Xavier, R. de A. M. *X-Wines: A Wine Dataset for Recommender Systems and Machine Learning.* Big Data and Cognitive Computing, MDPI, 2023. [Paper](https://www.mdpi.com/2504-2289/7/1/20) · [Repo](https://github.com/rogerioxavier/X-Wines)
- **ERA5 reanalysis** — Copernicus Climate Change Service (C3S) Climate Data Store.
- **SoilGrids** — ISRIC — World Soil Information. Hengl et al., *SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty.* SOIL, 2021. [soilgrids.org](https://soilgrids.org/)

---

## License

Code: MIT (see `LICENSE`).
Data: X-Wines is CC0 1.0 (public domain).
