# Vininator 3000 — Wine Rating & Tasting-Notes Predictor

## Domain

ML project that predicts Vivino-style wine ratings and tasting notes from grape, region, vintage, and producer — augmented with a **terroir feature block** that combines ERA5 climate reanalysis per `(region, vintage_year)` with SoilGrids soil/terrain properties per `region`.

Primary dataset: **WineSensed** (`Dakhoo/L2T-NeurIPS-2023` on HuggingFace, CC BY-NC-ND 4.0). ~824k Vivino reviews, ~40k with full structured attributes (the real working set).

For the full plan, phases, and sequencing, see [PROJECT.md](./PROJECT.md). That document is the source of truth for what we're building and in what order. This file is the source of truth for *how* we work.

---

## Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.12 |
| Env / deps | `uv` (lockfile committed) |
| Data | `polars` (preferred over pandas for the main tables) |
| ML | `catboost` (primary), `scikit-learn` (utilities), `sentence-transformers` (text embeddings) |
| Weather | `cdsapi` → ERA5 reanalysis (Copernicus CDS) |
| Soil | SoilGrids REST API (ISRIC), no auth |
| Terrain | SRTM 30 m via `elevation` or Open-Elevation |
| Geocoding | `geopy` (Nominatim) |
| Experiment tracking | `mlflow` *or* `wandb` — pick one in week 1, stick with it |
| API | FastAPI + uvicorn |
| Frontend | React + TypeScript + Vite + Tailwind |
| Hosting | Cloudflare Pages (static frontend) + Fly.io / HF Spaces (backend) + R2 / B2 (terroir cache) |
| Lint / test | ruff + pytest + pytest-asyncio |

---

## Project Structure

```text
src/vininator/
  data/         # WineSensed loader, geocoding (cached, resumable)
  features/     # climate.py (ERA5 → GDD/precip/anomalies), soil.py, terroir.py (joiner), text.py (parse body/acidity/flavors), build.py (assemble final table)
  models/       # rating.py, profile.py, tags.py, notes.py — one file per target (notes is retrieval-only, no generative)
  eval/         # metrics, ablations, SHAP
  api/          # FastAPI app (main, routes, schemas, service, terroir_provider — cache-first live ERA5 + SoilGrids fetch)
  cli.py        # typer CLI entrypoint: `vininator train rating`, etc.

frontend/src/
  lib/          # api client, app-wide constants
  types/        # shared TS types — mirror API schemas, single source of truth
  components/   # small, composable, no business logic
  pages/        # one folder per route

data/
  raw/          # WineSensed dump, ERA5 daily pulls, SoilGrids responses — never modified after write
  interim/      # geocoded regions, climate.parquet, soil.parquet, terroir.parquet
  processed/    # final feature parquets (train/test/future_vintage)

notebooks/      # 01_eda, 02_climate, 03_soil, 04_rating_baseline, 05_rating_terroir, 06_tags, 07_ablations
configs/        # yaml per experiment
deploy/         # fly.toml / Dockerfile / Pages config / cron warmer
tests/
```

**Navigation rule:** when working on a task, read only the folder relevant to that task. Grep before scanning. Notebooks are for exploration; production code lives in `src/vininator/`.

---

## Conventions

**Python**
- Ruff (formatter + linter). Type hints everywhere. `from __future__ import annotations` at the top of every module.
- All public functions get docstrings; explain *why*, not *what*.
- Pydantic v2 for API schemas. Plain dataclasses or `TypedDict` for internal config.
- `async def` for API routes and any I/O-bound work; sync is fine for CPU-bound ML code.
- Pathlib only — never `os.path.join`.
- No hardcoded paths. All paths come from `src/vininator/config.py` (which reads env vars with defaults).

**Frontend**
- Strict TS, no `any`. If you reach for `any`, fix the type instead.
- Functional components + hooks. Components stay small.
- API client generated from FastAPI's OpenAPI spec (orval or openapi-typescript) once the spec stabilises — until then, a thin hand-written client in `lib/api.ts`.
- All labels, durations, role names, and other string constants live in `lib/constants.ts`. Never inline.

**General**
- `.env` for local config; never committed.
- No commented-out code, no dead code, no `TODO`-as-placeholder. If it's not done, leave a real comment explaining what and why.

---

## ML & Data Standards

These are the rules that protect the *headline result*. They are non-negotiable.

- **Split by `wine_id`, not by review.** Same wine in train and test is leakage. Every split function in `src/vininator/data/` must enforce this.
- **Future-vintage holdout.** In addition to the random wine-id split, hold out vintages 2019–2021 as a separate test set. This is how we tell whether the model learned terroir or just memorized region averages.
- **No target leakage in producer aggregates.** Producer mean-rating / std / n_reviews features are computed **on the training fold only**, then applied to test. Never compute on the full dataset.
- **Cache every external call.** ERA5 and Nominatim are slow, rate-limited, and intermittently fail. Every external fetch goes through a function that checks a parquet/sqlite cache first, writes the result, and is resumable across restarts.
- **Raw data is immutable.** Files in `data/raw/` are never modified after write. Cleaning and joining happen on the way to `data/interim/` and `data/processed/`.
- **Track every experiment.** MLflow/W&B from run #1. Hyperparameters, dataset hash, git SHA, metrics, feature list — all logged. "I'll start tracking once it works" never happens.
- **Report ablations honestly.** The headline experiment compares rating-with-terroir vs. rating-without. If terroir adds 1% RMSE, that's the result — don't bury it.
- **Sample weighting.** Use `log(1 + n_ratings)` per wine. A wine with 5000 ratings is a different signal than a wine with 5.
- **Live serving needs the same features as training.** The model is trained on historical vintages but the API must answer for any `(region, vintage_year)` — including this year's. All on-demand terroir fetches go through `api/terroir_provider.py`, which is cache-first (in-process LRU → SQLite → R2/B2 → live ERA5 + SoilGrids). Inference paths never reach the upstream APIs directly.

---

## External Solutions First

Before implementing something in-house, check whether a stable, maintained library already solves it.

- **Boosted trees** → `catboost`. Native categoricals (don't manually target-encode).
- **Text embeddings** → `sentence-transformers`. Don't train your own.
- **Weather data** → `cdsapi` against ERA5. Don't scrape weather sites.
- **Geocoding** → `geopy` with Nominatim. Don't write a CSV of regions by hand.
- **Experiment tracking** → MLflow or W&B. Don't roll your own logging.
- **API framework** → FastAPI. Don't hand-write a Flask service.
- **General rule:** if a maintained PyPI/npm package solves ≥80% of the problem, use it. Reinventing is more bugs and more maintenance.

---

## Maintainability

Write code for the developer maintaining it 12 months from now.

- **No magic values.** Thresholds, paths, hyperparameter defaults, growing-season month ranges, the flavor-tag vocabulary — all live in `src/vininator/config.py` or a yaml in `configs/`. Frontend constants live in `frontend/src/lib/constants.ts`.
- **Single source per piece of behaviour.** Splitting logic, feature assembly, model loading — each defined once and reused. If you find yourself writing the same block in a second file, lift it.
- **Layering.**
  - `data/` → reads raw, returns dataframes. No feature engineering.
  - `features/` → takes dataframes, returns dataframes with new columns. No model training.
  - `models/` → takes processed dataframes, returns trained model artifacts + metrics. No file I/O outside the canonical paths.
  - `api/` → loads model artifacts at startup, serves predictions. Never re-trains, never re-engineers features inline.
  - `cli.py` → the only place that orchestrates phases end-to-end.
- **Notebooks are not production.** Exploration lives in `notebooks/`. Once a finding is real, the code moves into `src/vininator/`. Notebooks may import from the package but the package never imports from a notebook.
- **Think at scale.** Don't `pd.read_csv` the 800k-row WineSensed file then filter; use polars lazy + `scan_parquet` + predicate pushdown. The dataset is small enough to fit in RAM but big enough that lazy beats eager every time.
- **Reproducibility.** Set seeds. Log dataset hashes. The train script should produce the same metrics on a fresh checkout given the same config.
- **Explicit over clever.** A longer, obvious implementation beats a one-liner that requires context.
- **No half-finished features.** Leave code in the last working state. No disabled blocks, no broken branches in main.

---

## Working Approach

**Before writing:**

- Read PROJECT.md if you don't remember the phase you're in or what comes next.
- Read only the files you'll touch, plus their direct imports.
- Grep for the existing pattern before writing new code — match it exactly.
- If a stable external library solves the problem, prefer it.
- **Ask when genuinely split.** If two architecturally sound options exist with real trade-offs (e.g., "store the cache as parquet or sqlite"), present them and ask. Don't pick arbitrarily.
- **Ask before assuming on external resources.** If the task needs a CDS API key, HuggingFace token, or a downloaded artifact that isn't in the repo, stop and ask — don't write code that silently fails on a missing credential.

**While writing:**

- Scope changes tightly — a bug fix changes the bug, a feature adds the feature.
- Check for existing abstractions before building new ones.
- Flag observed debt in your response; don't silently fix it.
- ML rules never relax for "just to see if it works." No leakage shortcuts, no commented-out splits.

**After writing:**

- Run `ruff check` and `pytest` before declaring done.
- Verify importing modules still resolve.
- **Turn manual checks into tests.** If you verified something by running a script and eyeballing output, capture it as a pytest test — the specific case plus a general test of the surrounding behaviour.

**Maintaining this file:**

- After adding or removing a top-level folder, update the Project Structure section in the same change.
- When a cross-cutting rule changes (new ML standard, new layering boundary), update this file as part of the same change.
- For complex situational context spanning multiple prompts, create `.claude/<topic>.md` and add one reference line here — delete it when no longer relevant.
- Never add changelogs or task notes here; git tracks what changed.
