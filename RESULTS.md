# Vininator 3000 — Results

> **Status:** placeholder. Phase 4 (modeling) is still in progress. This file is regenerated end-to-end by `scripts/build_results.py` once the trained models, ablation runs, and recommender outputs exist. The structure below is the contract — sections will be populated with real numbers, tables, and figures as each phase completes. Do not hand-edit; edit the generator.

---

## 1. Headline result

*To be filled by `scripts/build_results.py`.*

One paragraph: did the terroir block (NASA POWER climate + SoilGrids soil) improve rating prediction over a producer + region + grape + price + `age_at_review` baseline, by how much (RMSE delta on the future-vintage split), and what's the honest takeaway. Reported even if the answer is "no meaningful improvement" — see [PROJECT.md §7](./PROJECT.md#7-realistic-things-to-know) on framing.

---

## 2. Setup

- **Dataset variant:** *(test | slim | full — filled at generation)*
- **Train / test / future-vintage split sizes:** *(filled)*
- **Seed:** *(filled)*
- **Git SHA:** *(filled)*
- **Experiment tracking run:** *(MLflow / W&B link, filled)*
- **Reproduction command:** see [§9](#9-reproduction).

---

## 3. Rating model

### 3.1 Held-out wines (random `WineID` split)

*Table: RMSE + MAE for the trained model and the five baselines from PROJECT.md Phase 4 — global mean, per-`WineryID` mean, per-`(WineID, Vintage)` mean (in-sample only), per-`(RegionName, Vintage)` mean, per-`(GrapeMajority, RegionName)` mean.*

### 3.2 Future-vintage holdout (train ≤ 2018, test 2019–2021)

*Same metrics, on the harder split. This is the number that says whether the model learned terroir or memorized region averages.*

### 3.3 Ablations

*Table with rows = ablated block (none / − terroir / − producer / − price / − `age_at_review`) and columns = RMSE on each split, plus the delta against the full model.*

### 3.4 Confidence intervals

*Per-prediction lo / hi from the quantile heads, summarized as coverage on the held-out set.*

---

## 4. Profile + Harmonize models

### 4.1 Body

*Confusion matrix and per-class F1 against the 5 X-Wines Body classes. Macro-F1 reported headline, not accuracy — the class skew (44% Full-bodied) makes accuracy uninformative.*

### 4.2 Acidity

*Same, for the 3 Acidity classes. Even more skewed (79% High) — class-weighted training compared against unweighted.*

### 4.3 Harmonize food-pairings

*Per-label F1 across the top-N Harmonize pairings, plus Hamming loss. A handful of example wines with their predicted vs. actual pairing vectors.*

---

## 5. SHAP analysis

*Top-20 features by mean absolute SHAP on the rating model, plus 3–4 dependence plots for the most interesting terroir variables (candidates: GDD, harvest-month precip, calcareous flag, diurnal range). Figures live under `reports/figures/` and are embedded here at generation time.*

---

## 6. Drink-now and age-well rankings

### 6.1 Drink-now (opening year 2026)

For each major grape, the top-N monogrape wines predicted to drink best in 2026. Default filter: `--max-vintage-age 5` (fresh-style only) for whites + aromatic reds; no age cap for cellar-style reds.

*Tables per grape (Cabernet Sauvignon, Pinot Noir, Chardonnay, Riesling, Nebbiolo, Tempranillo, Syrah, Sangiovese — list finalised at generation), each with columns: WineryName, WineName, RegionName, Vintage, predicted_rating, confidence band. CLI command that produced each table cited above it.*

### 6.2 Age-well (opening years 2026 → 2036)

For each major grape, the top-N monogrape wines whose predicted-rating trajectory still rises or peaks late within the 10-year horizon.

*Tables per grape with columns: WineryName, WineName, RegionName, Vintage, predicted_peak_year, predicted_peak_rating, slope_to_peak. Rows where `age_at_review` had to be clipped to the training range are flagged.*

### 6.3 Caveats

- Rankings are conditional on wines *in X-Wines*. Not a ranking of the entire wine world.
- Producer effects dominate — expect lists to skew toward well-rated wineries. That's signal, not bug, but worth knowing.
- Aged-wine projections beyond ~10 years post-vintage are extrapolation; clipped rows are flagged.
- Climate is region-centroid, not vineyard-parcel. See the disclaimer block in the README for the full list of scoping decisions.

---

## 7. Qualitative sanity check

*The 10 known-wines exercise from Phase 5: hand-picked wines I personally know, model's predictions for rating / Body / Acidity / pairings, and short commentary on agreements and disagreements. The disagreements are the interesting half.*

---

## 8. Limitations & caveats

Pulled from the README disclaimer block plus model-specific issues surfaced during evaluation. Generated at build time so the list stays in sync with the README.

---

## 9. Reproduction

Exact CLI sequence to rebuild every artifact in this document from a fresh clone. Includes data download, geocoding, NASA POWER + SoilGrids pulls, feature assembly, model training, ablations, recommender runs, and the final `scripts/build_results.py` invocation that emits this file.

---

## Acknowledgements

- **X-Wines dataset** — Xavier 2023, MDPI BDCC. CC0 1.0.
- **NASA POWER** — LaRC POWER Project. Underlying: MERRA-2 + CERES SYN1DEG.
- **SoilGrids** — ISRIC. Hengl et al., 2021.
