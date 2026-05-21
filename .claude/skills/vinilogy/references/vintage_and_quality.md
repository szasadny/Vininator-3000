# Vintage, climate, and quality

How weather translates into wine quality, why anomaly features matter, and what a Vivino rating actually measures. Use this when interpreting SHAP outputs on climate features and when explaining model behavior to non-specialists.

## The Winkler / Amerine GDD framework

The Winkler Index (Albert Winkler and Maynard Amerine, UC Davis, 1944) divides growing regions into five zones by accumulated growing-degree days from April 1 to October 31 (Northern hemisphere), with base 10 °C:

| Region | GDD (10 °C base) | Climate | Typical grape suitability |
| --- | --- | --- | --- |
| I | < 1390 (< 2500 °F-days) | Cool | Pinot Noir, Chardonnay, Gewürz, Riesling, sparkling base |
| II | 1390–1670 | Cool to moderate | Cab Franc, Merlot, lighter reds |
| III | 1670–1940 | Moderate | Cab Sauv, Sangiovese, Tempranillo |
| IV | 1940–2220 | Warm | Zinfandel, Grenache, table-wine workhorses |
| V | > 2220 | Hot | Heat-tolerant Mediterranean varieties, bulk wine; quality challenging |

Real-world calibration points:

- **Champagne** ≈ Region I (often borderline marginal in cold years).
- **Burgundy** ≈ Region Ib / II.
- **Bordeaux** ≈ Region II / III.
- **Napa Valley** ≈ Region III on the valley floor, II in cooler pockets, IV in inland sub-AVAs.
- **Barossa, McLaren Vale** ≈ Region IV.
- **Mendoza** ≈ Region IV (but altitude moderates).
- **Central Valley CA, La Mancha, North Africa** ≈ Region V.

The project's `gdd_10c` feature is the Winkler Index. The model can in principle re-derive these zones; humans interpreting SHAP should think in zones, not raw degree-days.

## What makes a great vintage in each climate

The same hot summer can be a triumph in one region and a disaster in another. Generalizing:

**Marginal cool climates** (Burgundy, Champagne, Mosel, Sancerre, Marlborough):

- Heat is usually good — pushes ripeness over the edge from green to fruit.
- Late rain is bad — dilutes flavor, encourages rot at harvest.
- Frost in April–May is catastrophic — wipes out the buds, sometimes the entire vintage.
- Anomaly features dominate. A "warm year" is +200 GDD over local norm; a "cold year" is −150. Both are visible in wine style.

**Moderate climates** (Bordeaux, Tuscany, Rioja, Napa):

- Even ripening is the goal; "warm and dry" is the standard great-vintage shape.
- Late summer rain compresses harvest decisions; some producers can sort/select, others can't.
- Hail is a wildcard — a single afternoon can destroy a producer's vintage but not the region's. Not captured by ERA5; this is one of the noise sources in `gdd_anom` predictions.

**Warm climates** (Barossa, southern Rhône, much of California, Mendoza):

- Excess heat is the new failure mode — alcohol levels balloon, acidity drops, sunburn on the fruit.
- Drought is variable: vines tolerate it well to a point; beyond it the fruit shrivels.
- Diurnal range — `tmax − tmin` averaged across the season — matters more here than GDD. High diurnal range preserves acidity even in hot years; this is why high-altitude regions (Mendoza, Spain's Ribera del Duero, Etna) make balanced wines despite warm summers.

This is why **anomaly features (gdd_anom, precip_anom) outperform absolute values** in tree models trained across regions: a tree split on `gdd_10c > 1800` separates Bordeaux from Mendoza, not good vintages from bad. A split on `gdd_10c_anom > +200` separates *hot for here* from *typical for here*, which is the actual quality lever.

## Specific famous vintages — sanity check fodder

Use these to test the model qualitatively (Phase 5 of PROJECT.md asks you to eyeball 10 wines):

| Region × Vintage | Known character | Expected signal |
| --- | --- | --- |
| Bordeaux 1982 | Hot, ripe, the "modern Bordeaux" benchmark | `gdd_anom` strongly positive |
| Bordeaux 2000 | Classic, even, dry harvest | Slightly above-average GDD, low harvest precip |
| Bordeaux 2003 | Heat dome; very ripe, lower acid, unusual structure | Very high `gdd_anom`, low precip, `heat_spike_days` elevated |
| Bordeaux 2005 | Universally considered great, dry, balanced | Above-avg GDD, low precip |
| Bordeaux 2013 | Cold, rainy, light, dilute | Negative `gdd_anom`, elevated precip |
| Burgundy 2003 | Same heat dome — divisive; some wines flabby | Same climate signal as Bordeaux 2003, region context flips interpretation |
| Burgundy 2004 | Cool, wet, ladybug taint; light wines | Negative `gdd_anom`, elevated precip |
| Burgundy 2005 | Across-the-board great | Above-avg GDD, low precip |
| Champagne 2008 | Cool, classic, age-worthy | Cool with adequate ripening, low harvest precip |
| Napa 2011 | Cold, late, smoky | Negative `gdd_anom`, late harvest precip |
| Napa 2017 | Wildfire smoke (Sonoma worse) | Not visible in ERA5; this is a known unknown |
| Barolo 2010 | Classic, slow, structured | Slightly cooler year |
| Barolo 2017 | Hot, dry | High `gdd_anom`, low precip |
| Rioja 2010 | Universally great | Above-avg GDD, low precip |
| Mendoza | Consistent | Anomalies near zero; absolute GDD high |

If the model predicts wildly differently for these, the terroir block isn't carrying its weight (or there's a bug). If it predicts approximately in line with consensus, the feature engineering is working.

## The Vivino rating distribution

The dataset's primary target is a 1–5 score from Vivino users. Critical facts:

- **Mean ~3.6**, median ~3.6. The distribution is right-shifted because users tend not to rate wines they actively disliked, and the population skews toward enthusiast / repeat-customer behavior.
- **~40% of all wines fall between 3.5 and 3.8** — this is the "fat middle" of the distribution. Any model that mostly predicts in this band gets decent RMSE without doing anything useful.
- **Percentile mapping:**
  - 3.6 ≈ 50th percentile (typical, drinkable, unremarkable).
  - 3.9 ≈ 80th percentile.
  - 4.0 ≈ 85th percentile (notably good).
  - 4.1 ≈ 90th percentile.
  - 4.3 ≈ 95th percentile.
  - 4.5 ≈ 99th percentile (rare and extraordinary).
- **4.0 Vivino ≈ 90 Robert Parker points.** The two systems correlate well at the top end; less well at the bottom (critics don't review wines that would score 80; Vivino users do, sometimes).
- **The headline baseline is ~0.4 RMSE** (global mean predictor on the 1–5 scale). The model needs to push noticeably below that to be doing real work. A reasonable target with terroir + producer + grape + region + price is ~0.30 RMSE; without terroir, ~0.32–0.33. The headline result of the project is the gap.

This distribution shape has model-design consequences:

- **Quantile regression** for confidence bands is valuable because the conditional variance is not constant (rare extreme wines have wider intervals).
- **Sample weighting via `log(1 + n_ratings)`** is essential: a wine with 5000 ratings has converged toward the population mean; a wine with 5 ratings is a noisy single-sample observation. Unweighted RMSE rewards predicting the population mean for noisy wines, which is statistically correct but uninteresting.
- **Future-vintage holdout (2019–2021)** is the real test. Random holdout overrates the model — same producer / region in train. Future-vintage forces extrapolation onto new climate cells.

## Reading SHAP on the rating model

When you compute SHAP for the rating model (Phase 5), interpret feature contributions in this order of expected magnitude:

1. **Producer features** (mean rating, std, n_reviews) — largest. Producer is summarizing the entire right half of the wine production pipeline; expect ±0.3 SHAP impact.
2. **Price** — second. Powerful but endogenous; producers price by reputation.
3. **Grape × region interactions** — large for distinctive combos.
4. **Climate anomalies** (`gdd_anom`, `precip_anom`, `precip_harvest_mm`) — expect ±0.05 to ±0.15 in marginal climates, smaller in stable ones.
5. **Soil features** — expect modest. Soil is region-level (one row of soil per region), so its variance is across regions, not within. SHAP will often attribute soil signal to "region" instead.
6. **`is_partial` / data quality flags** — should be near zero on training data (we filter `is_partial=False`); becomes meaningful only at inference for current vintages.

If climate features dominate over producer, something is wrong — likely the producer encoding is broken, or the dataset is filtered to a single producer.

If soil features are huge SHAP contributors, double-check that the soil features actually vary at the row level — a bug where every wine gets the same default soil row would show up as zero variance and zero SHAP, but a bug where soil is mis-joined could create spurious variance.

## Aging and the bottle-age effect

The frontend lets the user specify `opening_year`. The model uses `bottle_age = opening_year - vintage_year`.

- **Most reviews are young.** Vivino reviewers drink wines 1–5 years from vintage. The model is most accurate in that window.
- **Aged wines change predictably.** Tannin softens, primary fruit fades, tertiary descriptors (leather, tobacco, forest floor, petrol on whites) develop. The model can't predict this from training data alone if the training data is all young wines.
- **For wines designed to age** (high tannin reds, high acid Rieslings, Champagne, Vintage Port), `bottle_age` is genuinely informative. For others (Sauv Blanc, Beaujolais Nouveau, most rosé), `bottle_age > 3` is a negative signal — the wine is past its window.

Implementation guidance for `bottle_age`:

- Treat as a numeric.
- Don't extrapolate too far. If the user asks for `bottle_age = 30` on a Sauvignon Blanc, the model output should be flagged as out-of-distribution. Add a guardrail in the API service.
- Consider a per-(grape × region) "drinking window" lookup table for UX, even if the model itself stays numeric.

## What the model can't see

Be honest with the user when these limitations bite:

- **Smoke taint** (CA wildfires 2017, 2020 in Sonoma/Napa; Aus 2020) — ERA5 doesn't measure PM2.5. Wines from these vintages may be discounted by the market in ways the model can't predict from weather alone.
- **Hail** — local, single-afternoon events. ERA5 grid is too coarse.
- **Phylloxera, mildew, grape pests** — biological, not climatic.
- **Producer-level decisions** — whether to harvest early or late, whether to declassify into a lower-tier wine, whether to add Cab Franc to a Merlot blend — invisible to terroir features. Producer aggregates absorb some of this on average.
- **Bottle storage** — every bottle has its own thermal history. The aggregate Vivino rating averages over this.

These are reasons the model has a noise floor below which it cannot improve. The honest framing of the headline result includes them.
