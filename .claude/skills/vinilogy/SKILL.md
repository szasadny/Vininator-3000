---
name: vinilogy
description: Wine domain knowledge — oenology (winemaking) and viticulture (grape growing) — for the Vininator 3000 project. Use this skill whenever decisions hinge on what's plausible for a wine: building the flavor-tag vocabulary, parsing body/acidity/tannin from review text, sanity-checking model predictions, designing retrieval queries, writing user-facing wine copy in the frontend, debating whether a feature signal is real or spurious, or interpreting SHAP values for the rating model. Trigger on prompts mentioning grape variety, varietal, wine region, appellation, vintage, tasting notes, flavor descriptors, wine body / acidity / tannin / alcohol, wine faults (corked, brett, oxidation), Vivino score interpretation, wine aging, or "does this prediction make sense." Pull this skill *before* writing the flavor vocabulary in src/vininator/features/text.py, before drafting retrieval prompts in models/notes.py, and before adding rules-based features to build.py — domain mistakes here are expensive and easy to make for non-specialists.
---

# Vinilogy for Vininator 3000

This skill packages the wine-domain knowledge the project needs to build, evaluate, and serve a model that predicts ratings, structural attributes, and tasting notes. "Vinilogy" here covers both **oenology** (winemaking science) and **viticulture** (grape growing) — the two halves of the wine world that produce the signals the model learns from.

The model is generic; the data is not. CatBoost can't tell you that "Riesling with 14% ABV is unusual" or that "a 2003 Bordeaux is a heat-vintage outlier." This skill is the lens that lets the human review the system honestly and makes the predictions feel like wine rather than statistics.

## When to consult this skill

- **Building the ~150 flavor-tag vocabulary** in `src/vininator/features/text.py`. Don't invent a list from intuition; the literature has done this work — see `references/tasting_vocabulary.md`.
- **Parsing body / acidity / tannin** from review text. The standard scale is `{low, med-, med, med+, high}` (WSET-style). Phrases differ by reviewer; the regex must handle the variants.
- **Sanity-checking predictions.** A model that predicts a full-bodied Albariño or a tannic Champagne has a bug. Use the grape and region tables to catch these.
- **Designing the retrieval-based notes model.** Embedding choices, prompt templates, and which descriptors to surface to the user all depend on what's plausible for the input wine.
- **Writing copy in the frontend.** Tooltips on the profile bars, the regional summary panel, the autocomplete labels — all should use industry vocabulary.
- **Reading SHAP outputs.** "Does it make sense that `gdd_anom` matters more for Burgundy than for Mendoza?" Yes, and `references/vintage_and_quality.md` explains why.

## Core mental model

A bottle of wine is the output of three composed functions:

```
grape variety  ──┐
                 ├──► viticulture (grape growing) ──► fruit chemistry ──┐
region + soil ──┤                                                       ├──► winemaking ──► bottle
climate (vintage)┘                                                      │
                  producer style, oak, fermentation, aging ─────────────┘
```

The model has direct access to the inputs on the left. It does not see fruit chemistry or producer technique directly; it has to infer them from grape × region × vintage × producer. This is why **producer is the strongest single predictor** in practice — it summarizes the right half of the pipeline. Terroir features add signal precisely where the producer effect is ambiguous: same producer, different vintage; same region, different vintage; etc. Frame the project's narrative around this.

## The four-axis tasting frame

Every wine can be located in a four-dimensional structural space. The project models three of these directly; alcohol is an input. All four use the **WSET five-point scale** as the industry convention:

| Axis | Scale | What it means | Source |
| --- | --- | --- | --- |
| **Body** | light / med- / med / med+ / full | Perceived weight on the palate. Driven by alcohol, glycerol, residual sugar, extract. | parsed from review |
| **Acidity** | low / med- / med / med+ / high | Tartness, mouth-watering. Driven by tartaric + malic acid, modified by malolactic fermentation. | parsed from review |
| **Tannin** | low / med- / med / med+ / high | Drying astringency, mostly red wines. Driven by skin/seed/stem extraction + oak. | parsed from review |
| **Alcohol** | low (≤11) / med (11–13.5) / high (13.5–15) / very high (15+) | ABV. | input |

These are not independent — a high-alcohol wine almost always reads as fuller-bodied; very high tannin without matching acidity reads as bitter. The model can learn these correlations; you should know them so you can spot when it hasn't.

See `references/tasting_vocabulary.md` for the regex patterns that map review phrasing ("med+ acidity", "soft tannins", "full-bodied", "high acid") to the canonical labels.

## What's plausible — quick lookup table

| Wine | Body | Acidity | Tannin | Alcohol | Aging |
| --- | --- | --- | --- | --- | --- |
| Cabernet Sauvignon (Bordeaux, Napa) | full | med+ | high | 13.5–14.5 | 10–20+ yr |
| Pinot Noir (Burgundy, Oregon) | light–med | med+ | low–med | 12.5–14 | 5–15 yr |
| Syrah / Shiraz | med+–full | med | med+ | 13.5–15.5 | 5–20 yr |
| Sangiovese (Chianti, Brunello) | med | high | med+ | 13–14.5 | 5–25 yr |
| Tempranillo (Rioja) | med | med | med+ | 13–14 | 5–20 yr |
| Nebbiolo (Barolo, Barbaresco) | med | high | very high | 13.5–14.5 | 10–30 yr |
| Merlot | med–full | med | med | 13–14.5 | 5–15 yr |
| Malbec | full | med | med+ | 13.5–14.5 | 5–10 yr |
| Zinfandel | full | med | med | 14–16 | 3–8 yr |
| Chardonnay (oaked) | med–full | med | — | 13–14.5 | 3–15 yr |
| Chardonnay (unoaked, cool) | light–med | high | — | 12–13.5 | 2–8 yr |
| Sauvignon Blanc | light–med | high | — | 12–13.5 | 1–5 yr |
| Riesling (dry) | light | very high | — | 8–12.5 | 5–30 yr |
| Riesling (sweet, Auslese+) | light–med | very high | — | 7–10 | 10–50 yr |
| Pinot Grigio | light | med+ | — | 12–13 | 1–3 yr |
| Champagne | light–med | high | — | 12–12.5 | varies |
| Port (vintage) | full | med | med+ | 19–22 | 20–50+ yr |
| Sauternes | full | high | — | 13–14 | 15–40 yr |

Use this when the model returns a prediction. A Pinot Noir flagged as `tannin=high` is almost certainly a parsing bug or an extreme outlier worth investigating; the same prediction for Nebbiolo is expected.

## The flavor-tag vocabulary

The Wine Aroma Wheel (Noble, UC Davis 1990; ~87 descriptors across 12 categories) is the canonical starting point. The project needs ~150 tags, which means extending the wheel with **specific wine-style descriptors** (oak, vanilla, leather, minerality, petrol/gasoline for aged Riesling, etc.) that the wheel sub-summarizes into "wood" or "earthy."

Full vocabulary draft and category tree: **`references/tasting_vocabulary.md`**. Twelve top-level categories:

1. **Fruity** — red fruit, black fruit, blue fruit, citrus, stone fruit, tropical, dried fruit, cooked fruit
2. **Floral** — rose, violet, orange blossom, jasmine, honeysuckle
3. **Spicy** — black pepper, white pepper, clove, cinnamon, anise, licorice
4. **Herbal / vegetal** — mint, eucalyptus, bell pepper (pyrazine), tomato leaf, fresh-cut grass, hay
5. **Earthy** — forest floor, mushroom, wet leaves, truffle
6. **Mineral** — wet stone, flint, chalk, slate, petrichor
7. **Oak-derived** — vanilla, coconut, dill, toast, smoke, char, cedar
8. **Microbial / aged** — leather, tobacco, game, dried fig
9. **Nutty** — almond, hazelnut, walnut
10. **Confectionery** — honey, caramel, butterscotch, chocolate, mocha
11. **Reductive / sulfur** — struck match, flint (overlaps with mineral but distinct mechanism)
12. **Faults** (tag but flag) — band-aid (brett), wet cardboard (TCA), nail polish (VA), nutty-bruised (oxidized)

The fault category is a **separate axis** in the system — predicting "this wine smells like wet cardboard" is not a quality signal you want to learn as a tasting note; it's a fault. Keep faults as detected-but-not-predicted; see `references/faults_and_outliers.md`.

## Vintage variation — what the climate signal actually means

The terroir block isn't decorative; it encodes the difference between a good and a bad year for a given region. Concrete examples (these are the kind of cases the user will eyeball during the qualitative sanity check):

- **Bordeaux 2003** — heat dome, GDD anomaly +200, harvest precip near zero, alcohol levels jumped 0.5–1.0% ABV across the region. Wines are richer, lower in acid, faster-aging than typical. A good model should price 2003 differently from 2000 or 2005.
- **Burgundy 2004** — wet, low GDD, ladybug infestation. Famously light, sometimes herbaceous. A model trained only on producer + region misses this; a model with terroir should see the negative GDD anomaly.
- **Champagne** — does not declare a vintage in poor years. Non-vintage Champagnes blend across years to maintain house style; vintage Champagnes appear ~3 years per decade. The model's "vintage_year" feature is less informative for Champagne than for Bordeaux; consider whether to learn this implicitly or flag the wine type.
- **Mendoza** — extremely consistent due to high-altitude, dry, sunny climate. Year-to-year terroir signal is smaller than in marginal climates. This is a feature, not a bug — the model should learn that anomaly variance is regional.

Full vintage / climate reference: **`references/vintage_and_quality.md`**.

## What "rating" actually represents on Vivino

The dataset target is a Vivino-style 1–5 score. Critical facts for evaluation:

- **Mean rating is ~3.6, not 2.5.** The distribution is shifted right because users rarely rate wines they actively disliked; a wine you wouldn't recommend rarely gets logged. Expect heavy density between 3.4 and 4.2.
- **A 4.0 wine is ~85th percentile.** A 4.5 is ~99th percentile. The headline RMSE target of "around 0.4" matches the global-mean baseline; the model needs to push noticeably below that to be useful.
- **Vivino correlates with expert critics around 4.0**, where 4.0 Vivino ≈ 90 Parker points. This matters for the **future-vintage holdout**: critics rate young vintages on potential; Vivino users rate the wine they're drinking. The two scores can disagree on the same vintage for genuine reasons.
- **Sample weighting via `log(1 + n_ratings)` matters.** A wine with 5000 ratings has converged toward the population mean; a wine with 5 ratings is noise. The project already does this; understand why before changing it.

The distribution shape constrains model evaluation: an RMSE of 0.3 on a distribution with stddev 0.4 is doing real work even though the absolute number looks small.

## Faults — what to detect, what to suppress

Some review descriptors are **wine faults**, not quality signals. Letting the model learn "wet cardboard → low rating" as a positive correlation embeds the fault into the prediction; the model will then predict "wet cardboard" as a tasting note for genuinely bad wines, which is wrong and unflattering.

Practical rule:

- **Flag** in tasting-note multi-label predictions: faults appear in the **detection** output (the system noticed the user described it that way) but are **excluded** from the predicted tag set for new wines unless the model is specifically asked.
- **Don't denoise the training data.** A corked bottle is a real review of a real experience; removing fault-tagged reviews biases the training distribution toward perfect storage. Keep them, but be aware of their presence.

See `references/faults_and_outliers.md` for the descriptor patterns and the typical confusion (cork taint vs. oxidation, brett vs. earthy-but-clean, reduction vs. minerality).

## Anti-patterns specific to this domain

Things that are easy to do, sound smart, and quietly hurt the model:

- **Treating grape variety as a flat categorical.** It has hierarchy: Pinot Noir is a parent class; Pinot Grigio is the same species but a different wine. CatBoost handles flat high-cardinality categoricals fine, but cross-wine generalization improves if blends are multi-hot encoded over varieties (the project already specifies this — keep it).
- **Joining vintage_year as a numeric.** It's both a feature (years correlate with weather patterns) and a categorical (specific vintages have reputations). The project's "binned decade" plus raw year is the right compromise — don't simplify.
- **Treating "minerality" as a known chemical signal.** It isn't; it's a perceptual cluster that correlates with high acidity, low oak, certain soils, and reductive sulfur compounds. Predicting it from terroir features is reasonable; *explaining* it via a single feature is not. Don't write copy that implies one-to-one correspondence.
- **Equating high price with high quality.** Price is endogenous — producers set it based on past ratings and demand. As a feature it's powerful and you should keep it; as a causal explanation it's circular.
- **Ignoring the bottle-age effect.** A 2015 Cabernet drunk in 2026 tastes different from one drunk in 2017. The project uses `bottle_age = opening_year - vintage_year` for this. Most reviews on Vivino are of young wines (1–5 years from vintage); the model will be most accurate in that window and degrade for very old or very young bottles.

## Reference files

Read the relevant file in full when you start the corresponding task. Each is written to be skimmable for the section you need.

- **`references/tasting_vocabulary.md`** — Wine Aroma Wheel structure, the ~150-tag vocabulary draft for the project, the WSET five-point scale, regex patterns for parsing body/acidity/tannin/alcohol from review text, variant phrasings.
- **`references/grape_varieties.md`** — top 30 red + 20 white varieties, characteristic flavors, body/acidity/tannin ranges, common regions, blending partners, common mis-spellings and synonyms (Syrah = Shiraz, Pinot Grigio = Pinot Gris).
- **`references/regions_and_styles.md`** — major wine regions by country, hemisphere (drives growing-season window in `features/climate.py`), characteristic grapes, expected style, appellation hierarchy, vintage variability profile.
- **`references/vintage_and_quality.md`** — what makes a great vintage in different climates, the GDD / Winkler index relationship, Vivino score distribution and percentile mapping, how to interpret SHAP outputs on climate features, sanity-check ranges per region.
- **`references/faults_and_outliers.md`** — the major wine faults (TCA, brett, oxidation, reduction, VA, mousiness), their descriptor patterns, how they appear in review text, and the project's handling rule (flag, don't suppress, don't predict).
