# Faults and outliers

Wine faults are defects produced by spoilage organisms, oxidation, or environmental contamination. They show up in WineSensed review text as specific descriptors, and the project's tasting-note model has to distinguish them from positive descriptors that sound similar. Getting this wrong means the system may recommend "wet cardboard" as a flavor note, which is embarrassing.

## The major faults

### 2,4,6-trichloroanisole (TCA) — "cork taint"

- **What it is.** A chemical contamination, usually from natural cork closures treated with chlorine compounds.
- **Sensory signature.** Damp basement, wet newspaper, wet dog, mouldy cardboard. Critically: TCA suppresses fruit perception, so a corked wine smells "flat" and "muted" before it smells overtly mouldy.
- **Prevalence.** ~3–5% of cork-sealed bottles, less for screwcap. Different bottle of the same wine can be perfect.
- **Review-text patterns:**
  ```
  \b(corked|cork taint|TCA|wet cardboard|musty|mouldy|wet dog|wet newspaper|damp basement)\b
  ```
- **Project handling.** Tag as fault, do not include in the predicted-notes output. Keep in training set — corked reviews exist and the per-wine aggregation will partially average them out across multiple reviews.

### Brettanomyces — "brett"

- **What it is.** A wild yeast that produces 4-ethylphenol (4-EP) and 4-ethylguaiacol (4-EG), among other volatile phenols.
- **Sensory signature.** Band-aid, sticking plaster, horse sweat, barnyard, stable, sweaty saddle, smoked meat, clove (the latter from 4-EG, sometimes pleasant).
- **Subtlety.** Low-level brett is considered a positive complexity element in some traditional styles (older Rhône, aged Bordeaux, some Italian reds). High brett is universally a fault. The line is somewhere around 400 µg/L 4-EP — undetectable in literature, very personal in perception.
- **Review-text patterns** (high/faulty brett):
  ```
  \b(band[\- ]aid|sticking plaster|horse|horsey|sweaty|barnyard|stable|brett(y|anomyces)?)\b
  ```
- **Adjacent positive descriptors** (do NOT flag as fault):
  - `leather`, `tobacco`, `meaty`, `game`, `bacon`, `cured meat` — characteristic of old-world reds, often Syrah and aged Bordeaux. These exist with or without brett at low levels.
- **Project handling.** A separate "brett-suspect" boolean per review (regex above). Aggregated per wine, the fraction of reviews mentioning the regex words is the signal. Do not use as a positive feature — train the model with `brett_suspect_frac` as an auxiliary feature so it can learn the population-level effect, then evaluate.

### Volatile acidity (VA) — acetic / ethyl acetate

- **What it is.** Acetic acid produced by acetic acid bacteria, sometimes accompanied by ethyl acetate.
- **Sensory signature.** Nail polish remover, model glue, vinegar, Magic Marker, balsamic (at low levels). High VA is a serious fault; low VA is structurally invisible.
- **Subtlety.** Some classic styles (Amarone, some natural wines, some old-world reds) have noticeable VA as part of their character. Above a threshold (~1.0 g/L acetic acid), it's universally faulty.
- **Review-text patterns:**
  ```
  \b(nail polish|acetone|nail polish remover|vinegar|vinegary|magic marker|model (glue|airplane))\b
  ```
- **Project handling.** Flag, don't predict. The `vinegar` descriptor in particular has positive uses (balsamic-like in some Italian reds); confine the fault flag to the more specific terms.

### Oxidation — premature aging

- **What it is.** Excessive air exposure during winemaking, bottling, or storage. Most often a winemaking error or a closure failure (random bottles in a case can be oxidized).
- **Sensory signature.** Bruised apple, sherry-like nuttiness (in young wines that shouldn't be sherry-like), brown color (whites), brick-orange (reds — but normal aging also does this; context matters), dried fig, flat fruit.
- **Subtlety.** Oxidative styles are deliberate: Sherry under flor, Tawny Port, Vin Jaune, some orange wines. These have to be flagged as wine_type before applying the fault detector — otherwise the system will call all Sherry "oxidized."
- **Review-text patterns** (only flag for non-oxidative wine types):
  ```
  \b(oxidized|oxidised|bruised apple|sherry[\- ]like|dried out|flat|prematurely aged|prematox|premox)\b
  ```
- **The "premox" phenomenon.** White Burgundy from ~1995–2010 has a known premature-oxidation problem. The model may see lower-than-expected ratings on these wines without an obvious climate explanation. This is a known dataset confound.
- **Project handling.** Flag-but-don't-predict, conditional on wine type.

### Reduction — sulfide compounds

- **What it is.** Insufficient oxygen during winemaking creates volatile sulfur compounds (H₂S, mercaptans, dimethyl sulfide).
- **Sensory signature.** Struck match, gunflint, rubber, cabbage, rotten egg, garlic. Light reduction often blows off after decanting; heavy reduction doesn't.
- **Subtlety.** Some white Burgundy producers (notably Domaine Roulot, Coche-Dury) deliberately work in a reductive style — light struck-match is a feature, not a bug. **`flint` and `gunflint` are positive descriptors in this context** (they overlap with mineral; chemically they're often reductive).
- **Review-text patterns** (overtly faulty reduction):
  ```
  \b(rotten egg|sulfur|sulphurous|rubber|burnt rubber|cabbage|garlic|reduced)\b
  ```
- **Positive overlap** (do NOT flag):
  - `flint`, `gunflint`, `struck match`, `match head` — characteristic of certain reductive white styles.

### Light strike

- **What it is.** UV degradation, mostly from clear or green bottles in store windows. Champagne and rosé are particularly susceptible.
- **Sensory signature.** Cooked cabbage, wet wool, mothballs.
- **Subtlety.** Hard to distinguish from reduction in review text. Often presents as "off" without specific descriptors.

### Mousiness

- **What it is.** Tetrahydropyridines from microbial contamination, predominantly in natural / low-sulfur wines.
- **Sensory signature.** Mouse cage, popcorn, dirty washing-up cloth. Notoriously delayed in perception — comes on 10–30 seconds after sipping, not on the nose.
- **Review-text patterns:**
  ```
  \b(mousy|mousey|mouse cage|popcorn off|stale popcorn)\b
  ```
- **Subtlety.** Some palates can't detect it at all (~30% of tasters). Reviews can be inconsistent.

### Heat damage

- **What it is.** Bottle storage above ~25 °C for extended periods, or short exposure to very high temperatures (truck in summer, hot warehouse).
- **Sensory signature.** Cooked / stewed fruit, prune-like in reds, flat oxidized character. Often accompanied by **pushed cork** (the cork starts to extrude from the bottle).
- **Review-text patterns:**
  ```
  \b(cooked|stewed|prune[\- ]?like|heat damaged|burnt)\b
  ```
- **Subtlety.** Cooked + stewed are positive descriptors in some warm-climate reds and dessert wines. Context matters.

## Implementation rules for the project

For `src/vininator/features/text.py` and `src/vininator/models/tags.py`:

1. **Build a `is_fault_mention` regex** that ORs all the unambiguous fault patterns above. Compile once at module load.
2. **Per review, store `has_fault: bool` and `fault_types: list[str]`.** Useful as auxiliary features and for diagnostic queries.
3. **Per wine, aggregate `fault_review_frac = mean(has_fault)`.** A wine with `fault_review_frac > 0.15` is either consistently faulty (bad producer or distribution chain) or has a confounded style (Amarone, natural wine). This is a feature, not an exclusion criterion.
4. **In the multi-label tag predictor**, do not include the fault descriptors in the label space. The output of the model is a recommended tasting note for an *unfaulted* bottle. Fault probability is a separate output if needed.
5. **In the retrieval-based notes model**, optionally filter out reviews where `has_fault` is True before building the embedding index for similar-wine retrieval — you don't want the system to surface a corked review as the "5-star match" for a new wine.

## Outliers that aren't faults

Things that look fault-like in the data but aren't:

- **Aged red color shift toward brown / brick.** Expected at 10+ years. Wine_type and bottle_age explain it.
- **Tartrate crystals** ("wine diamonds"). Visual, not flavor. Sometimes flagged by users as "sediment" or "crystals" — never a fault.
- **High residual sugar.** Some users dislike sweet wines and rate them down with descriptors like "syrupy" or "cloying" — those are stylistic preferences, not faults. Don't flag.
- **Old-world funk.** Aged Bordeaux, traditional Rhône, Etna, old Rioja — leather, tobacco, forest floor, slightly oxidative notes are characteristic. Sample weights and regional context should let the model handle this; don't preemptively filter.
- **Natural wine cloudiness / VA / brett.** Stylistic, polarizing, often correctly labeled "natural" in the dataset. Consider an `is_natural` feature if natural wines are over-represented in some regions.

## Outlier wines worth knowing about

These are wines the model will see in training and may underestimate or mishandle:

- **Vintage Port** — 19–22% ABV, residual sugar 80–120 g/L. The model may not expect this combination.
- **Amarone della Valpolicella** — dried grapes, 15–17% ABV, raisin/prune dominant. Misclassifying as "ripe Cab" is a likely error mode.
- **Sherry (Fino, Manzanilla)** — low alcohol relative to fortified expectations (15%), saline, biological-aging signature. Very different from Oloroso (oxidative, dark).
- **Eiswein / Icewine** — frozen-grape concentration, extreme residual sugar, low alcohol, very high acid.
- **Vin Jaune** — Savagnin under flor in Jura. Should be wholly distinct from anything else in the dataset.
- **Orange wine** — extended-skin-contact whites. Tannic for a white; oxidative; often natural.
- **Pet-nat** — ancestral-method sparkling. Cloudy, often funky.

Where these appear in WineSensed, consider whether the model has enough samples to learn them. If fewer than ~50 wines of a style, the model will treat them as noise within whichever larger category dominates their region/grape combination.
