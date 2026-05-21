# Tasting vocabulary

Source material for `src/vininator/features/text.py` (parsing) and `src/vininator/models/tags.py` (multi-label vocabulary). The structure below is grounded in Ann Noble's UC Davis Wine Aroma Wheel (1990, 87 descriptors, 12 categories) extended with Vivino-frequency descriptors that the wheel groups too coarsely.

## The WSET five-point scale

Used by sommeliers worldwide for structural attributes. The model targets are mapped to these labels:

| Label | Body | Acidity | Tannin | Alcohol |
| --- | --- | --- | --- | --- |
| low | watery, thin | flabby, flat | none / absent | ≤11.0% |
| med- | light- to medium-light | tart but soft | gentle, barely there | 11.0–12.5 |
| med | balanced, average weight | balanced, fresh | noticeable, integrated | 12.5–13.5 |
| med+ | medium- to medium-full | crisp, sharp, lively | grippy, present | 13.5–14.5 |
| high | full, rich, weighty | searing, mouth-puckering | drying, harsh | 14.5+ |

Some reviewers and apps use `medium-minus` / `medium-plus` spelled out, others compress to `M-` / `M+`, others just say `light`, `medium`, `bold`. The regex below normalizes all of these.

## Regex patterns for parsing review text

For each axis, match left-to-right in this order — the more specific patterns first. Save them in `src/vininator/features/text.py` as module-level compiled regexes; never recompile per row.

```python
import re

# Body
BODY = [
    (r"\b(full[\- ]?bodied|full body|heavy|big|rich(?:ly)?\s+textured)\b", "high"),
    (r"\b(medium[\- ]?plus|med[\- ]?\+|medium\+\s*body|medium\s*to\s*full)\b", "med+"),
    (r"\b(medium[\- ]?bodied|medium body|moderately\s+bodied)\b", "med"),
    (r"\b(medium[\- ]?minus|med[\- ]?-|medium-\s*body|light\s*to\s*medium)\b", "med-"),
    (r"\b(light[\- ]?bodied|light body|delicate|thin|watery)\b", "low"),
]

# Acidity — "acid" is searched within a window of ±5 tokens from the descriptor
ACIDITY = [
    (r"\b(searing|racy|zippy|bracing|very\s+high)\s+acid", "high"),
    (r"\b(crisp|sharp|lively|fresh|high)\s+acid", "med+"),
    (r"\b(balanced|moderate|medium|med)\s+acid", "med"),
    (r"\b(soft|mild|gentle|low|low-medium)\s+acid", "med-"),
    (r"\b(flabby|flat|dull|lacking)\s+acid", "low"),
    # Inverse phrasing
    (r"\bhigh[\- ]acid", "med+"),
    (r"\blow[\- ]acid", "med-"),
]

# Tannin (red wines only — gate by wine_type before applying)
TANNIN = [
    (r"\b(harsh|aggressive|astringent|searing|very\s+high)\s+tannin", "high"),
    (r"\b(grippy|firm|chewy|powerful|high)\s+tannin", "med+"),
    (r"\b(medium|moderate|balanced|integrated)\s+tannin", "med"),
    (r"\b(soft|silky|smooth|gentle|low-medium)\s+tannin", "med-"),
    (r"\b(none|no|absent|very\s+low|low)\s+tannin", "low"),
]
```

Per-wine aggregation: majority vote across that wine's reviews. Ties broken by the more central label (a tie between `med` and `med+` resolves to `med`). The aggregation function lives at the per-`wine_id` level — see `references/leakage_rules.md` in the `terroir-features` skill for why this granularity is leakage-free.

Coverage expectation: structural attributes are mentioned in ~30–60% of reviews; for the remaining wines, leave the field null and let CatBoost handle missingness. Imputation would inject systematic bias toward whatever value the imputer chose.

## The ~150-tag flavor vocabulary

Use this list as the seed for `src/vininator/models/tags.py`. Verify each tag's frequency in the WineSensed review text before final inclusion — drop any tag below ~0.5% review-frequency (model can't learn it; just adds noise).

### 1. Fruity — red

`red cherry`, `sour cherry`, `black cherry`, `raspberry`, `strawberry`, `cranberry`, `pomegranate`, `red currant`

### 2. Fruity — black/blue

`blackberry`, `black cherry`, `blueberry`, `cassis`, `black plum`, `damson`, `mulberry`, `boysenberry`

### 3. Fruity — citrus

`lemon`, `lime`, `grapefruit`, `orange`, `tangerine`, `lemon zest`, `orange peel`

### 4. Fruity — stone

`peach`, `apricot`, `nectarine`, `yellow plum`

### 5. Fruity — tropical

`pineapple`, `mango`, `passion fruit`, `guava`, `lychee`, `banana`, `melon`

### 6. Fruity — orchard

`green apple`, `red apple`, `pear`, `quince`

### 7. Fruity — dried / cooked

`raisin`, `fig`, `prune`, `dried cherry`, `dried apricot`, `jammy`, `stewed fruit`, `dried cranberry`

### 8. Floral

`rose`, `violet`, `orange blossom`, `acacia`, `honeysuckle`, `jasmine`, `elderflower`, `lavender`, `geranium`

### 9. Spice — pepper

`black pepper`, `white pepper`, `pink pepper`

### 10. Spice — warm

`clove`, `cinnamon`, `nutmeg`, `allspice`, `star anise`, `cardamom`

### 11. Spice — savory

`licorice`, `anise`, `fennel`

### 12. Herbal — fresh

`mint`, `eucalyptus`, `basil`, `thyme`, `rosemary`, `sage`

### 13. Vegetal

`bell pepper`, `green pepper`, `jalapeño`, `tomato leaf`, `fresh-cut grass`, `asparagus`, `hay`, `straw`, `dried herbs`

### 14. Earthy

`forest floor`, `mushroom`, `truffle`, `wet leaves`, `damp earth`, `compost`, `potting soil`

### 15. Mineral / stony

`wet stone`, `flint`, `chalk`, `slate`, `gravel`, `iron`, `oyster shell`, `saline`, `iodine`

### 16. Oak-derived

`vanilla`, `coconut`, `dill`, `toast`, `cedar`, `sandalwood`, `smoke`, `char`, `cigar box`, `pencil shavings`

### 17. Microbial / aged red

`leather`, `tobacco`, `tobacco leaf`, `tar`, `game`, `meaty`, `cured meat`, `bacon`

### 18. Aged white / oxidative

`bruised apple`, `almond skin`, `nuts`, `nutty`, `dried fig`, `honeyed`, `beeswax`, `petrol`, `gasoline`, `kerosene`

### 19. Nutty

`almond`, `hazelnut`, `walnut`, `pecan`

### 20. Sweet / confectionery

`honey`, `caramel`, `butterscotch`, `chocolate`, `dark chocolate`, `cocoa`, `mocha`, `coffee`, `espresso`, `maple`

### 21. Dairy

`butter`, `cream`, `yogurt`, `crème fraîche`, `brioche`, `bread dough`, `biscuit`

### 22. Reductive / sulfurous

`struck match`, `gunflint`, `rubber`, `cabbage`

### 23. Effervescence (sparkling only)

`fine bubbles`, `mousse`, `creamy bubbles`, `aggressive bubbles`

### Notes on extension

- **Petrol / TDN** in aged Riesling is a positive descriptor, not a fault — leave it in the standard vocabulary, not the fault axis.
- **Saline / iodine** for coastal whites (Albariño, Manzanilla sherry) is positive — same.
- **Struck match / gunflint** is a deliberate winemaking style in some white Burgundies; only fault-y if extreme. Don't suppress.
- **Brett-adjacent descriptors** (leather, barnyard, game) are positive in some red wine styles (Rhône, aged Bordeaux) — these go in the standard vocabulary. Genuinely faulty brett shows up as **band-aid / sticking-plaster / horse sweat**, which lives in `references/faults_and_outliers.md`.

## Embedding approach for retrieval

For `src/vininator/models/notes.py`, embed the full review text with `sentence-transformers/all-MiniLM-L6-v2` (384 dim, fast, strong on short paragraphs). At inference, return k-nearest reviews from the training corpus. Two consequences:

1. **The retrieved notes use real wine language** — including grape-specific descriptors the model never explicitly learned. This is the main reason retrieval beats generative for the project: a generative model invents plausible-sounding descriptors that don't match the wine's actual style; retrieval can't.
2. **Diversity matters more than top-1.** k=5 with MMR (maximal marginal relevance) gives the user a richer view than 5 near-duplicate reviews. Implement if the vanilla nearest-neighbor approach feels repetitive.

The embedding model is multilingual; don't worry about the share of non-English reviews in WineSensed.
