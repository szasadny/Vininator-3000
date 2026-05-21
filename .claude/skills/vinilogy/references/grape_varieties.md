# Grape varieties

Quick-reference profile for the varieties that dominate WineSensed. Use to sanity-check predictions and to design the multi-hot encoding for blends (the project encodes the top ~50 grapes + an "other" bucket).

For each variety: typical body/acidity/tannin/alcohol ranges, signature flavor descriptors, principal regions, common blending partners, and synonyms (matters for canonicalization in `src/vininator/features/build.py`).

## Red varieties

### Cabernet Sauvignon
- Profile: full body, med+ acidity, high tannin, 13.5–14.5% ABV.
- Flavors: blackcurrant (cassis), black cherry, bell pepper (in cool years / underripe), cedar, pencil shavings (with oak), tobacco, graphite.
- Regions: Bordeaux (Left Bank — Médoc, Pauillac, St-Estèphe), Napa, Coonawarra (Australia), Maipo (Chile), Stellenbosch (South Africa).
- Blends: classic with Merlot + Cabernet Franc + Petit Verdot + Malbec ("Bordeaux blend").
- Synonyms: just "Cabernet" in casual review text — disambiguate from Cabernet Franc.

### Merlot
- Profile: medium-full body, medium acidity, medium tannin, 13–14.5% ABV.
- Flavors: black plum, black cherry, chocolate, herbal (bay leaf), softer / rounder than Cab Sauv.
- Regions: Bordeaux (Right Bank — Pomerol, St-Émilion), Napa, Washington State, northern Italy.
- Synonyms: none — but watch for "Merlot" mis-typed in user-entered text.

### Cabernet Franc
- Profile: medium body, med+ acidity, med+ tannin, 12.5–13.5% ABV.
- Flavors: red fruit, strawberry, graphite, bell pepper (more than Cab Sauv), violet, tobacco.
- Regions: Loire (Chinon, Bourgueil), Bordeaux Right Bank blends, Niagara, Long Island.

### Pinot Noir
- Profile: light to medium body, med+ acidity, low to medium tannin, 12.5–14% ABV.
- Flavors: red cherry, raspberry, strawberry, rose petal, forest floor (mushroom, earth) with age, clove.
- Regions: Burgundy (Côte de Nuits, Côte de Beaune), Champagne (sparkling base), Oregon (Willamette), Sonoma, Central Otago (NZ), Mornington (Aus).
- Synonyms: Spätburgunder (Germany), Pinot Nero (Italy), Blauburgunder (Austria, German Switzerland).

### Syrah / Shiraz
- Profile: medium+ to full body, medium acidity, med+ tannin, 13.5–15.5% ABV.
- Flavors: blackberry, black pepper, smoked meat, olive, violet (in cooler regions), licorice.
- Regions: Northern Rhône (Côte-Rôtie, Hermitage, Cornas), Barossa + McLaren Vale (Aus, "Shiraz"), Washington, South Africa.
- Synonyms: **Syrah = Shiraz** — same grape, different stylistic conventions (Syrah → peppery and savory, Shiraz → riper and fruitier). Canonicalize to one label in the dataset; the region carries the style information.

### Grenache / Garnacha
- Profile: medium-full body, medium acidity, low-medium tannin, 14–15.5% ABV.
- Flavors: strawberry, raspberry, white pepper, dried herbs, candied red fruit.
- Regions: Southern Rhône (GSM blends — Châteauneuf-du-Pape, Gigondas), Priorat (Spain), Sardinia (as "Cannonau").
- Synonyms: Garnacha (Spain), Cannonau (Sardinia), Grenache Noir.

### Sangiovese
- Profile: medium body, high acidity, med+ tannin, 13–14.5% ABV.
- Flavors: sour cherry, tomato, dried herbs, leather, balsamic.
- Regions: Tuscany (Chianti, Brunello di Montalcino, Vino Nobile di Montepulciano), Romagna, Umbria.
- Synonyms: Brunello (in Montalcino), Prugnolo Gentile (in Montepulciano), Morellino (Maremma). These are clones; canonicalize to Sangiovese at the variety level, keep the region.

### Tempranillo
- Profile: medium body, medium acidity, med+ tannin, 13–14% ABV.
- Flavors: red plum, dried fig, leather, tobacco, dill (with American oak), vanilla.
- Regions: Rioja, Ribera del Duero, Toro (Spain), Douro (Portugal — as Tinta Roriz / Aragonez).
- Synonyms: Tinta Roriz (Portugal), Aragonez (southern Portugal), Cencibel (Castilla-La Mancha).

### Nebbiolo
- Profile: medium body (light color, heavy structure — the "iron fist in velvet glove"), high acidity, very high tannin, 13.5–14.5% ABV.
- Flavors: rose, tar, sour cherry, leather, truffle, dried herbs.
- Regions: Piedmont (Barolo, Barbaresco, Gattinara, Roero).
- Note: distinctive structural profile — light color belies the tannin and aging potential. A model predicting "light tannin" for Nebbiolo is wrong.

### Malbec
- Profile: full body, medium acidity, med+ tannin, 13.5–14.5% ABV.
- Flavors: blueberry, blackberry, plum, violet, cocoa, mocha.
- Regions: Mendoza / Uco Valley (Argentina), Cahors (France — original home, more rustic).

### Zinfandel
- Profile: full body, medium acidity, medium tannin, 14–16% ABV (notoriously high).
- Flavors: jammy black fruit, dried cranberry, sweet spice, brambly, raisin.
- Regions: California (Lodi, Dry Creek, Amador, Sonoma). Genetically identical to **Primitivo** in Italy.
- Synonyms: Primitivo (Puglia, Italy).

### Other reds worth knowing for the dataset

- **Carménère** — Chile signature, herbaceous (capsicum), spicy. Once confused with Merlot.
- **Petite Sirah** (= Durif) — inky, big tannin, often blends.
- **Pinotage** — South Africa hybrid (Pinot Noir × Cinsault); polarizing, can show coffee / banana from yeast strains.
- **Mourvèdre / Monastrell / Mataró** — Bandol, GSM blends, Jumilla. Game, leather, blackberry, high tannin.
- **Aglianico** — Campania, Basilicata. Tannic, age-worthy, smoky.
- **Tannat** — Madiran (France), Uruguay. Highest tannin of any commonly-vinified grape.
- **Carignan / Mazuelo / Cariñena** — old-vine bottlings in Priorat, Languedoc.
- **Cinsault** — light, southern French blender; mostly seen in rosé.
- **Gamay** — Beaujolais. Light, fresh, banana from carbonic maceration.
- **Barbera** — Piedmont. High acid, low tannin, red fruit — the everyday wine of the region.
- **Dolcetto** — Piedmont. Medium tannin, low acid, blackberry. Opposite structural profile to Barbera.

## White varieties

### Chardonnay
- Profile: variable. Cool-climate unoaked: light-medium body, high acidity, 12–13.5%. Warm + oaked: medium-full body, medium acidity, 13.5–14.5%.
- Flavors: green apple, lemon (cool); pineapple, peach, vanilla, butter, brioche (warm + oaked).
- Regions: Burgundy (Chablis to Meursault to Mâcon — the whole stylistic range), Napa, Sonoma Coast, Russian River, Margaret River, Tasmania, Limoux (sparkling), Champagne (Blanc de Blancs).
- Note: stylistic extremes share a grape. Chardonnay alone tells the model little — Chardonnay × region tells it a lot.

### Sauvignon Blanc
- Profile: light to medium body, high acidity, 12–13.5% ABV.
- Flavors: grass, gooseberry, passion fruit (Marlborough style), grapefruit, bell pepper, asparagus.
- Regions: Marlborough (NZ — the iconic tropical/grassy style), Loire (Sancerre, Pouilly-Fumé — more flinty), Bordeaux (as Sauternes base + dry whites), Chile, South Africa.

### Riesling
- Profile: light body, very high acidity, 8–12.5% ABV (depending on sweetness level).
- Flavors: lime, green apple, jasmine, slate / petrol (with age — positive in this variety), apricot (Auslese+), honey (botrytized).
- Regions: Mosel, Rheingau, Nahe, Pfalz (Germany), Alsace (France), Wachau (Austria), Clare + Eden Valley (Australia), Finger Lakes (NY), Washington State.
- Sweetness scale (Germany): Trocken (dry), Halbtrocken / Feinherb (off-dry), Kabinett (light, ~ off-dry), Spätlese, Auslese, Beerenauslese, Trockenbeerenauslese (sweetest), Eiswein.

### Pinot Gris / Pinot Grigio
- Profile: light to medium body, med+ acidity, 12–13.5%.
- Flavors: pear, white peach, honey (Alsace style — richer), lemon, almond.
- Regions: Alsace (Pinot Gris — richer), northeast Italy + Friuli (Pinot Grigio — leaner). **Same grape, different style** — keep region as the discriminator.
- Synonyms: Grauburgunder (Germany), Pinot Gris (France, NZ, Oregon), Pinot Grigio (Italy).

### Gewürztraminer
- Profile: medium body, medium acidity (low for white standards), 13–14% ABV.
- Flavors: lychee, rose, ginger, sweet spice. **The most distinctive aromatic profile of any major white** — easy to call blind.
- Regions: Alsace, Trentino-Alto Adige, Germany (Pfalz).

### Other whites for the dataset

- **Viognier** — Condrieu, Northern Rhône blends. Stone fruit, honeysuckle, oily texture.
- **Albariño** — Rías Baixas (Spain). Saline, citrus, peach, high acid. Coastal.
- **Verdejo** — Rueda. Fennel, citrus, herbal.
- **Vermentino** — Sardinia, Liguria. Saline, citrus, almond bitterness.
- **Chenin Blanc** — Loire (Vouvray, Savennières), South Africa (Steen). Honey, quince, beeswax. Made dry to very sweet.
- **Sémillon** — Bordeaux (with Sauv Blanc), Hunter Valley (long-aging dry). Lanolin, honey, fig.
- **Grüner Veltliner** — Austria. White pepper, lentil, citrus.
- **Furmint** — Tokaj (Hungary). Honey, ginger, often botrytized.
- **Trebbiano / Ugni Blanc** — Italy / Cognac base. Neutral, high acid.
- **Garganega** — Soave. Almond, lemon, chamomile.
- **Assyrtiko** — Santorini (Greece). Saline, lemon, very high acid, volcanic.
- **Torrontés** — Argentina. Floral, aromatic, often confused with Gewürz / Muscat.

## Sparkling, fortified, sweet

These genuinely behave differently in the model — wine_type should be a feature, and the structural targets are conditional on it.

- **Champagne, Cava, Prosecco, English sparkling** — Pinot Noir + Chardonnay + Pinot Meunier (Champagne), Glera (Prosecco), Xarel-lo + Macabeo + Parellada (Cava). High acid, low tannin, the carbonation itself is a perceptual axis.
- **Port** — fortified, 19–22% ABV. Vintage Port is age-worthy 30+ years. Tawny is barrel-aged + oxidative — different flavor regime (caramel, nuts).
- **Sherry** — fortified, oxidative or biological aging (under flor). Fino / Manzanilla (dry, under flor), Amontillado (partial flor), Oloroso (oxidative), PX (sweet). Each is structurally distinct.
- **Sauternes / Tokaji / Ice Wine** — sweet, high acid, botrytis or freezing as the concentration mechanism. Honey, apricot, marmalade, ginger.
- **Madeira** — heated oxidative aging, virtually indestructible. Dry to sweet.

## Canonicalization rules for the dataset

Apply during ETL in `src/vininator/data/load.py` before any aggregation. Use a single canonical-name table (committed to `data/raw/grape_canon.csv` or hardcoded as a dict in the loader). Key cases:

| As written in WineSensed | Canonical |
| --- | --- |
| Shiraz | Syrah |
| Pinot Nero, Spätburgunder, Blauburgunder | Pinot Noir |
| Primitivo | Zinfandel |
| Garnacha, Cannonau | Grenache |
| Tinta Roriz, Aragonez | Tempranillo |
| Brunello, Prugnolo Gentile, Morellino | Sangiovese |
| Pinot Grigio | Pinot Gris |
| Grauburgunder | Pinot Gris |
| Mazuelo, Cariñena | Carignan |
| Ugni Blanc | Trebbiano |
| Steen | Chenin Blanc |
| Durif | Petite Sirah |

Do **not** collapse stylistic distinctions that share a grape across regions (Sangiovese from Brunello vs. Chianti). The region carries the style; collapsing grape names is enough.
