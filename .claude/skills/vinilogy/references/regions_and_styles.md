# Wine regions and styles

Reference for canonicalizing regions, choosing growing-season windows (`features/climate.py`), and sanity-checking model predictions. Hemisphere is the input that flips the growing-season window in ERA5 fetching.

## Hierarchy

Most wine regions have a multi-level appellation hierarchy. The model gets all levels; preserve them in the dataset.

```
Country → Region → Sub-region → Appellation → Sub-appellation/Cru
France  → Burgundy → Côte d'Or → Côte de Nuits → Gevrey-Chambertin → Clos de Bèze
Italy   → Tuscany  → Chianti → Chianti Classico
USA     → California → North Coast → Napa Valley → Oakville
Spain   → Castile and León → Ribera del Duero
```

WineSensed's region field is often a single string mixing levels ("Burgundy, France" or "Margaret River, Western Australia"). The parser should split on `,` and infer level from a static gazetteer rather than guessing.

## Northern hemisphere — growing season April–October

### France

| Region | Sub-region / appellation | Climate | Grapes | Style notes |
| --- | --- | --- | --- | --- |
| Bordeaux | Médoc, Pauillac, Margaux, St-Julien, St-Estèphe (Left Bank) | Maritime, gravelly | Cab Sauv-dominant blends + Merlot | Tannic, age-worthy. Cab Sauv ripens reliably here. |
| Bordeaux | St-Émilion, Pomerol (Right Bank) | Maritime, clay-limestone | Merlot-dominant + Cab Franc | Plusher, softer. Pomerol is the most expensive Merlot in the world. |
| Bordeaux | Sauternes, Barsac | Maritime + autumn fog | Sémillon + Sauv Blanc | Botrytized sweet. |
| Burgundy | Chablis | Cool continental, Kimmeridgian limestone | Chardonnay | Steely, mineral, unoaked. The cool-climate Chardonnay benchmark. |
| Burgundy | Côte de Nuits (Gevrey-Chambertin, Vosne-Romanée, Chambolle-Musigny, Nuits-St-Georges) | Cool continental | Pinot Noir | Most expensive Pinot Noir on earth. Highly vintage-variable. |
| Burgundy | Côte de Beaune (Pommard, Volnay, Meursault, Puligny-Montrachet, Chassagne-Montrachet) | Cool continental | Pinot Noir (reds) + Chardonnay (whites) | Reds lighter than CdN; whites are the world reference for oaked Chardonnay. |
| Burgundy | Mâconnais (Pouilly-Fuissé) | Slightly warmer | Chardonnay | Riper, rounder, less premium. |
| Beaujolais | Cru villages (Morgon, Fleurie, Moulin-à-Vent…) | Continental | Gamay | Carbonic maceration; fresh fruit; the high-end Crus age. |
| Champagne | Montagne de Reims, Côte des Blancs, Vallée de la Marne | Cool marginal | Pinot Noir, Chardonnay, Pinot Meunier | Vintage declared only in good years. |
| Loire | Sancerre, Pouilly-Fumé | Cool continental | Sauv Blanc | Flinty, mineral. The original style benchmark. |
| Loire | Vouvray, Savennières | Cool continental, slate/tuffeau | Chenin Blanc | Bone dry to lusciously sweet. |
| Loire | Chinon, Bourgueil | Cool continental | Cab Franc | Pencil-shaving + raspberry. |
| Loire | Muscadet | Maritime | Melon de Bourgogne | Light, saline, oyster pairing. |
| Northern Rhône | Côte-Rôtie, Hermitage, Cornas, St-Joseph, Crozes-Hermitage | Continental, granite | Syrah (+ optional Viognier co-ferment in Côte-Rôtie) | Peppery, savory, age-worthy. |
| Northern Rhône | Condrieu | Continental | Viognier | Stone fruit + honeysuckle, oily texture. |
| Southern Rhône | Châteauneuf-du-Pape, Gigondas, Vacqueyras | Warm Mediterranean | GSM blends | Riper, broader, 14–15% ABV common. |
| Alsace | Riesling, Pinot Gris, Gewürz, Pinot Blanc | Cool continental, sheltered | varietally labeled | Drier, richer than Germany; Grand Cru system. |
| Languedoc-Roussillon | Faugères, Pic St-Loup, Maury, Banyuls | Warm Mediterranean | Carignan, Grenache, Syrah | Old-vine bottlings increasingly esteemed. |
| Provence | Bandol | Warm Mediterranean | Mourvèdre | Long-aging rosé and red. |
| Jura | Arbois, Côtes du Jura | Cool continental | Savagnin, Poulsard, Trousseau, Chardonnay | Niche — Vin Jaune is oxidative under flor. |

### Italy

| Region | Appellation | Climate | Grapes | Style notes |
| --- | --- | --- | --- | --- |
| Piedmont | Barolo, Barbaresco, Roero, Gattinara | Continental, foggy | Nebbiolo | Tar, rose, dried cherry, very tannic, age-worthy. |
| Piedmont | Barbera d'Alba, Barbera d'Asti | Continental | Barbera | High acid, fruit-forward, low tannin. |
| Piedmont | Dolcetto d'Alba | Continental | Dolcetto | Low acid, medium tannin, casual drinking. |
| Tuscany | Chianti Classico | Mediterranean continental | Sangiovese (+ small Cab/Merlot) | Sour cherry, leather, tomato. |
| Tuscany | Brunello di Montalcino | Warmer Mediterranean | Sangiovese Grosso | Bigger, age-worthy 20+ years. |
| Tuscany | Bolgheri ("Super Tuscan") | Mediterranean coastal | Bordeaux blends | Sassicaia, Ornellaia. International style. |
| Veneto | Soave | Continental | Garganega | Almond, lemon, chamomile. |
| Veneto | Valpolicella + Amarone | Continental | Corvina + Rondinella + Molinara | Amarone is dried-grape — high alcohol, intense. |
| Friuli-Venezia Giulia | Collio, Colli Orientali | Continental | Pinot Grigio, Friulano, Ribolla Gialla | Whites a major focus. |
| Trentino-Alto Adige | — | Alpine | Pinot Grigio, Lagrein, Gewürz | High-elevation freshness. |
| Sicily | Etna, Vittoria | Volcanic, warm | Nerello Mascalese (Etna), Nero d'Avola | Etna producing very serious wines. |
| Campania | Taurasi | Volcanic | Aglianico | "Barolo of the South" — tannic, age-worthy. |
| Sardinia | — | Mediterranean | Cannonau (Grenache), Vermentino | — |

### Spain

| Region | Climate | Grapes | Style notes |
| --- | --- | --- | --- |
| Rioja (Alta, Alavesa, Oriental) | Continental, modified by Cantabrian range | Tempranillo + Garnacha + Mazuelo + Graciano | American oak vanilla/dill (traditional); French oak (modern). Crianza / Reserva / Gran Reserva = aging levels. |
| Ribera del Duero | Continental, high elevation | Tempranillo (locally "Tinto Fino") | Riper, more structured than Rioja. |
| Priorat | Mediterranean, llicorella slate | Garnacha + Carignan | Concentrated, mineral, age-worthy. |
| Toro | Continental | Tempranillo ("Tinta de Toro") | Bigger, alcoholic. |
| Rías Baixas | Cool Atlantic | Albariño | Saline, citrusy, coastal. |
| Rueda | Continental | Verdejo | Sauv-Blanc-adjacent style. |
| Jerez / Sherry | Andalusia, Atlantic-modified | Palomino, Pedro Ximénez | Fortified; under-flor or oxidative aging. |
| Cava region (mostly Penedès) | Mediterranean | Xarel-lo, Macabeo, Parellada | Méthode traditionnelle sparkling. |

### Germany / Austria

| Region | Climate | Grapes | Style notes |
| --- | --- | --- | --- |
| Mosel (Germany) | Cool, steep slate slopes | Riesling | Lightest German style; off-dry classics, lifelong agers. |
| Rheingau | Cool, south-facing | Riesling, Spätburgunder | Drier than Mosel. |
| Pfalz | Warmer | Riesling, Pinots, Dornfelder | Riper style. |
| Wachau (Austria) | Cool continental, Danube | Grüner Veltliner, Riesling | Smaragd is the ripest tier. |
| Burgenland | Warmer | Blaufränkisch, Zweigelt, Riesling | Reds + sweet wines (Ausbruch). |

### Rest of Europe

- **Portugal — Douro:** Port + dry reds. Touriga Nacional + Touriga Franca + Tinta Roriz blends.
- **Portugal — Dão, Alentejo:** dry reds and whites.
- **Portugal — Vinho Verde:** light, slightly spritzy whites (Loureiro, Alvarinho).
- **Greece — Santorini:** Assyrtiko, volcanic, saline, very high acid.
- **Greece — Naoussa:** Xinomavro (Greek "Nebbiolo").
- **Hungary — Tokaj:** Furmint, botrytized sweet. Aszú levels by puttonyos.
- **England — South Downs / Sussex:** sparkling Chardonnay + Pinot Noir + Meunier, increasingly serious.

### North America (Northern hemisphere)

| Region | Climate | Grapes | Style notes |
| --- | --- | --- | --- |
| Napa Valley (CA) | Warm Mediterranean | Cab Sauv, Chardonnay, Merlot | Ripe, oaked, internationally styled. Sub-AVAs (Oakville, Rutherford, Stags Leap, Howell Mountain) matter. |
| Sonoma County (CA) | Maritime to continental | Pinot Noir + Chardonnay (Russian River, Sonoma Coast), Zin (Dry Creek, Sonoma Valley), Cab (Alexander Valley) | More stylistic diversity than Napa. |
| Paso Robles, Santa Barbara, Santa Lucia Highlands (CA) | Variable | Rhône varieties, Pinot Noir, Chard | Santa Rita Hills Pinot is a benchmark. |
| Willamette Valley (Oregon) | Cool maritime | Pinot Noir, Chardonnay, Pinot Gris | The American Burgundy. |
| Columbia Valley + Walla Walla (Washington) | Continental, irrigated desert | Cab, Merlot, Syrah, Riesling | Underrated; intensely structured reds. |
| Finger Lakes (NY) | Cool continental | Riesling, Cab Franc | Cool-climate East Coast. |
| Long Island (NY) | Maritime | Bordeaux varieties | — |
| Niagara Peninsula (Ontario) | Cool continental | Riesling, Cab Franc, Icewine | Icewine center. |
| Okanagan Valley (BC) | Continental, dry | Wide range | Emerging. |
| Mexico — Valle de Guadalupe (Baja CA) | Mediterranean | Range | Boutique scene. |

## Southern hemisphere — growing season October–April

| Country / Region | Climate | Grapes | Style notes |
| --- | --- | --- | --- |
| Argentina — Mendoza, Uco Valley | High-altitude, dry, sunny | Malbec, Cab Sauv, Bonarda, Torrontés | Consistent vintages, intense color, ripe fruit. |
| Argentina — Patagonia | Cool, windy | Pinot Noir, Merlot | Cooler, leaner. |
| Chile — Maipo, Cachapoal | Mediterranean | Cab Sauv | Maipo Cabs rival Bordeaux at much lower prices. |
| Chile — Colchagua | Mediterranean | Carménère, Syrah, Cab | Signature: Carménère. |
| Chile — Casablanca, Limarí, Leyda | Cool coastal | Sauv Blanc, Chardonnay, Pinot Noir | Cool-climate Chile, recent wave. |
| Australia — Barossa Valley | Warm | Shiraz, Grenache, Mataró | Old vines (>100 yr), big rich style. |
| Australia — McLaren Vale | Warm Mediterranean | Shiraz, Grenache | Slightly cooler than Barossa. |
| Australia — Coonawarra | Cool continental, terra rossa | Cab Sauv | "Other" Cab benchmark; eucalyptus / mint signature. |
| Australia — Hunter Valley | Warm humid | Sémillon, Shiraz | Hunter Sémillon is unique: low-alcohol, ages 20+ years to honeyed complexity. |
| Australia — Margaret River | Maritime cool | Cab + Merlot, Chardonnay, Sauv Blanc + Sémillon | Bordeaux-blend benchmark in the New World. |
| Australia — Yarra Valley, Mornington, Tasmania | Cool | Pinot Noir, Chardonnay | Tasmania for sparkling. |
| Australia — Clare + Eden Valley | Continental | Riesling | Dry, citrus, ages on petrol. |
| New Zealand — Marlborough | Cool maritime | Sauv Blanc, Pinot Noir, Chardonnay | The Sauv Blanc benchmark globally. |
| New Zealand — Central Otago | Cool continental | Pinot Noir | Higher latitude than expected (45°S). |
| New Zealand — Hawke's Bay, Gimblett Gravels | Warmer | Syrah, Bordeaux blends | — |
| South Africa — Stellenbosch | Mediterranean | Cab, Chenin Blanc, Pinotage | Chenin Blanc here is world-class. |
| South Africa — Swartland | Mediterranean | Chenin, Syrah, Grenache | Old-vine, low-intervention scene. |
| South Africa — Hemel-en-Aarde | Cool maritime | Pinot Noir, Chardonnay | Cool-climate enclave. |
| Uruguay — Canelones | Maritime | Tannat | The unofficial Madiran of South America. |
| Brazil — Vale dos Vinhedos | Subtropical | Sparkling + Bordeaux varieties | Niche. |

## Style implication for the model

- **Hemisphere flips the climate fetch.** Encode `is_southern_hemisphere` from a static country/region table — Argentina, Chile, Australia, NZ, South Africa, Brazil, Uruguay = south; everything else = north. This is what `features/climate.py` reads to choose Apr–Oct vs. Oct–Apr.
- **Marginal climates have larger anomaly variance.** Burgundy, Champagne, Mosel, Sancerre, English sparkling — vintage variation dominates. Anomaly features should carry more weight here. Mendoza, Barossa — vintage variation is smaller; absolute climate matters more than anomalies.
- **Some regions don't have a meaningful single-year vintage signal.** Non-vintage Champagne is a blend; Sherry is a solera blend (no vintage at all in the traditional case); Port is fortified across years except for declared vintages. If the dataset includes these, vintage_year is misleading. Consider flagging via wine_type and letting CatBoost learn the interaction.
- **Sub-region matters.** Within Burgundy, Gevrey-Chambertin and Volnay are 30 km apart and produce structurally different Pinot Noir. The terroir block's coordinate-level resolution captures some of this; preserving the appellation-level categorical lets CatBoost capture the rest.

## Canonicalization notes

WineSensed will have spelling variants and casing inconsistencies. Build a `data/raw/region_canon.csv` table mapping observed strings to a canonical hierarchy. Examples:

| Observed | Canonical (country / region / sub-region / appellation) |
| --- | --- |
| "Burgundy" | France / Burgundy / null / null |
| "Gevrey-Chambertin, France" | France / Burgundy / Côte de Nuits / Gevrey-Chambertin |
| "Napa" | USA / California / Napa Valley / null |
| "Napa Valley, California" | USA / California / Napa Valley / null |
| "Oakville, Napa Valley" | USA / California / Napa Valley / Oakville |
| "Mendoza, Argentina" | Argentina / Mendoza / null / null |
| "Uco Valley" | Argentina / Mendoza / Uco Valley / null |
| "Barolo, Piedmont" | Italy / Piedmont / Langhe / Barolo |
| "Chianti Classico, Tuscany" | Italy / Tuscany / Chianti / Chianti Classico |

The geocoder runs after canonicalization — that way "Burgundy" and "Bourgogne" resolve to the same centroid, not two slightly different points.
