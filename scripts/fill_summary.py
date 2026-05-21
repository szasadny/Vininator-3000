"""Replace the final summary cell of the EDA notebook in-place, preserving outputs."""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/01_eda.ipynb")

SUMMARY = """## 9. What's actually usable - written summary

_Phase 1 deliverable, filled in against the **`full` X-Wines variant** (100,646 wines / 21,013,536 ratings, dropped in from the X-Wines Google Drive)._

- **Variant in use:** `full` (100,646 wines, 21,013,536 ratings)
- **Wines table missingness:** 0% on every column except `Website` (17.75%) — irrelevant for modeling.
- **Ratings table missingness:** 2.10% on `Vintage` and `age_at_review` (non-vintage wines, see caveats).
- **`age_at_review` summary:** count = 20,571,145; mean = 4.16y; median = 3y; std = 4.38y; p25 = 2y, p75 = 5y; min = 0y, max = 71y. **0 negative ages**, 12,484 ratings (0.06%) above 50y.
- **Review-Date range:** 2012-01-03 -> 2021-12-31 (exactly 10 calendar years). Ratings volume ramps from 54k (2012) → 1.4M (2014) → 3.66M (2020), then 3.63M (2021).
- **Rating distribution:** mean = 3.89, median = 4.00, std = 0.74 (1-5 scale, half-steps).
- **Vintage coverage on ratings:** 1950 - 2021 (73 distinct vintage years).
- **Cardinalities:** **2,160** unique regions, **62** countries, **30,510** wineries, **777** grape varieties. Top countries: France (24,371 wines), Italy (19,358), USA (13,139), Spain (7,109), Portugal (4,958). Top grapes: Cabernet Sauvignon (14,371), Chardonnay (12,416), Merlot (10,554), Pinot Noir (10,474), Syrah/Shiraz (10,090). Top regions: Mendoza (2,404), Bourgogne (2,182), California (1,831), Champagne (1,794), Napa Valley (1,741).
- **Ratings per wine:** median = 58, mean = 209, min = 5, max = 27,415. All 100,646 wines have at least 5 ratings (X-Wines applies that filter upstream).
- **(RegionName, Vintage) cells with `>= 5` wines:** **18,546 eligible cells out of 46,532 total**, covering **959,399 (wine, vintage) instances**. That is the working set Phase 2 will geocode and pull ERA5 for.
- **Notable surprises / caveats:**
  - 2.10% of ratings are for non-vintage wines (`Vintage = "N.V."`, e.g. Champagne / Port multi-vintage blends); `age_at_review` is null for these. Phase 3 should either filter them out or treat the wine itself as the unit (no growing season exists for an N.V. blend, so they cannot join to ERA5 by vintage).
  - 12,484 ratings (0.06%) have `age_at_review > 50y`, max 71y. Plausible for old fortifieds (Port, Madeira) and library Bordeaux, but worth a spot-check in Phase 3 — the long tail can pull `age_at_review` aggregates around if not winsorized.
  - The data window (2012-2021) means `age_at_review` is correlated with vintage year: a 2010 vintage can be rated at age 2-11, but a 2020 vintage can only be rated at age 0-1. The future-vintage holdout (`Vintage >= 2019`) controls for this, but per-rating splits will not.
  - 18,546 eligible cells means **the ERA5 pull is large** — at one daily pull per `(region, vintage)`, this is ~18k × ~210 days = ~3.9M daily reanalysis records. The Phase 2 cache + resumability requirements in PROJECT.md are not optional.
  - Rating mean = 3.89 (vs. 3.5 sometimes assumed for Vivino-style data). Baselines should beat this, not 3.5.

Phase 1 done. Phase 2 (geocoding + ERA5 + SoilGrids) unblocked.
"""


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    last_cell = nb["cells"][-1]
    if last_cell["cell_type"] != "markdown":
        raise SystemExit(
            f"Expected the last cell to be markdown (the summary), got {last_cell['cell_type']!r}"
        )
    last_cell["source"] = SUMMARY.splitlines(keepends=True)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated summary cell in {NB_PATH}")


if __name__ == "__main__":
    main()
