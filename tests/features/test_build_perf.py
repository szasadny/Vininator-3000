"""Performance regression test for the feature build.

The previous build burned 16.5 CPU-hours on the full 21M-rating variant before
being killed — driven by per-row Python lambdas in multi-hot encoding,
triple-collection of the 21M-row join, and vocabularies counted at rating
scale. This test catches gross regressions to any of those patterns by
asserting that a 500-wine × 20k-rating fixture builds well under wall-time
budget.

The fixture is ~6,500× smaller than the full variant (by ratings). The
rewritten code path completes it in well under a second on a developer
laptop; the budget is set to 5 seconds to leave headroom for CI noise while
still failing unambiguously if someone reintroduces per-row Python in the
hot path.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from tests.features.test_build import (
    _RATINGS_SCHEMA,
    _WINES_SCHEMA,
    _terroir_row,
)
from vininator.config import get_settings
from vininator.features.build import build_processed_tables
from vininator.features.terroir import TERROIR_SCHEMA

# Pool sizes chosen so vocabularies fill out beyond the trivial case but stay
# well under the configured TOP_K_GRAPES (50) and TOP_K_HARMONIZE (30). Mix of
# vintages spans the historical/future-vintage boundary so all three splits
# get non-empty output.
_GRAPE_POOL: list[list[str]] = [
    ["Merlot", "Cabernet Sauvignon"],
    ["Pinot Noir"],
    ["Chardonnay"],
    ["Sauvignon Blanc"],
    ["Syrah", "Grenache"],
    ["Tempranillo"],
    ["Riesling"],
    ["Sangiovese"],
    ["Malbec"],
    ["Nebbiolo"],
]

_HARMONIZE_POOL: list[str] = [
    "['Beef', 'Lamb']",
    "['Pork']",
    "['Cheese', 'Beef']",
    "['Fish', 'Shellfish']",
    "['Pasta']",
    "['Game', 'Lamb']",
]

_REGIONS: list[str] = ["Bordeaux", "Burgundy", "Loire"]
_VINTAGES: list[int] = [2015, 2016, 2017, 2018, 2020]


def _write_perf_fixture(n_wines: int, n_ratings: int) -> None:
    """Write wines / ratings / terroir parquets sized for the perf test."""
    settings = get_settings()

    wines: list[dict[str, Any]] = []
    for i in range(1, n_wines + 1):
        wines.append({
            "WineID": i,
            "WineName": f"Wine {i}",
            "Type": "Red",
            "Elaborate": "",
            "Grapes": _GRAPE_POOL[i % len(_GRAPE_POOL)],
            "Harmonize": _HARMONIZE_POOL[i % len(_HARMONIZE_POOL)],
            "ABV": 13.5,
            "Body": "Full-bodied",
            "Acidity": "High",
            "Code": "",
            "Country": "France",
            "RegionID": 1,
            "RegionName": _REGIONS[i % len(_REGIONS)],
            "WineryID": (i % 200) + 1,
            "WineryName": f"Winery {(i % 200) + 1}",
            "Website": None,
            "Vintages": list(_VINTAGES),
        })
    pl.DataFrame(wines, schema=_WINES_SCHEMA).write_parquet(
        settings.xwines_wines_parquet
    )

    ratings: list[dict[str, Any]] = []
    for i in range(1, n_ratings + 1):
        wine_id = ((i - 1) % n_wines) + 1
        vintage = _VINTAGES[i % len(_VINTAGES)]
        ratings.append({
            "RatingID": i,
            "UserID": 1,
            "WineID": wine_id,
            "Vintage": vintage,
            "Rating": 3.5 + (i % 10) * 0.15,
            "Date": datetime(2019, 6, 1),
            "age_at_review": 2019 - vintage,
        })
    pl.DataFrame(ratings, schema=_RATINGS_SCHEMA).write_parquet(
        settings.xwines_ratings_parquet
    )

    terroir_rows: list[dict[str, Any]] = []
    for region in _REGIONS:
        for vintage in _VINTAGES:
            terroir_rows.append(_terroir_row(region, "France", vintage))
    pl.DataFrame(terroir_rows, schema=TERROIR_SCHEMA).write_parquet(
        settings.terroir_parquet
    )


def test_build_completes_under_budget(tmp_data_dir: Path) -> None:
    """500-wine × 20k-rating build must finish under 5 seconds wall time.

    Catches three classes of regression:
      1. Per-row Python lambdas in multi-hot encoding (the original killer).
      2. Triple-collect of the joined frame (original second issue).
      3. Vocabulary built on rating-level instead of wine-level data.

    Any of these on this fixture would push wall time well past the budget.
    """
    _write_perf_fixture(n_wines=500, n_ratings=20_000)

    t0 = time.perf_counter()
    report = build_processed_tables(force=True)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, (
        f"build took {elapsed:.2f}s on 500-wine × 20k-rating fixture, "
        f"budget is 5.0s — likely regression to per-row Python in the hot path"
    )

    # Sanity: the build actually produced something.
    total = (
        report.train_rows
        + report.test_rows
        + report.future_vintage_test_rows
    )
    assert total > 0, "perf fixture produced zero output rows"
