"""Cumulative top-N region coverage by ratings AND by (region, vintage) cells."""

from __future__ import annotations

import polars as pl

from vininator.data.load import scan_xwines_ratings, scan_xwines_wines


def main() -> None:
    w = scan_xwines_wines()
    r = scan_xwines_ratings()

    # Join ratings → wines once.
    joined = r.join(w.select("WineID", "RegionName"), on="WineID", how="inner")

    # Per-region rating volume, sorted desc.
    per_region = (
        joined.filter(pl.col("Vintage").is_not_null())
        .group_by("RegionName")
        .agg(pl.len().alias("n_ratings"))
        .sort("n_ratings", descending=True)
        .collect()
    )
    total_ratings = int(per_region["n_ratings"].sum())
    per_region = per_region.with_columns(
        (pl.col("n_ratings").cum_sum() / total_ratings).alias("cum_share")
    )

    print(f"total ratings (Vintage not null): {total_ratings:,}")
    print(f"unique regions:                   {len(per_region):,}\n")

    # Top-N tables: how many regions to cover X% of ratings.
    thresholds = [0.50, 0.75, 0.80, 0.90, 0.95, 0.99]
    print("regions needed to cover X% of ratings:")
    for t in thresholds:
        idx = per_region.filter(pl.col("cum_share") >= t).head(1)
        if len(idx) == 0:
            continue
        n = per_region.with_row_index().filter(pl.col("cum_share") >= t).head(1)["index"][0] + 1
        print(f"  {int(t*100):>2}%  ->  top {n:>5} regions  "
              f"(last region added: {idx['RegionName'][0]!r}, {int(idx['n_ratings'][0]):,} ratings)")

    # And how this translates to (region, vintage) cells eligible for ERA5.
    print("\n(region, vintage) cell counts for the top-N regions:")
    cells_full = (
        joined.filter(pl.col("Vintage").is_not_null())
        .group_by(["RegionName", "Vintage"])
        .agg(pl.col("WineID").n_unique().alias("n_wines"))
        .collect()
    )
    print(f"  all cells (>=1 wine):       {len(cells_full):,}")
    print(f"  cells with >=5 wines:       {len(cells_full.filter(pl.col('n_wines') >= 5)):,}")
    print(f"  cells with >=10 wines:      {len(cells_full.filter(pl.col('n_wines') >= 10)):,}")
    print(f"  cells with >=20 wines:      {len(cells_full.filter(pl.col('n_wines') >= 20)):,}")
    print()

    for top_n in (100, 200, 300, 500, 1000):
        top_regions = per_region.head(top_n)["RegionName"].to_list()
        eligible = cells_full.filter(
            pl.col("RegionName").is_in(top_regions) & (pl.col("n_wines") >= 5)
        )
        ratings_covered = (
            joined.filter(pl.col("RegionName").is_in(top_regions))
            .select(pl.len()).collect().item()
        )
        share = 100 * ratings_covered / total_ratings
        print(
            f"  top {top_n:>4} regions  ->  {len(eligible):>6,} eligible cells (>=5 wines), "
            f"covers {ratings_covered:>12,} ratings ({share:.1f}%)"
        )


if __name__ == "__main__":
    main()
