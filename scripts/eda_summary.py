"""Compute the figures needed to fill the EDA notebook summary."""

from __future__ import annotations

import polars as pl

from vininator.data.load import scan_xwines_ratings, scan_xwines_wines


def main() -> None:
    w = scan_xwines_wines()
    r = scan_xwines_ratings()

    rs = r.select("Rating").collect().to_series()
    print(f"rating: mean={rs.mean():.3f}  median={rs.median():.3f}  std={rs.std():.3f}  n={len(rs):,}")

    n_regions = w.select(pl.col("RegionName").n_unique()).collect().item()
    n_countries = w.select(pl.col("Country").n_unique()).collect().item()
    n_wineries = w.select(pl.col("WineryID").n_unique()).collect().item()
    n_grapes = (
        w.select(pl.col("Grapes").explode().alias("g"))
        .drop_nulls()
        .select(pl.col("g").n_unique())
        .collect()
        .item()
    )
    print(f"cardinality: regions={n_regions} countries={n_countries} wineries={n_wineries} grapes={n_grapes}")

    v_min = r.select(pl.col("Vintage").min()).collect().item()
    v_max = r.select(pl.col("Vintage").max()).collect().item()
    v_n = r.select(pl.col("Vintage").n_unique()).collect().item()
    print(f"Vintage range: {v_min} - {v_max}  ({v_n} distinct)")

    per_wine = r.group_by("WineID").agg(pl.len().alias("n")).collect()
    pw = per_wine["n"]
    print(
        f"ratings per wine: median={pw.median()}  min={pw.min()}  max={pw.max()}  mean={pw.mean():.1f}"
        f"  wines_with_ratings={len(per_wine)}"
    )

    joined = r.join(w.select("WineID", "RegionName"), on="WineID", how="inner")
    cells = (
        joined.group_by(["RegionName", "Vintage"])
        .agg(pl.col("WineID").n_unique().alias("n_wines"), pl.len().alias("n_ratings"))
        .collect()
    )
    eligible = cells.filter(pl.col("n_wines") >= 5)
    n_covered = int(eligible["n_wines"].sum()) if len(eligible) else 0
    print(
        f"(region, vintage) cells: total={len(cells)}  eligible(>=5 wines)={len(eligible)}"
        f"  wines_covered={n_covered}"
    )

    print()
    print("top 5 regions by wine count:")
    for row in (
        w.group_by("RegionName").agg(pl.len().alias("n")).sort("n", descending=True).head(5).collect().iter_rows()
    ):
        print(f"  {row[0]:<30s} {row[1]}")
    print("top 5 grapes by wine count:")
    for row in (
        w.select(pl.col("Grapes").explode().alias("g"))
        .drop_nulls()
        .group_by("g")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(5)
        .collect()
        .iter_rows()
    ):
        print(f"  {row[0]:<30s} {row[1]}")
    print("top 5 countries:")
    for row in (
        w.group_by("Country").agg(pl.len().alias("n")).sort("n", descending=True).head(5).collect().iter_rows()
    ):
        print(f"  {row[0]:<30s} {row[1]}")


if __name__ == "__main__":
    main()
