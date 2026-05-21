"""Inspect X-Wines Body / Acidity / Type label distributions for the profile model."""

from __future__ import annotations

import polars as pl

from vininator.data.load import scan_xwines_ratings, scan_xwines_wines


def main() -> None:
    w = scan_xwines_wines()
    r = scan_xwines_ratings()

    for col in ("Body", "Acidity", "Type"):
        print(f"--- {col} ---")
        counts = (
            w.group_by(col).agg(pl.len().alias("n")).sort("n", descending=True).collect()
        )
        total = counts["n"].sum()
        for row in counts.iter_rows():
            print(f"  {str(row[0]):<25s} {row[1]:>6}  ({100*row[1]/total:.1f}%)")
        print()

    # Baseline RMSEs.
    print("--- baselines (predict each scheme's mean for held-out rows) ---")
    ratings = r.select("Rating", "WineID", "Vintage").collect()
    global_mean = ratings["Rating"].mean()
    global_std = ratings["Rating"].std()
    print(f"global mean rating = {global_mean:.3f}  std = {global_std:.3f}")
    print(f"  RMSE of global-mean predictor (in-sample) = {global_std:.3f}")

    # Per-wine mean baseline (in-sample RMSE — overestimates real perf since leaky).
    wine_mean = ratings.group_by("WineID").agg(pl.col("Rating").mean().alias("wmean"))
    j = ratings.join(wine_mean, on="WineID", how="inner")
    err = j["Rating"] - j["wmean"]
    print(f"  RMSE of per-wine mean (in-sample, optimistic) = {(err.pow(2).mean()) ** 0.5:.3f}")

    # Per (WineID, Vintage) mean baseline.
    wv_mean = (
        ratings.group_by(["WineID", "Vintage"]).agg(pl.col("Rating").mean().alias("wvmean"))
    )
    j = ratings.join(wv_mean, on=["WineID", "Vintage"], how="inner")
    err = j["Rating"] - j["wvmean"]
    print(f"  RMSE of per-(wine,vintage) mean (in-sample, optimistic) = {(err.pow(2).mean()) ** 0.5:.3f}")


if __name__ == "__main__":
    main()
