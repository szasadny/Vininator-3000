"""age_at_review + Date stats on the full variant."""

from __future__ import annotations

import polars as pl

from vininator.data.load import scan_xwines_ratings


def main() -> None:
    r = scan_xwines_ratings()

    age = r.select("age_at_review").collect().to_series()
    print(age.describe())
    n = len(age)
    n_negative = (age < 0).sum()
    n_over_50 = (age > 50).sum()
    n_null = age.null_count()
    print()
    print(f"total ratings:       {n:,}")
    print(f"  null age:          {n_null:,} ({100*n_null/n:.3f}%)")
    print(f"  age < 0:           {n_negative:,} ({100*n_negative/n:.3f}%)")
    print(f"  age > 50:          {n_over_50:,} ({100*n_over_50/n:.3f}%)")

    drng = r.select(
        pl.col("Date").min().alias("min"),
        pl.col("Date").max().alias("max"),
    ).collect().row(0)
    print(f"Date range: {drng[0]} -> {drng[1]}")

    by_year = (
        r.with_columns(pl.col("Date").dt.year().alias("y"))
        .group_by("y").agg(pl.len().alias("n"))
        .sort("y").collect()
    )
    print("ratings per review year:")
    for row in by_year.iter_rows():
        print(f"  {row[0]}  {row[1]:>10,}")


if __name__ == "__main__":
    main()
