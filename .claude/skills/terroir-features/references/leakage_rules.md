# Leakage rules for the terroir + feature pipeline

The reason the headline result (terroir's contribution to predicting wine ratings) is interesting at all is that it has been computed without leakage. Read this before adding **any** aggregate feature.

## The split contract

The wine table is split by `wine_id`. A wine is either entirely in train or entirely in test. There are two test sets:

1. **Random wine holdout** — 15% of `wine_id`s, sampled with a fixed seed.
2. **Future-vintage holdout** — vintages 2019–2021 across the rest.

A wine in the future-vintage holdout may have other vintages (2017, 2018) of the same wine ID in train. That's intentional — it tests whether the model generalizes the terroir signal across years for a wine it has seen, not whether it generalizes to entirely unknown wines.

Both splits must be applied **before** any aggregation. The split function lives in `src/vininator/data/splits.py`; everything else (features, model) operates on the splits, not the union.

## What's safe to compute on the full dataset

Anything whose value for a row depends only on **inputs that the model would have at inference time** is safe. That's the test.

- **Soil features.** Depend on `region` only. Region is a model input. Safe.
- **Climate features.** Depend on `(region, vintage_year)`. Both are model inputs. Safe.
- **Geocoding.** Depends on `region` only. Safe.
- **Grape multi-hot encoding.** Depends on the wine's grape composition. Safe.
- **Country / sub-region from a static gazetteer.** Safe.

What these have in common: nothing about the row's *label* leaks into the feature.

## What is NOT safe to compute on the full dataset

Any feature that aggregates other rows' labels — directly or indirectly — leaks. The big ones:

### Producer aggregates

Producer mean rating, producer rating std, producer n_reviews — all computed **on the training fold only**, then left-joined onto test/val on `producer_id`:

```python
def producer_stats(train: pl.DataFrame) -> pl.DataFrame:
    return train.group_by("producer_id").agg(
        pl.col("rating").mean().alias("producer_mean_rating"),
        pl.col("rating").std().alias("producer_std_rating"),
        pl.col("rating").count().alias("producer_n_reviews"),
    )

train_aug = train.join(producer_stats(train), on="producer_id", how="left")
test_aug = test.join(producer_stats(train), on="producer_id", how="left")  # same stats, test never contributes
```

For producers that appear only in test, the join yields nulls. CatBoost handles nulls natively; do **not** impute with the global mean (that's a sneaky leak: the global mean was computed including test, or you'd need a train-only global, which is fine if explicit).

### Price imputation by `(region × grape)` mean

Missing prices are imputed with the median price of similar wines. Compute the imputation table on the training fold and apply to test the same way:

```python
imputation = train.group_by(["region", "grape"]).agg(pl.col("price").median())
test = test.join(imputation, on=["region", "grape"], how="left").with_columns(
    pl.coalesce("price", "price_right").alias("price")
)
```

### Cross-wine text aggregates

Mean review length per region, average flavor-tag frequency per grape, etc. — never. If you want a regional flavor signal, derive it from terroir features (which are not labels), not from review text.

### Sample weights

`log(1 + n_ratings)` per wine — `n_ratings` is part of the wine's metadata, not its label. Safe.

## Per-wine text aggregation is safe — but watch the boundary

Each wine has multiple reviews. Body/acidity/tannin are parsed from individual review texts and aggregated per wine (majority vote across that wine's reviews). Because the split is by `wine_id`, all reviews for a given wine are on the same side of the split. So:

- Per-wine majority vote: **safe** (uses only that wine's own reviews).
- Per-region majority vote, applied to all wines in that region: **leaks** (test wines pull from train labels in the same region; per-wine vote for a test wine pulls from its own reviews, which is fine, but doing it region-wide overshoots).

Implement per-wine aggregation as a `group_by("wine_id")` *before* the split is consulted — it's a property of the wine, not of the fold. The aggregation is leakage-free because the granularity matches the split key.

## The fold-only feature engineering pattern

For any feature whose computation depends on the label, use the following pattern. It's verbose; do it anyway.

```python
def build_features(train: pl.DataFrame, test: pl.DataFrame, future: pl.DataFrame):
    # 1) Compute training-fold-only artifacts
    producer_stats_df = producer_stats(train)
    price_imp_df = price_imputation(train)

    # 2) Apply the same artifacts to every fold
    def apply(df):
        df = df.join(producer_stats_df, on="producer_id", how="left")
        df = apply_price_imputation(df, price_imp_df)
        return df

    return apply(train), apply(test), apply(future)
```

If a downstream training script needs to recompute splits (e.g., for cross-validation), it must re-derive producer stats on each inner fold. This is what makes grouped CV faithful — every fold treats the rest as if it were unseen.

## Future-vintage holdout caveats

The future-vintage test (2019–2021) exists to detect a specific failure mode: the model has memorized "Burgundy reds are around 4.0" rather than learned "Burgundy reds in years with anomalous GDD score X higher." If the model performs equivalently with and without the terroir block on the random holdout but **worse without it** on the future-vintage holdout, that's the signal terroir is doing real work.

To preserve that signal:

- **Climatology must be computed on 1991–2020** — overlapping the future-vintage years with the climatology baseline would normalize away the anomaly we want to detect. Specifically, do not use a climatology window that includes 2019–2021.
- **No producer stats on future-vintage rows.** Same rule as for the random test — compute on training fold (≤2018), apply to 2019–2021.
- **Per-wine features for wines that only exist in 2019–2021 yield nulls.** That's accurate. CatBoost can handle it.

## Verification

Before declaring the dataset assembly done, add a test that:

1. Builds train + test with `build_features`.
2. Asserts `assert test["producer_mean_rating"].mean() != test["rating"].mean()` (would be near-equal if leakage existed).
3. Asserts no test row's `producer_mean_rating` is computed using any test row — by hashing the inputs to `producer_stats` and ensuring it touches only train indices.

These are cheap and catch the kind of mistake that's invisible in a final RMSE number.
