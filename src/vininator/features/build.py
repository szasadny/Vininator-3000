"""Assemble the final modeling table from X-Wines + terroir → data/processed/.

Phase 3. Produces three parquets (train / test / future_vintage_test), each
with one row per rating. The split boundary is at vintage_year 2018: vintages
≤ 2018 are randomly split 85/15 by WineID; vintages 2019–2021 form the
future-vintage holdout regardless of WineID.

Leakage rules enforced here — not just documented:
  1. Producer aggregates (mean_rating, rating_std, n_reviews) are computed on
     the train split only, then left-joined onto all three splits. Wineries
     unseen in train get null — CatBoost handles that natively.
  2. Vocabulary for the Harmonize multi-hot (top-30 food pairings) and the
     Grapes multi-hot (top-50 varieties) is built from the train split only,
     then applied to all splits. Unseen entries → all-zero row.
  3. Sample weights (log(1 + n_ratings_for_wine_in_train)) are derived from
     the train split only. The weight column is also populated on test /
     future-vintage rows using the train-derived count for that wine_id (null
     for wines not in train — only relevant for analysis, not the loss).

Shape, after the 21M-row rewrite:
  1. Build a wine-level features table (~100k rows) — parse Harmonize once
     into list[str], then encode multi-hot at THIS scale, not at the
     21M-rating scale. ~200× less work than rating-level encoding.
  2. Compute train-only producer aggs and sample weights against minimal
     ratings projections (3 cols, no joins).
  3. Stream each split's output via `sink_parquet`: scan ratings, assign the
     per-rating split, filter, hash-join the four tiny lookup tables
     (wine_features, terroir, producer_aggs, wine_weights), select the pinned
     column order, sink to disk. The 21M-row frame is never materialised in
     Python memory.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import polars as pl

from vininator.config import (
    FUTURE_VINTAGE_HOLDOUT,
    TOP_K_GRAPES,
    TOP_K_HARMONIZE,
    TRAIN_HOLDOUT_FRAC,
    WINE_SPLIT_SEED,
    get_settings,
)
from vininator.data.load import scan_xwines_ratings, scan_xwines_wines
from vininator.features.terroir import TERROIR_SCHEMA, scan_terroir
from vininator.features.text import (
    _slugify,
    build_grape_vocab,
    build_harmonize_vocab,
    encode_grape_multihot,
    encode_harmonize_multihot,
    parse_harmonize_column,
)

NotifyFn = Callable[[str], None]


# Terroir columns to carry into the processed table. Drop QA-only metadata and
# join-key geography columns.
_TERROIR_FEATURE_COLS: list[str] = [
    c for c in TERROIR_SCHEMA
    if not any(c.endswith(s) for s in ("_status", "_error", "_fetched_at"))
    and c not in ("region", "country", "lat", "lon", "vintage_year")
]

_PREFIX_COLS: list[str] = [
    "rating_id",
    "wine_id",
    "rating",
    "rating_date",
    "split",
    "sample_weight",
    "vintage_year",
    "age_at_review",
    "abv",
    "wine_type",
    "country",
    "region_name",
    "winery_id",
    "grape_majority",
]

_SUFFIX_COLS: list[str] = [
    *_TERROIR_FEATURE_COLS,
    "producer_mean_rating",
    "producer_rating_std",
    "producer_n_reviews",
    "body_label",
    "acidity_label",
]


@dataclasses.dataclass(frozen=True)
class BuildReport:
    """Summary returned by build_processed_tables."""
    train_rows: int
    test_rows: int
    future_vintage_test_rows: int
    grape_vocab_size: int
    harmonize_vocab_size: int
    output_columns: int


def build_processed_tables(
    *,
    force: bool = False,
    notify_fn: NotifyFn | None = None,
) -> BuildReport:
    """Assemble and write train / test / future_vintage_test parquets.

    Args:
        force: Overwrite existing outputs. When False and all three parquets
            exist, returns immediately with counts derived from a cheap
            schema scan.
        notify_fn: Optional callback invoked with milestone messages
            (loading, vocab build, per-split write). Follows the same
            pattern as `geocode_regions` / `build_climate_table`.

    Returns:
        BuildReport with row counts and vocab sizes.
    """
    settings = get_settings()
    settings.ensure_dirs()

    train_path = settings.processed_train_parquet
    test_path = settings.processed_test_parquet
    fv_path = settings.processed_future_vintage_test_parquet

    if not force and all(p.exists() for p in (train_path, test_path, fv_path)):
        return _report_from_cache(train_path, test_path, fv_path)

    # --- Stage 1: lazy scans ---
    _notify(notify_fn, "... loading wines and parsing Harmonize")
    wines_lf = _load_wine_features()
    ratings_lf = _scan_ratings_renamed()
    terroir_lf = _scan_terroir_subset()

    # --- Stage 2: assign train/test on the historical wine_id set ---
    _notify(notify_fn, "... assigning train/test split by wine_id")
    test_wine_ids = _assign_test_wine_ids(ratings_lf)

    # --- Stage 3: build vocabularies from train wines only ---
    _notify(notify_fn, f"... building grape vocabulary (top {TOP_K_GRAPES})")
    grape_vocab, harmonize_vocab = _build_vocabs_from_train_wines(
        wines_lf, test_wine_ids
    )
    _notify(
        notify_fn,
        f"... building food-pairing vocabulary (top {TOP_K_HARMONIZE})",
    )

    # --- Stage 4: encode multi-hot at wine-level scale (~100k rows) ---
    _notify(notify_fn, "... encoding wine-level multi-hot features")
    wine_features = _attach_wine_multihot(wines_lf, grape_vocab, harmonize_vocab)

    # --- Stage 5: train-only producer aggs and weight counts ---
    _notify(notify_fn, "... computing producer aggregates (train fold)")
    producer_aggs = _compute_producer_aggs(ratings_lf, wines_lf, test_wine_ids)
    _notify(notify_fn, "... computing sample weights (train fold)")
    wine_weights = _compute_wine_weights(ratings_lf, test_wine_ids)

    # --- Stage 6: stream each split's output ---
    col_order = _build_col_order(grape_vocab, harmonize_vocab)
    split_expr = _split_assignment_expr(test_wine_ids)

    train_rows = _stream_split(
        ratings_lf, wine_features, terroir_lf, producer_aggs, wine_weights,
        split_expr, "train", col_order, train_path, notify_fn,
    )
    test_rows = _stream_split(
        ratings_lf, wine_features, terroir_lf, producer_aggs, wine_weights,
        split_expr, "test", col_order, test_path, notify_fn,
    )
    fv_rows = _stream_split(
        ratings_lf, wine_features, terroir_lf, producer_aggs, wine_weights,
        split_expr, "future_vintage_test", col_order, fv_path, notify_fn,
    )

    return BuildReport(
        train_rows=train_rows,
        test_rows=test_rows,
        future_vintage_test_rows=fv_rows,
        grape_vocab_size=len(grape_vocab),
        harmonize_vocab_size=len(harmonize_vocab),
        output_columns=len(col_order),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _notify(notify_fn: NotifyFn | None, message: str) -> None:
    if notify_fn is not None:
        notify_fn(message)


def _report_from_cache(
    train_path: Path, test_path: Path, fv_path: Path
) -> BuildReport:
    """Cheap BuildReport from existing parquets (no rebuild)."""
    schema = pl.scan_parquet(train_path).collect_schema()
    g_size = sum(
        1 for c in schema
        if c.startswith("grape_") and c not in ("grape_majority", "grape_other")
    )
    p_size = sum(1 for c in schema if c.startswith("pair_"))
    return BuildReport(
        train_rows=pl.scan_parquet(train_path).select(pl.len()).collect().item(),
        test_rows=pl.scan_parquet(test_path).select(pl.len()).collect().item(),
        future_vintage_test_rows=pl.scan_parquet(fv_path).select(pl.len()).collect().item(),
        grape_vocab_size=g_size,
        harmonize_vocab_size=p_size,
        output_columns=len(schema),
    )


def _load_wine_features() -> pl.LazyFrame:
    """Wines parquet → renamed columns + parsed harmonize_list + grape_majority."""
    return (
        scan_xwines_wines()
        .rename({
            "WineID": "wine_id",
            "Type": "wine_type",
            "ABV": "abv",
            "Body": "body_label",
            "Acidity": "acidity_label",
            "Country": "country",
            "RegionName": "region_name",
            "WineryID": "winery_id",
        })
        .with_columns(
            pl.col("region_name").str.strip_chars(),
            pl.col("country").str.strip_chars(),
            pl.col("Grapes").list.first().alias("grape_majority"),
        )
        .pipe(parse_harmonize_column, src="Harmonize", dst="harmonize_list")
        .select([
            "wine_id", "wine_type", "abv", "body_label", "acidity_label",
            "country", "region_name", "winery_id", "Grapes", "harmonize_list",
            "grape_majority",
        ])
    )


def _scan_ratings_renamed() -> pl.LazyFrame:
    """Ratings with canonical lowercase names; null-vintage rows dropped."""
    return (
        scan_xwines_ratings()
        .rename({
            "RatingID": "rating_id",
            "WineID": "wine_id",
            "Vintage": "vintage_year",
            "Rating": "rating",
            "Date": "rating_date",
        })
        .filter(pl.col("vintage_year").is_not_null())
        .select([
            "rating_id", "wine_id", "vintage_year", "rating",
            "rating_date", "age_at_review",
        ])
    )


def _scan_terroir_subset() -> pl.LazyFrame:
    """Terroir lazy frame with whitespace-stripped join keys + feature cols only."""
    return (
        scan_terroir()
        .with_columns(
            pl.col("region").str.strip_chars(),
            pl.col("country").str.strip_chars(),
        )
        .select(["region", "country", "vintage_year", *_TERROIR_FEATURE_COLS])
    )


def _assign_test_wine_ids(ratings_lf: pl.LazyFrame) -> pl.Series:
    """Return the held-out test wine_ids from the historical-vintage shuffle.

    Only wines with at least one rating in vintage < FUTURE_VINTAGE_HOLDOUT[0]
    participate in the shuffle. Future-vintage-only wines (if any) are routed
    by vintage_year regardless of wine_id and never appear here.

    Unique wine_ids are sorted before the seeded shuffle so the test set is
    reproducible across runs regardless of `unique()`'s implementation order.
    """
    fv_lo, _ = FUTURE_VINTAGE_HOLDOUT
    historical_ids = (
        ratings_lf
        .filter(pl.col("vintage_year") < fv_lo)
        .select("wine_id")
        .unique()
        .sort("wine_id")
        .collect()
        .get_column("wine_id")
        .shuffle(seed=WINE_SPLIT_SEED)
    )
    n_test = max(1, int(len(historical_ids) * TRAIN_HOLDOUT_FRAC))
    return historical_ids[:n_test].rename("test_wine_id")


def _build_vocabs_from_train_wines(
    wines_lf: pl.LazyFrame, test_wine_ids: pl.Series
) -> tuple[list[str], list[str]]:
    """Build grape + pair vocabs from UNIQUE TRAIN wines only.

    Counting at the wine level (not rating level) is the load-bearing
    optimisation — for the full variant that's ~85k unique wines vs ~17M
    train ratings, a ~200× reduction in counting work.
    """
    test_set = test_wine_ids.implode()
    train_wines = (
        wines_lf
        .filter(~pl.col("wine_id").is_in(test_set))
        .select(["Grapes", "harmonize_list"])
        .collect()
    )
    grape_vocab = build_grape_vocab(train_wines["Grapes"], top_k=TOP_K_GRAPES)
    harmonize_vocab = build_harmonize_vocab(
        train_wines["harmonize_list"], top_k=TOP_K_HARMONIZE
    )
    return grape_vocab, harmonize_vocab


def _attach_wine_multihot(
    wines_lf: pl.LazyFrame,
    grape_vocab: list[str],
    harmonize_vocab: list[str],
) -> pl.DataFrame:
    """Run multi-hot encoding at wines-table scale (~100k rows). Collect once.

    Source list columns are dropped — the multi-hot columns supersede them
    and downstream joins don't need them.
    """
    return (
        wines_lf
        .pipe(encode_grape_multihot, vocab=grape_vocab, src="Grapes")
        .pipe(
            encode_harmonize_multihot,
            vocab=harmonize_vocab,
            src="harmonize_list",
        )
        .drop("Grapes", "harmonize_list")
        .collect()
    )


def _compute_producer_aggs(
    ratings_lf: pl.LazyFrame,
    wines_lf: pl.LazyFrame,
    test_wine_ids: pl.Series,
) -> pl.DataFrame:
    """Train-fold-only producer aggregates by winery_id.

    Operates on a 2-col ratings projection joined to a 2-col wines projection
    — both sides are minimal so the lazy plan finishes in a single streaming
    pass over the train ratings.
    """
    fv_lo, _ = FUTURE_VINTAGE_HOLDOUT
    test_set = test_wine_ids.implode()
    return (
        ratings_lf
        .filter(pl.col("vintage_year") < fv_lo)
        .filter(~pl.col("wine_id").is_in(test_set))
        .select("wine_id", "rating")
        .join(
            wines_lf.select("wine_id", "winery_id"),
            on="wine_id",
            how="inner",
        )
        .group_by("winery_id")
        .agg(
            pl.col("rating").mean().alias("producer_mean_rating"),
            pl.col("rating").std().alias("producer_rating_std"),
            pl.len().alias("producer_n_reviews"),
        )
        .collect()
    )


def _compute_wine_weights(
    ratings_lf: pl.LazyFrame,
    test_wine_ids: pl.Series,
) -> pl.DataFrame:
    """Train-fold-only `sample_weight` per wine_id (native log1p)."""
    fv_lo, _ = FUTURE_VINTAGE_HOLDOUT
    test_set = test_wine_ids.implode()
    return (
        ratings_lf
        .filter(pl.col("vintage_year") < fv_lo)
        .filter(~pl.col("wine_id").is_in(test_set))
        .group_by("wine_id")
        .agg(pl.len().alias("_n_ratings_train"))
        .with_columns(
            pl.col("_n_ratings_train").cast(pl.Float64).log1p().alias("sample_weight")
        )
        .drop("_n_ratings_train")
        .collect()
    )


def _split_assignment_expr(test_wine_ids: pl.Series) -> pl.Expr:
    """Per-rating split assignment expression.

    Future-vintage routing wins over wine_id routing — a wine whose
    historical ratings landed in `test_wine_ids` still sees its 2019-21
    ratings sent to `future_vintage_test`.
    """
    fv_lo, fv_hi = FUTURE_VINTAGE_HOLDOUT
    test_set = test_wine_ids.implode()
    return (
        pl.when(pl.col("vintage_year").is_between(fv_lo, fv_hi))
        .then(pl.lit("future_vintage_test"))
        .when(pl.col("wine_id").is_in(test_set))
        .then(pl.lit("test"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )


def _stream_split(
    ratings_lf: pl.LazyFrame,
    wine_features: pl.DataFrame,
    terroir_lf: pl.LazyFrame,
    producer_aggs: pl.DataFrame,
    wine_weights: pl.DataFrame,
    split_expr: pl.Expr,
    split_name: str,
    col_order: list[str],
    target: Path,
    notify_fn: NotifyFn | None,
) -> int:
    """Build the lazy plan for one split, stream it to .tmp, atomic-rename, count rows.

    Each split runs its own join independently — but the probe side of every
    hash join is a tiny in-memory frame (wine_features ~100k, terroir ~2k,
    producer_aggs ~50k, wine_weights ~85k), so the only thing streamed is the
    21M-row ratings scan. Polars handles the join in Rust without
    materialising the joined frame.
    """
    _notify(notify_fn, f"... streaming {target.name}")

    lazy_plan = (
        ratings_lf
        .with_columns(split_expr)
        .filter(pl.col("split") == split_name)
        .join(wine_features.lazy(), on="wine_id", how="left")
        .join(
            terroir_lf,
            left_on=["region_name", "country", "vintage_year"],
            right_on=["region", "country", "vintage_year"],
            how="left",
        )
        .join(producer_aggs.lazy(), on="winery_id", how="left")
        .join(wine_weights.lazy(), on="wine_id", how="left")
    )

    available = lazy_plan.collect_schema().names()
    final_cols = [c for c in col_order if c in available]
    plan = lazy_plan.select(final_cols)

    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        plan.sink_parquet(tmp)
    except Exception:
        # Fallback for any polars version that can't stream a sub-expression
        # in this plan — collect with the streaming engine, then write. Still
        # avoids the per-row Python lambdas that killed the previous build.
        plan.collect(engine="streaming").write_parquet(tmp)
    tmp.replace(target)

    row_count = pl.scan_parquet(target).select(pl.len()).collect().item()
    _notify(notify_fn, f"... wrote {row_count:,} rows to {target}")
    return row_count


def _build_col_order(
    grape_vocab: list[str],
    harmonize_vocab: list[str],
) -> list[str]:
    """Build the deterministic column order for the output parquets."""
    grape_dynamic = [f"grape_{_slugify(v)}" for v in grape_vocab] + ["grape_other"]
    pair_dynamic = [f"pair_{_slugify(v)}" for v in harmonize_vocab]
    return [*_PREFIX_COLS, *grape_dynamic, *pair_dynamic, *_SUFFIX_COLS]
