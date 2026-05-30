"""Pure functions for parsing structured text columns into model features.

X-Wines ships no review text. The only "text" we have is:
  - `Grapes` — already a list[str] after load.py's _parse_python_list_column.
  - `Harmonize` — food-pairing list still encoded as a Python-literal string
    (e.g. "['Pork', 'Rich Fish', 'Shellfish']") because load.py left it as-is.

Both become multi-hot feature blocks in the final training table. Vocabularies
are built from the training fold only and applied to all three splits so unseen
grapes or pairings in test/future-vintage rows produce all-zero rows rather than
leaking test-set distribution into the vocabulary.

Everything in the hot path is native polars list/regex — no per-row Python
callbacks. The legacy `parse_harmonize` is kept as a single-string helper for
ad-hoc parsing and the doctest, but the column-level pipeline uses
`parse_harmonize_column` (vectorised regex extract) plus `list.contains` /
`list.eval` for the multi-hot encoding.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Sequence

import polars as pl


def _slugify(name: str) -> str:
    """Stable ASCII column-name fragment from an arbitrary string.

    Lowercases, strips accents, replaces runs of non-alphanumeric characters
    with a single underscore, and strips leading/trailing underscores.

    >>> _slugify("Soft Cheese")
    'soft_cheese'
    >>> _slugify("Café au Lait")
    'cafe_au_lait'
    """
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    lower = ascii_str.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
    return slug or "unknown"


def parse_harmonize(s: str | None) -> list[str]:
    """Parse a single raw Harmonize string into a list of food-pairing strings.

    Single-string helper retained for ad-hoc callers and the doctest. The
    column-level pipeline uses `parse_harmonize_column` instead — it runs the
    same extraction in vectorised Rust and avoids the GIL.

    Returns an empty list on None, empty string, or any parse failure.

    >>> parse_harmonize("['Pork', 'Rich Fish']")
    ['Pork', 'Rich Fish']
    >>> parse_harmonize(None)
    []
    >>> parse_harmonize("not-a-list")
    []
    """
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item is not None]


def parse_harmonize_column(
    lf: pl.LazyFrame, src: str = "Harmonize", dst: str = "harmonize_list"
) -> pl.LazyFrame:
    """Parse the raw Harmonize string column into a list[str] column.

    Uses `str.extract_all` to grab every single-quoted token from the
    Python-literal source (e.g. ``"['Pork', 'Rich Fish']"``) and then strips
    the surrounding quotes with `list.eval(str.strip_chars)`. Both ops are
    vectorised in Rust — no Python callback, no GIL.

    Null inputs land as null. Empty strings and ``"[]"`` land as empty lists,
    matching the contract of `parse_harmonize`.
    """
    return lf.with_columns(
        pl.col(src)
        .str.extract_all(r"'[^']*'")
        .list.eval(pl.element().str.strip_chars("'"))
        .alias(dst)
    )


def build_harmonize_vocab(harmonize_list_series: pl.Series, top_k: int) -> list[str]:
    """Return the top-k food pairings by frequency, ties broken lexicographically.

    `harmonize_list_series` must be a List(String) series — i.e. the column
    after `parse_harmonize_column`. Null and empty-string entries are silently
    skipped.

    Counts every occurrence in the series; if the caller wants per-wine counts
    they should dedupe to unique wines first (the rating-level series repeats
    the same value many times per wine).

    The returned list is in descending-frequency order (most common first).
    """
    counts = (
        pl.DataFrame({"x": harmonize_list_series})
        .lazy()
        .explode("x")
        .drop_nulls("x")
        .filter(pl.col("x") != "")
        .group_by("x")
        .agg(pl.len().alias("n"))
        .sort(["n", "x"], descending=[True, False])
        .head(top_k)
        .select("x")
        .collect()
    )
    return counts["x"].to_list()


def build_grape_vocab(grapes_series: pl.Series, top_k: int) -> list[str]:
    """Return the top-k grape varieties by frequency across rows.

    `grapes_series` must be a List(String) series — i.e. the `Grapes` column
    after load.py's normalization. Null and empty-string entries are skipped.

    Returns varieties in descending-frequency order, ties broken lexicographically.
    """
    counts = (
        pl.DataFrame({"x": grapes_series})
        .lazy()
        .explode("x")
        .drop_nulls("x")
        .filter(pl.col("x") != "")
        .group_by("x")
        .agg(pl.len().alias("n"))
        .sort(["n", "x"], descending=[True, False])
        .head(top_k)
        .select("x")
        .collect()
    )
    return counts["x"].to_list()


def encode_harmonize_multihot(
    frame: pl.LazyFrame,
    vocab: Sequence[str],
    src: str = "harmonize_list",
) -> pl.LazyFrame:
    """Add one UInt8 column per vocabulary entry (prefix: ``pair_``).

    Expects ``src`` to be a List(String) column — call `parse_harmonize_column`
    upstream if the source is still the raw Harmonize string. Each output
    column is 1 when the pairing appears in the row's list, 0 otherwise. Null
    list rows get all-zero columns.

    Implementation uses `list.contains` — no `map_elements`, no GIL.
    """
    exprs = [
        pl.col(src)
        .list.contains(pl.lit(v))
        .fill_null(False)
        .cast(pl.UInt8)
        .alias(f"pair_{_slugify(v)}")
        for v in vocab
    ]
    return frame.with_columns(exprs)


def encode_grape_multihot(
    frame: pl.LazyFrame,
    vocab: Sequence[str],
    src: str = "Grapes",
) -> pl.LazyFrame:
    """Add one UInt8 column per vocab entry (prefix: ``grape_``) plus ``grape_other``.

    ``grape_other`` is 1 when at least one grape in the row's list is not in
    the vocab, 0 otherwise. Null grape rows get all-zero columns.

    Computed as ``len(grapes) > sum(in-vocab indicators)`` — fully vectorised,
    no `map_elements`. Relies on grape names being unique within each wine
    (X-Wines convention), so the count of in-vocab indicators equals the
    count of in-vocab grapes.
    """
    grape_cols = [f"grape_{_slugify(v)}" for v in vocab]
    multihot_exprs = [
        pl.col(src)
        .list.contains(pl.lit(v))
        .fill_null(False)
        .cast(pl.UInt8)
        .alias(col_name)
        for col_name, v in zip(grape_cols, vocab, strict=True)
    ]
    frame = frame.with_columns(multihot_exprs)
    if grape_cols:
        other_expr = (
            pl.col(src).is_not_null()
            & (pl.col(src).list.len() > pl.sum_horizontal(*grape_cols))
        )
    else:
        # No vocab → every non-null wine with at least one grape is "other".
        other_expr = pl.col(src).is_not_null() & (pl.col(src).list.len() > 0)
    return frame.with_columns(
        other_expr.fill_null(False).cast(pl.UInt8).alias("grape_other")
    )


def grape_majority(grapes_series: pl.Series) -> pl.Series:
    """Return the first (dominant) grape from each row's Grapes list.

    Null Grapes → null. Single-grape wines return that grape. Blends return the
    first listed grape (X-Wines convention: most prominent variety first).
    """
    return grapes_series.list.first()
