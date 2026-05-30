"""Tests for features/text.py — pure Harmonize/Grape parsing functions.

After the 21M-rating rewrite, `encode_harmonize_multihot` and
`build_harmonize_vocab` operate on pre-parsed `list[str]` data — the raw
Harmonize string is parsed once at the wines-table scale via
`parse_harmonize_column`. These tests cover the new contract.
"""

from __future__ import annotations

import polars as pl

from vininator.features.text import (
    _slugify,
    build_grape_vocab,
    build_harmonize_vocab,
    encode_grape_multihot,
    encode_harmonize_multihot,
    grape_majority,
    parse_harmonize,
    parse_harmonize_column,
)

# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


def test_slugify_lowercases_and_replaces_spaces() -> None:
    assert _slugify("Rich Fish") == "rich_fish"


def test_slugify_strips_accents() -> None:
    assert _slugify("Café au Lait") == "cafe_au_lait"


def test_slugify_collapses_non_alnum_runs() -> None:
    assert _slugify("game & wild") == "game_wild"


def test_slugify_strips_leading_trailing_underscores() -> None:
    assert _slugify("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# parse_harmonize (single-string helper, retained for ad-hoc callers)
# ---------------------------------------------------------------------------


def test_parse_harmonize_roundtrips_literal_list() -> None:
    assert parse_harmonize("['Pork', 'Rich Fish', 'Shellfish']") == [
        "Pork",
        "Rich Fish",
        "Shellfish",
    ]


def test_parse_harmonize_returns_empty_on_none() -> None:
    assert parse_harmonize(None) == []


def test_parse_harmonize_returns_empty_on_empty_string() -> None:
    assert parse_harmonize("") == []


def test_parse_harmonize_returns_empty_on_non_list_string() -> None:
    assert parse_harmonize("not-a-list") == []


def test_parse_harmonize_returns_empty_on_dict_literal() -> None:
    assert parse_harmonize("{'a': 1}") == []


def test_parse_harmonize_single_element_list() -> None:
    assert parse_harmonize("['Beef']") == ["Beef"]


# ---------------------------------------------------------------------------
# parse_harmonize_column (vectorised column-level parser)
# ---------------------------------------------------------------------------


def test_parse_harmonize_column_extracts_tokens() -> None:
    df = pl.DataFrame(
        {"Harmonize": ["['Pork', 'Rich Fish']", "['Beef']"]}
    )
    result = parse_harmonize_column(df.lazy()).collect()
    assert result["harmonize_list"].to_list() == [
        ["Pork", "Rich Fish"],
        ["Beef"],
    ]


def test_parse_harmonize_column_null_stays_null() -> None:
    df = pl.DataFrame(
        {"Harmonize": [None]},
        schema={"Harmonize": pl.String},
    )
    result = parse_harmonize_column(df.lazy()).collect()
    assert result["harmonize_list"][0] is None


def test_parse_harmonize_column_empty_string_is_empty_list() -> None:
    df = pl.DataFrame({"Harmonize": [""]})
    result = parse_harmonize_column(df.lazy()).collect()
    assert result["harmonize_list"].to_list()[0] == []


def test_parse_harmonize_column_empty_list_literal_is_empty_list() -> None:
    df = pl.DataFrame({"Harmonize": ["[]"]})
    result = parse_harmonize_column(df.lazy()).collect()
    assert result["harmonize_list"].to_list()[0] == []


def test_parse_harmonize_column_custom_dst() -> None:
    df = pl.DataFrame({"raw": ["['X']"]})
    result = parse_harmonize_column(df.lazy(), src="raw", dst="parsed").collect()
    assert "parsed" in result.columns
    assert result["parsed"].to_list() == [["X"]]


# ---------------------------------------------------------------------------
# build_harmonize_vocab (new contract: list[str] series)
# ---------------------------------------------------------------------------


def test_build_harmonize_vocab_top_k() -> None:
    series = pl.Series(
        [
            ["Pork", "Beef", "Pork"],
            ["Beef", "Chicken"],
            ["Pork"],
        ],
        dtype=pl.List(pl.String),
    )
    # Pork=3, Beef=2, Chicken=1 → top 2 is Pork, Beef.
    vocab = build_harmonize_vocab(series, top_k=2)
    assert vocab == ["Pork", "Beef"]


def test_build_harmonize_vocab_tie_broken_lexicographically() -> None:
    series = pl.Series([["Apple", "Zebra"]], dtype=pl.List(pl.String))
    vocab = build_harmonize_vocab(series, top_k=2)
    assert vocab == ["Apple", "Zebra"]


def test_build_harmonize_vocab_skips_null_and_empty() -> None:
    series = pl.Series(
        [["Pork"], None, []],
        dtype=pl.List(pl.String),
    )
    vocab = build_harmonize_vocab(series, top_k=5)
    assert vocab == ["Pork"]


def test_build_harmonize_vocab_empty_series() -> None:
    series = pl.Series([], dtype=pl.List(pl.String))
    assert build_harmonize_vocab(series, top_k=10) == []


# ---------------------------------------------------------------------------
# encode_harmonize_multihot (new contract: harmonize_list column)
# ---------------------------------------------------------------------------


def _harmonize_frame(values: list[list[str] | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"harmonize_list": values},
        schema={"harmonize_list": pl.List(pl.String)},
    )


def test_encode_harmonize_multihot_adds_correct_columns() -> None:
    df = _harmonize_frame([["Pork", "Beef"], ["Pork"], None])
    vocab = ["Pork", "Beef"]
    result = encode_harmonize_multihot(df.lazy(), vocab).collect()

    assert "pair_pork" in result.columns
    assert "pair_beef" in result.columns
    assert result["pair_pork"].to_list() == [1, 1, 0]
    assert result["pair_beef"].to_list() == [1, 0, 0]


def test_encode_harmonize_multihot_column_count() -> None:
    df = _harmonize_frame([["X"]])
    vocab = ["Apple", "Beef", "Chicken", "Duck"]
    result = encode_harmonize_multihot(df.lazy(), vocab).collect()
    new_cols = [c for c in result.columns if c.startswith("pair_")]
    assert len(new_cols) == 4


def test_encode_harmonize_multihot_dtype_is_uint8() -> None:
    df = _harmonize_frame([["Pork"]])
    result = encode_harmonize_multihot(df.lazy(), ["Pork"]).collect()
    assert result["pair_pork"].dtype == pl.UInt8


def test_encode_harmonize_multihot_unseen_entry_is_zero() -> None:
    df = _harmonize_frame([["Game"]])
    result = encode_harmonize_multihot(df.lazy(), ["Pork"]).collect()
    assert result["pair_pork"][0] == 0


def test_encode_harmonize_multihot_preserves_other_columns() -> None:
    df = pl.DataFrame(
        {"wine_id": [1, 2], "harmonize_list": [["Pork"], ["Beef"]]},
        schema={"wine_id": pl.Int64, "harmonize_list": pl.List(pl.String)},
    )
    result = encode_harmonize_multihot(df.lazy(), ["Pork"]).collect()
    assert "wine_id" in result.columns


def test_encode_harmonize_multihot_custom_src() -> None:
    df = pl.DataFrame(
        {"foo": [["Pork"]]},
        schema={"foo": pl.List(pl.String)},
    )
    result = encode_harmonize_multihot(df.lazy(), ["Pork"], src="foo").collect()
    assert result["pair_pork"][0] == 1


# ---------------------------------------------------------------------------
# build_grape_vocab
# ---------------------------------------------------------------------------


def test_build_grape_vocab_counts_across_rows() -> None:
    series = pl.Series(
        [["Merlot", "Cabernet"], ["Merlot"], ["Cabernet"]],
        dtype=pl.List(pl.String),
    )
    vocab = build_grape_vocab(series, top_k=2)
    # Merlot=2, Cabernet=2 — tied → lexicographic: Cabernet < Merlot.
    assert set(vocab) == {"Merlot", "Cabernet"}
    assert vocab[0] == "Cabernet"


def test_build_grape_vocab_top_k_truncates() -> None:
    series = pl.Series([["A", "B", "C"]], dtype=pl.List(pl.String))
    vocab = build_grape_vocab(series, top_k=2)
    assert len(vocab) == 2


def test_build_grape_vocab_skips_null_rows() -> None:
    series = pl.Series([["Merlot"], None], dtype=pl.List(pl.String))
    vocab = build_grape_vocab(series, top_k=5)
    assert vocab == ["Merlot"]


# ---------------------------------------------------------------------------
# encode_grape_multihot
# ---------------------------------------------------------------------------


def test_encode_grape_multihot_adds_vocab_and_other_columns() -> None:
    df = pl.DataFrame(
        {"Grapes": [["Merlot", "Cabernet"], ["Pinot"], []]},
        schema={"Grapes": pl.List(pl.String)},
    )
    vocab = ["Merlot", "Cabernet"]
    result = encode_grape_multihot(df.lazy(), vocab).collect()

    assert "grape_merlot" in result.columns
    assert "grape_cabernet" in result.columns
    assert "grape_other" in result.columns
    assert result["grape_merlot"].to_list() == [1, 0, 0]
    assert result["grape_cabernet"].to_list() == [1, 0, 0]
    # Pinot is not in vocab → grape_other=1 for row 2; empty list → 0 for row 3.
    assert result["grape_other"].to_list() == [0, 1, 0]


def test_encode_grape_multihot_null_grapes_all_zero() -> None:
    df = pl.DataFrame(
        {"Grapes": [None]},
        schema={"Grapes": pl.List(pl.String)},
    )
    result = encode_grape_multihot(df.lazy(), ["Merlot"]).collect()
    assert result["grape_merlot"][0] == 0
    assert result["grape_other"][0] == 0


def test_encode_grape_multihot_dtype_is_uint8() -> None:
    df = pl.DataFrame(
        {"Grapes": [["Merlot"]]},
        schema={"Grapes": pl.List(pl.String)},
    )
    result = encode_grape_multihot(df.lazy(), ["Merlot"]).collect()
    assert result["grape_merlot"].dtype == pl.UInt8
    assert result["grape_other"].dtype == pl.UInt8


# ---------------------------------------------------------------------------
# grape_majority
# ---------------------------------------------------------------------------


def test_grape_majority_returns_first_entry() -> None:
    series = pl.Series([["Merlot", "Cabernet"], ["Pinot"]], dtype=pl.List(pl.String))
    result = grape_majority(series)
    assert result.to_list() == ["Merlot", "Pinot"]


def test_grape_majority_null_on_null_entry() -> None:
    series = pl.Series([None], dtype=pl.List(pl.String))
    result = grape_majority(series)
    assert result[0] is None
