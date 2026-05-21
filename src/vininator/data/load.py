"""X-Wines dataset loader.

X-Wines (Xavier 2023, MDPI BDCC, CC0) ships three variants:

| variant | wines  | ratings    | source                     |
|---------|--------|------------|----------------------------|
| test    |    100 |      1,000 | GitHub raw (auto-fetched)  |
| slim    |  1,007 |    150,000 | Google Drive (manual drop) |
| full    | 100,646 | 21,013,536 | Google Drive (manual drop) |

The variant is selected via the `VININATOR_XWINES_VARIANT` env var (defaults
to `test`). For slim / full, the user is expected to download the CSVs from
the official Google Drive and drop them into `data/raw/` under their canonical
filenames; `download_xwines()` then normalizes them into two parquets:

- `data/raw/xwines_wines.parquet`   — one row per wine. Grapes and Vintages
                                       columns are stored as `list[str]` /
                                       `list[int]` (parsed out of the Python
                                       list literals shipped in the CSV).
- `data/raw/xwines_ratings.parquet` — one row per rating, with a parsed
                                       `Date` (`Datetime`) plus a derived
                                       `age_at_review` (`year(Date) - Vintage`).

Idempotent — re-running is a no-op when both parquets already exist and
`force=False`.
"""

from __future__ import annotations

import ast
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl

from vininator.config import (
    XWINES_GITHUB_RAW,
    XWINES_VARIANTS,
    get_settings,
)


def download_xwines(force: bool = False) -> dict[str, Path]:
    """Fetch (when possible) and normalize the X-Wines CSVs into parquets.

    Returns `{"wines": Path, "ratings": Path}`. The `test` variant is fetched
    from GitHub; `slim` / `full` must be downloaded manually from the X-Wines
    Google Drive and dropped into `data/raw/` first — see the error message
    raised when the CSV is missing.
    """
    settings = get_settings()
    settings.ensure_dirs()
    targets = {
        "wines": settings.xwines_wines_parquet,
        "ratings": settings.xwines_ratings_parquet,
    }

    if not force and all(p.exists() for p in targets.values()):
        return targets

    variant = settings.xwines_variant
    csv_paths = {
        "wines": settings.xwines_wines_csv,
        "ratings": settings.xwines_ratings_csv,
    }

    for name, csv_path in csv_paths.items():
        if csv_path.exists():
            continue
        if variant == "test":
            _fetch_github(csv_path.name, csv_path)
        else:
            raise FileNotFoundError(
                f"X-Wines {variant!r} {name} CSV not found at {csv_path}. "
                f"Download {XWINES_VARIANTS[variant][f'{name}_csv']!r} from the X-Wines "
                "Google Drive (linked from https://github.com/rogerioxavier/X-Wines) "
                f"and drop it into {csv_path.parent} before re-running."
            )

    _normalize_wines(csv_paths["wines"], targets["wines"])
    _normalize_ratings(csv_paths["ratings"], targets["ratings"])
    return targets


def _fetch_github(filename: str, target: Path) -> None:
    """Stream a file from the X-Wines GitHub raw mirror."""
    url = f"{XWINES_GITHUB_RAW}/{filename}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, target.open("wb") as f:
        while chunk := resp.read(65536):
            f.write(chunk)


def _parse_python_list_column(series: pl.Series, *, item_type: type) -> pl.Series:
    """Convert a `str` column holding `"['a', 'b']"` literals into a list column."""

    def parse(val: str | None) -> list | None:
        if val is None or val == "":
            return None
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(parsed, list):
            return None
        result: list = []
        for item in parsed:
            try:
                result.append(item_type(item))
            except (ValueError, TypeError):
                continue
        return result

    return series.map_elements(parse, return_dtype=pl.List(_polars_dtype(item_type)))


def _polars_dtype(item_type: type) -> pl.DataType:
    if item_type is int:
        return pl.Int64()
    if item_type is float:
        return pl.Float64()
    return pl.String()


def _normalize_wines(csv_path: Path, parquet_path: Path) -> None:
    """Load the wines CSV and write the canonical parquet."""
    df = pl.read_csv(csv_path, infer_schema_length=10_000)
    df = df.with_columns(
        _parse_python_list_column(df["Grapes"], item_type=str).alias("Grapes"),
        _parse_python_list_column(df["Vintages"], item_type=int).alias("Vintages"),
    )
    df.write_parquet(parquet_path)


def _normalize_ratings(csv_path: Path, parquet_path: Path) -> None:
    """Load the ratings CSV, parse `Date`, coerce `Vintage` to Int, derive `age_at_review`.

    Vintage occasionally arrives as a string (e.g. "N.V." for non-vintage wines);
    we cast it with `strict=False` so those rows land as null and propagate to a
    null `age_at_review` rather than crashing the load.
    """
    # `Vintage` ships as integer-looking years but the full variant intersperses
    # "N.V." for non-vintage wines past row ~50k — force it to String so the cast
    # below can null those out without polars' inferrer crashing mid-file.
    df = pl.read_csv(
        csv_path,
        infer_schema_length=10_000,
        schema_overrides={"Vintage": pl.String, "Date": pl.String},
    )
    df = df.with_columns(
        pl.col("Date")
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        .alias("Date"),
        pl.col("Vintage").cast(pl.Int64, strict=False).alias("Vintage"),
    )
    df = df.with_columns(
        (pl.col("Date").dt.year() - pl.col("Vintage")).alias("age_at_review")
    )
    df.write_parquet(parquet_path)


def scan_xwines_wines() -> pl.LazyFrame:
    """Lazy frame over the per-wine parquet."""
    path = get_settings().xwines_wines_parquet
    if not path.exists():
        raise FileNotFoundError(
            f"X-Wines wines parquet not found at {path}. "
            "Run `uv run vininator data download` first."
        )
    return pl.scan_parquet(path)


def scan_xwines_ratings() -> pl.LazyFrame:
    """Lazy frame over the per-rating parquet."""
    path = get_settings().xwines_ratings_parquet
    if not path.exists():
        raise FileNotFoundError(
            f"X-Wines ratings parquet not found at {path}. "
            "Run `uv run vininator data download` first."
        )
    return pl.scan_parquet(path)


def xwines_info() -> dict[str, Any]:
    """Summarize both X-Wines parquets: row counts, schemas, missingness."""
    settings = get_settings()
    return {
        "variant": settings.xwines_variant,
        "wines": _summarize(scan_xwines_wines(), settings.xwines_wines_parquet),
        "ratings": _summarize(scan_xwines_ratings(), settings.xwines_ratings_parquet),
    }


def _summarize(lf: pl.LazyFrame, path: Path) -> dict[str, Any]:
    schema = lf.collect_schema()
    columns = list(schema.names())
    row_count = lf.select(pl.len()).collect().item()
    null_counts = (
        lf.select([pl.col(c).null_count().alias(c) for c in columns]).collect().row(0)
    )
    missingness = {
        col: (nulls / row_count if row_count else 0.0)
        for col, nulls in zip(columns, null_counts, strict=True)
    }
    return {
        "path": str(path),
        "rows": row_count,
        "columns": columns,
        "dtypes": {col: str(schema[col]) for col in columns},
        "missingness": missingness,
    }
