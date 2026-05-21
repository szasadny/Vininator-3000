from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from vininator.config import get_settings
from vininator.data.load import (
    scan_xwines_ratings,
    scan_xwines_wines,
    xwines_info,
)


def _write_fake_wines(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "WineID": [1, 2, 3],
            "WineName": ["A", "B", "C"],
            "Type": ["Red", "White", "Red"],
            "Grapes": [["Merlot"], ["Chardonnay"], ["Sangiovese"]],
            "ABV": [13.0, 13.5, 14.0],
            "Country": ["France", "USA", "Italy"],
            "RegionName": ["Bordeaux", "Napa", "Tuscany"],
            "WineryID": [10, 20, 30],
            "Vintages": [[2018, 2019], [2020], [2017, 2018, 2019]],
        }
    ).write_parquet(target)


def _write_fake_ratings(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "RatingID": [1, 2, 3, 4],
            "UserID": [100, 100, 200, 300],
            "WineID": [1, 1, 2, 3],
            "Vintage": [2018, 2019, 2020, 2017],
            "Rating": [4.0, 4.5, 3.5, 5.0],
            "Date": [
                datetime(2020, 6, 1),
                datetime(2021, 7, 15),
                datetime(2021, 9, 30),
                datetime(2022, 1, 10),
            ],
            "age_at_review": [2, 2, 1, 5],
        }
    ).write_parquet(target)


def test_scan_wines_errors_when_missing(tmp_data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="wines parquet not found"):
        scan_xwines_wines()


def test_scan_ratings_errors_when_missing(tmp_data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ratings parquet not found"):
        scan_xwines_ratings()


def test_scan_wines_returns_lazyframe(tmp_data_dir: Path) -> None:
    _write_fake_wines(get_settings().xwines_wines_parquet)
    lf = scan_xwines_wines()
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().shape == (3, 9)


def test_scan_ratings_returns_lazyframe(tmp_data_dir: Path) -> None:
    _write_fake_ratings(get_settings().xwines_ratings_parquet)
    lf = scan_xwines_ratings()
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().shape == (4, 7)


def test_xwines_info_summarizes_both_parquets(tmp_data_dir: Path) -> None:
    _write_fake_wines(get_settings().xwines_wines_parquet)
    _write_fake_ratings(get_settings().xwines_ratings_parquet)

    info = xwines_info()

    assert info["variant"] == "test"
    assert info["wines"]["rows"] == 3
    assert info["ratings"]["rows"] == 4
    assert "Date" in info["ratings"]["columns"]
    assert "age_at_review" in info["ratings"]["columns"]
    assert info["ratings"]["missingness"]["age_at_review"] == 0.0
