from __future__ import annotations

from pathlib import Path

import pytest

from vininator.config import (
    XWINES_RATINGS_PARQUET,
    XWINES_VARIANTS,
    XWINES_WINES_PARQUET,
    Settings,
    get_settings,
)


def test_paths_resolve_under_data_dir(tmp_data_dir: Path) -> None:
    s = get_settings()
    assert s.data_dir == tmp_data_dir
    assert s.raw_dir == tmp_data_dir / "raw"
    assert s.interim_dir == tmp_data_dir / "interim"
    assert s.processed_dir == tmp_data_dir / "processed"
    assert s.xwines_wines_parquet == tmp_data_dir / "raw" / XWINES_WINES_PARQUET
    assert s.xwines_ratings_parquet == tmp_data_dir / "raw" / XWINES_RATINGS_PARQUET


def test_default_variant_is_test(tmp_data_dir: Path) -> None:
    s = get_settings()
    assert s.xwines_variant == "test"
    assert s.xwines_wines_csv.name == XWINES_VARIANTS["test"]["wines_csv"]
    assert s.xwines_ratings_csv.name == XWINES_VARIANTS["test"]["ratings_csv"]


def test_variant_switch_via_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VININATOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VININATOR_XWINES_VARIANT", "slim")
    get_settings.cache_clear()
    try:
        s = Settings()
        assert s.xwines_variant == "slim"
        assert s.xwines_wines_csv.name == XWINES_VARIANTS["slim"]["wines_csv"]
    finally:
        get_settings.cache_clear()


def test_invalid_variant_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VININATOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VININATOR_XWINES_VARIANT", "huge")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError):
            Settings()
    finally:
        get_settings.cache_clear()


def test_paths_are_path_instances(tmp_data_dir: Path) -> None:
    s = get_settings()
    for p in (
        s.data_dir,
        s.raw_dir,
        s.interim_dir,
        s.processed_dir,
        s.xwines_wines_parquet,
        s.xwines_ratings_parquet,
        s.xwines_wines_csv,
        s.xwines_ratings_csv,
    ):
        assert isinstance(p, Path)


def test_ensure_dirs_creates_layout(tmp_data_dir: Path) -> None:
    s = get_settings()
    s.ensure_dirs()
    assert s.raw_dir.is_dir()
    assert s.interim_dir.is_dir()
    assert s.processed_dir.is_dir()


def test_relative_data_dir_anchors_to_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`VININATOR_DATA_DIR=./data` must resolve identically from any CWD."""
    monkeypatch.setenv("VININATOR_DATA_DIR", "data")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        s = Settings()
        assert s.data_dir.is_absolute()
        assert s.data_dir.name == "data"
        assert s.data_dir.parent != tmp_path
        assert (s.data_dir.parent / "pyproject.toml").is_file()
    finally:
        get_settings.cache_clear()
