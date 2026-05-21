from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from vininator.config import Settings, get_settings


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point `get_settings()` at a fresh tmp data dir for the duration of one test."""
    monkeypatch.setenv("VININATOR_DATA_DIR", str(tmp_path))
    # Pin variant so tests are independent of whatever the local .env says.
    monkeypatch.setenv("VININATOR_XWINES_VARIANT", "test")
    get_settings.cache_clear()
    settings = get_settings()
    assert isinstance(settings, Settings)
    settings.ensure_dirs()
    yield tmp_path
    get_settings.cache_clear()
