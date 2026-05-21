from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from geopy.exc import GeocoderRateLimited, GeocoderTimedOut

from vininator.config import GEOCODE_BAD_RESULT_TYPES, get_settings
from vininator.data.geocode import (
    GEOCODE_SCHEMA,
    filter_to_usable,
    geocode_one,
    geocode_regions,
    load_unique_regions,
    result_type_distribution,
    scan_geocode,
)


class _FakeLocation:
    """Minimal stand-in for `geopy.location.Location`.

    Geocode result objects expose `.latitude`, `.longitude`, `.address`, `.raw`;
    that's the whole contract we depend on.
    """

    def __init__(self, lat: float, lon: float, address: str, raw_type: str = "administrative"):
        self.latitude = lat
        self.longitude = lon
        self.address = address
        self.raw = {"type": raw_type}


def _write_fake_wines(target: Path) -> None:
    """Three wines, two unique regions, plus a duplicate to exercise dedup."""
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "WineID": [1, 2, 3, 4],
            "RegionName": ["Bordeaux", "Napa Valley", "Bordeaux", "  Mendoza  "],
            "Country": ["France", "USA", "France", "Argentina"],
        }
    ).write_parquet(target)


def _write_n_fake_wines(target: Path, n: int) -> None:
    """One row per unique synthetic region — handy for backoff tests that need
    a known number of todo rows in a deterministic order (sort = country asc)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "WineID": list(range(1, n + 1)),
            # Country and region prefixes share the same numeric suffix so the
            # sorted (country, region) order matches the suffix order — tests
            # can reason about which row is hit first.
            "RegionName": [f"R{i:02d}" for i in range(1, n + 1)],
            "Country": [f"C{i:02d}" for i in range(1, n + 1)],
        }
    ).write_parquet(target)


def test_load_unique_regions_dedupes_and_strips(tmp_data_dir: Path) -> None:
    _write_fake_wines(get_settings().xwines_wines_parquet)
    df = load_unique_regions()
    assert set(df.columns) == {"region", "country"}
    pairs = set(zip(df["region"].to_list(), df["country"].to_list(), strict=True))
    assert pairs == {
        ("Bordeaux", "France"),
        ("Napa Valley", "USA"),
        ("Mendoza", "Argentina"),
    }


def test_geocode_one_ok() -> None:
    def fake(query: str) -> _FakeLocation:
        assert query == "Bordeaux, France"
        return _FakeLocation(44.84, -0.58, "Bordeaux, Gironde, France")

    row = geocode_one("Bordeaux", "France", fake)
    assert row["status"] == "ok"
    assert row["lat"] == pytest.approx(44.84)
    assert row["lon"] == pytest.approx(-0.58)
    assert row["raw_address"] == "Bordeaux, Gironde, France"
    assert row["result_type"] == "administrative"
    assert row["error"] is None


def test_geocode_one_not_found() -> None:
    row = geocode_one("Atlantis", "Nowhere", lambda _q: None)
    assert row["status"] == "not_found"
    assert row["lat"] is None
    assert row["lon"] is None
    assert row["error"] is None


def test_geocode_one_error_on_geocoder_exception() -> None:
    def boom(_query: str) -> Any:
        raise GeocoderTimedOut("upstream slow")

    row = geocode_one("Bordeaux", "France", boom)
    assert row["status"] == "error"
    assert row["lat"] is None
    assert "GeocoderTimedOut" in row["error"]
    assert "upstream slow" in row["error"]


def test_geocode_one_no_country_query_string() -> None:
    captured: dict[str, str] = {}

    def fake(query: str) -> _FakeLocation:
        captured["query"] = query
        return _FakeLocation(0.0, 0.0, "X")

    geocode_one("LoneRegion", None, fake)
    assert captured["query"] == "LoneRegion"


def test_geocode_regions_writes_cache_and_resumes(tmp_data_dir: Path) -> None:
    _write_fake_wines(get_settings().xwines_wines_parquet)

    coords: dict[str, tuple[float, float]] = {
        "Bordeaux, France": (44.84, -0.58),
        "Napa Valley, USA": (38.5, -122.3),
        "Mendoza, Argentina": (-32.89, -68.85),
    }
    calls: list[str] = []

    def fake(query: str) -> _FakeLocation | None:
        calls.append(query)
        lat, lon = coords[query]
        return _FakeLocation(lat, lon, query)

    path = geocode_regions(geocode_fn=fake)
    assert path == get_settings().geocode_parquet
    assert path.exists()
    df = pl.read_parquet(path)
    assert df.shape == (3, len(GEOCODE_SCHEMA))
    assert (df["status"] == "ok").all()
    assert sorted(calls) == sorted(coords)

    # Second pass: nothing new to do, so the fake must not be called again.
    calls.clear()
    geocode_regions(geocode_fn=fake)
    assert calls == []
    assert pl.read_parquet(path).shape == (3, len(GEOCODE_SCHEMA))


def test_geocode_regions_only_geocodes_new_rows(tmp_data_dir: Path) -> None:
    _write_fake_wines(get_settings().xwines_wines_parquet)

    # Pre-seed cache with just Bordeaux so only Napa + Mendoza are todo.
    cache_path = get_settings().geocode_parquet
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {
                "region": "Bordeaux",
                "country": "France",
                "lat": 44.84,
                "lon": -0.58,
                "raw_address": "Bordeaux",
                "result_type": "administrative",
                "status": "ok",
                "error": None,
                "fetched_at": datetime(2025, 1, 1, tzinfo=UTC),
            }
        ],
        schema=GEOCODE_SCHEMA,
    ).write_parquet(cache_path)

    calls: list[str] = []

    def fake(query: str) -> _FakeLocation:
        calls.append(query)
        return _FakeLocation(1.0, 2.0, query)

    geocode_regions(geocode_fn=fake)
    assert sorted(calls) == ["Mendoza, Argentina", "Napa Valley, USA"]
    df = pl.read_parquet(cache_path)
    assert df.shape == (3, len(GEOCODE_SCHEMA))


def test_geocode_regions_limit_caps_work(tmp_data_dir: Path) -> None:
    _write_fake_wines(get_settings().xwines_wines_parquet)
    calls: list[str] = []

    def fake(query: str) -> _FakeLocation:
        calls.append(query)
        return _FakeLocation(0.0, 0.0, query)

    geocode_regions(limit=1, geocode_fn=fake)
    assert len(calls) == 1
    assert pl.read_parquet(get_settings().geocode_parquet).shape[0] == 1


def test_geocode_regions_keeps_failures_in_cache(tmp_data_dir: Path) -> None:
    _write_fake_wines(get_settings().xwines_wines_parquet)

    def fake(_query: str) -> None:
        return None

    geocode_regions(geocode_fn=fake)
    df = pl.read_parquet(get_settings().geocode_parquet)
    assert df.shape[0] == 3
    assert (df["status"] == "not_found").all()
    # Failed rows must still be considered "done" — next run is a no-op.
    calls: list[str] = []

    def fake2(query: str) -> None:
        calls.append(query)
        return None

    geocode_regions(geocode_fn=fake2)
    assert calls == []


def test_geocode_regions_force_clears_cache(tmp_data_dir: Path) -> None:
    _write_fake_wines(get_settings().xwines_wines_parquet)

    def fake(query: str) -> _FakeLocation:
        return _FakeLocation(0.0, 0.0, query)

    geocode_regions(geocode_fn=fake)
    n_before = pl.read_parquet(get_settings().geocode_parquet).shape[0]

    calls: list[str] = []

    def fake2(query: str) -> _FakeLocation:
        calls.append(query)
        return _FakeLocation(1.0, 1.0, query)

    geocode_regions(force=True, geocode_fn=fake2)
    assert len(calls) == n_before  # all rows re-fetched


def test_scan_geocode_errors_when_missing(tmp_data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Geocode parquet not found"):
        scan_geocode()


def test_progress_fn_called_at_checkpoints_and_end(tmp_data_dir: Path) -> None:
    _write_fake_wines(get_settings().xwines_wines_parquet)

    def fake(query: str) -> _FakeLocation:
        return _FakeLocation(0.0, 0.0, query)

    events: list[tuple[int, int]] = []
    geocode_regions(
        geocode_fn=fake,
        progress_fn=lambda done, total: events.append((done, total)),
        checkpoint_every=2,
    )
    # 3 todo rows, checkpoint_every=2 -> one mid-run flush at i=2, one final at i=3.
    assert events == [(2, 3), (3, 3)]


def test_checkpoint_persists_partial_on_crash(tmp_data_dir: Path) -> None:
    """A geocode_fn that raises mid-loop must not lose already-flushed work."""
    _write_fake_wines(get_settings().xwines_wines_parquet)

    calls: list[str] = []

    def fake(query: str) -> _FakeLocation:
        calls.append(query)
        if len(calls) == 3:
            raise RuntimeError("kaboom on row 3")
        return _FakeLocation(0.0, 0.0, query)

    with pytest.raises(RuntimeError, match="kaboom"):
        geocode_regions(geocode_fn=fake, checkpoint_every=1)

    # Two successful rows checkpointed before the crash, both persisted.
    df = pl.read_parquet(get_settings().geocode_parquet)
    assert df.shape[0] == 2
    assert (df["status"] == "ok").all()


def test_resume_after_crash_picks_up_where_we_left_off(tmp_data_dir: Path) -> None:
    """Crash, then re-run with a working fake — only the missing rows hit the API."""
    _write_fake_wines(get_settings().xwines_wines_parquet)

    crash_calls: list[str] = []

    def crashing(query: str) -> _FakeLocation:
        crash_calls.append(query)
        if len(crash_calls) == 3:
            raise RuntimeError("kaboom")
        return _FakeLocation(0.0, 0.0, query)

    with pytest.raises(RuntimeError):
        geocode_regions(geocode_fn=crashing, checkpoint_every=1)

    # Resume.
    resume_calls: list[str] = []

    def working(query: str) -> _FakeLocation:
        resume_calls.append(query)
        return _FakeLocation(0.0, 0.0, query)

    geocode_regions(geocode_fn=working)
    assert len(resume_calls) == 1  # only the one survivor of the crash
    df = pl.read_parquet(get_settings().geocode_parquet)
    assert df.shape[0] == 3
    assert (df["status"] == "ok").all()


# ---------------------------------------------------------------------------
# Rate-limit handling (Nominatim 429).
# ---------------------------------------------------------------------------


def test_geocode_one_reraises_rate_limited() -> None:
    """Rate-limit must NOT be caught — the outer loop applies backoff. If we
    swallowed it here, the row would be cached as `status='error'` and the
    anti-join would skip retrying it on the next run."""

    def fake(_q: str) -> None:
        raise GeocoderRateLimited("nope", retry_after=15)

    with pytest.raises(GeocoderRateLimited):
        geocode_one("X", "Y", fake)


def test_rate_limited_rows_are_not_cached(tmp_data_dir: Path) -> None:
    """When every call 429s, nothing lands in the cache — the next run via the
    anti-join must see all rows still as todo."""
    _write_n_fake_wines(get_settings().xwines_wines_parquet, n=3)

    def always_429(_q: str) -> None:
        raise GeocoderRateLimited("nope")

    geocode_regions(
        geocode_fn=always_429,
        sleep_fn=lambda _s: None,
        backoff_base_sec=1.0,
        backoff_cap_sec=10.0,
    )
    # The parquet may not exist (no successful rows to flush) OR may exist as
    # an empty frame with the schema — both are valid "nothing cached".
    cache = get_settings().geocode_parquet
    if cache.exists():
        df = pl.read_parquet(cache)
        assert df.shape[0] == 0


def test_backoff_grows_exponentially(tmp_data_dir: Path) -> None:
    """Consecutive 429s double the sleep: 10s, 20s, 40s."""
    _write_n_fake_wines(get_settings().xwines_wines_parquet, n=3)

    def always_429(_q: str) -> None:
        raise GeocoderRateLimited("nope")

    sleeps: list[float] = []
    geocode_regions(
        geocode_fn=always_429,
        sleep_fn=sleeps.append,
        backoff_base_sec=10.0,
        backoff_cap_sec=1_000.0,
    )
    assert sleeps == [10.0, 20.0, 40.0]


def test_backoff_capped_at_cap(tmp_data_dir: Path) -> None:
    """Once the exponential would exceed cap, every subsequent sleep is cap."""
    _write_n_fake_wines(get_settings().xwines_wines_parquet, n=4)

    def always_429(_q: str) -> None:
        raise GeocoderRateLimited("nope")

    sleeps: list[float] = []
    geocode_regions(
        geocode_fn=always_429,
        sleep_fn=sleeps.append,
        backoff_base_sec=100.0,
        backoff_cap_sec=250.0,
    )
    # 100, 200, then 400 would breach cap → 250, then 800 → 250.
    assert sleeps == [100.0, 200.0, 250.0, 250.0]


def test_retry_after_honored_when_larger_than_backoff(tmp_data_dir: Path) -> None:
    """If the server tells us how long to wait, respect it when it's longer
    than the computed backoff."""
    _write_n_fake_wines(get_settings().xwines_wines_parquet, n=1)

    def server_says_wait(_q: str) -> None:
        raise GeocoderRateLimited("nope", retry_after=200)

    sleeps: list[float] = []
    geocode_regions(
        geocode_fn=server_says_wait,
        sleep_fn=sleeps.append,
        backoff_base_sec=10.0,
        backoff_cap_sec=1_000.0,
    )
    # Server's 200s wins over backoff's 10s.
    assert sleeps == [200.0]


def test_consecutive_counter_resets_after_success(tmp_data_dir: Path) -> None:
    """A successful row resets the backoff counter — the next 429 sleeps the
    base interval, not the exponentiated one."""
    _write_n_fake_wines(get_settings().xwines_wines_parquet, n=4)
    # Calls in sort order: C01/R01, C02/R02, C03/R03, C04/R04.
    # Pattern: 429, 429, ok, 429.

    sequence = iter([429, 429, "ok", 429])

    def patterned(query: str) -> _FakeLocation:
        nxt = next(sequence)
        if nxt == 429:
            raise GeocoderRateLimited("nope")
        return _FakeLocation(0.0, 0.0, query)

    sleeps: list[float] = []
    geocode_regions(
        geocode_fn=patterned,
        sleep_fn=sleeps.append,
        backoff_base_sec=10.0,
        backoff_cap_sec=1_000.0,
    )
    # 1st 429: sleep 10 (cnt=1). 2nd 429: sleep 20 (cnt=2). Success resets.
    # 4th call 429s: sleep 10 (cnt=1 again).
    assert sleeps == [10.0, 20.0, 10.0]

    # Only the successful row is cached.
    df = pl.read_parquet(get_settings().geocode_parquet)
    assert df.shape[0] == 1
    assert df["status"].to_list() == ["ok"]


def test_pending_flushed_before_sleep(tmp_data_dir: Path) -> None:
    """A Ctrl+C during the backoff sleep must not lose already-completed rows.
    Verify by asserting the cache is already written when sleep_fn fires."""
    _write_n_fake_wines(get_settings().xwines_wines_parquet, n=3)
    # Pattern: ok, ok, 429.

    sequence = iter(["ok", "ok", 429])

    def patterned(query: str) -> _FakeLocation:
        nxt = next(sequence)
        if nxt == 429:
            raise GeocoderRateLimited("nope")
        return _FakeLocation(0.0, 0.0, query)

    cache_rows_at_sleep: list[int] = []

    def sleep_fn(_s: float) -> None:
        cache = get_settings().geocode_parquet
        cache_rows_at_sleep.append(pl.read_parquet(cache).shape[0] if cache.exists() else 0)

    geocode_regions(
        geocode_fn=patterned,
        sleep_fn=sleep_fn,
        backoff_base_sec=1.0,
        backoff_cap_sec=10.0,
        # Force the cache to only be written by the rate-limit flush, not by
        # checkpoint_every (which is 25 by default and wouldn't fire here).
    )
    # Sleep was triggered once (the 429 on row 3). At that point the 2 ok rows
    # must already be in the cache.
    assert cache_rows_at_sleep == [2]


def test_notify_fn_receives_backoff_message(tmp_data_dir: Path) -> None:
    """The CLI relies on notify_fn for visibility during long sleeps."""
    _write_n_fake_wines(get_settings().xwines_wines_parquet, n=1)

    def always_429(_q: str) -> None:
        raise GeocoderRateLimited("nope")

    messages: list[str] = []
    geocode_regions(
        geocode_fn=always_429,
        sleep_fn=lambda _s: None,
        notify_fn=messages.append,
        backoff_base_sec=10.0,
        backoff_cap_sec=100.0,
    )
    assert len(messages) == 1
    assert "rate-limited" in messages[0].lower()
    assert "sleeping" in messages[0].lower()


# ---------------------------------------------------------------------------
# filter_to_usable (PR2.5) — result_type blacklist + status='ok' gate
# ---------------------------------------------------------------------------


def _geocode_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Hand-build a geocode dataframe with sensible defaults; tests override."""
    defaults: dict[str, Any] = {
        "region": "",
        "country": None,
        "lat": 0.0,
        "lon": 0.0,
        "raw_address": None,
        "result_type": "administrative",
        "status": "ok",
        "error": None,
        "fetched_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    return pl.DataFrame([{**defaults, **r} for r in rows], schema=GEOCODE_SCHEMA)


def test_filter_to_usable_drops_non_ok_status() -> None:
    df = _geocode_df([
        {"region": "A", "status": "ok"},
        {"region": "B", "status": "not_found"},
        {"region": "C", "status": "error"},
    ])
    out = filter_to_usable(df)
    assert out["region"].to_list() == ["A"]


def test_filter_to_usable_drops_blacklisted_result_types() -> None:
    bad = next(iter(GEOCODE_BAD_RESULT_TYPES))  # whatever's in the blacklist
    df = _geocode_df([
        {"region": "Good",  "result_type": "administrative"},
        {"region": "Junk",  "result_type": bad},
    ])
    out = filter_to_usable(df)
    assert out["region"].to_list() == ["Good"]


def test_filter_to_usable_keeps_null_result_type() -> None:
    """A status='ok' row with result_type=null shouldn't be silently dropped
    by the blacklist check — we don't know what it is, but we know it's ok."""
    df = _geocode_df([
        {"region": "Mystery", "result_type": None},
    ])
    out = filter_to_usable(df)
    assert out["region"].to_list() == ["Mystery"]


def test_filter_to_usable_keeps_long_tail_of_real_wine_types() -> None:
    """The audit showed many real wine regions land under odd result_types
    (restaurant, residential, river, volcano, peak). These must pass."""
    df = _geocode_df([
        {"region": "Cote de Beaune",       "result_type": "restaurant"},
        {"region": "Limestone Coast",      "result_type": "residential"},
        {"region": "Etna",                 "result_type": "volcano"},
        {"region": "La Clape",             "result_type": "peak"},
        {"region": "Nahe",                 "result_type": "river"},
        {"region": "Hunter Valley",        "result_type": "valley"},
        {"region": "Crete",                "result_type": "island"},
        {"region": "Bonnes-Mares Grand Cru", "result_type": "vineyard"},
    ])
    out = filter_to_usable(df)
    assert out.shape[0] == 8  # none dropped


def test_filter_to_usable_returns_a_dataframe_not_lazyframe() -> None:
    out = filter_to_usable(_geocode_df([{"region": "X"}]))
    assert isinstance(out, pl.DataFrame)


def test_result_type_distribution_orders_by_count_desc() -> None:
    df = _geocode_df([
        {"region": "a", "result_type": "vineyard"},
        {"region": "b", "result_type": "administrative"},
        {"region": "c", "result_type": "administrative"},
        {"region": "d", "result_type": "administrative"},
        {"region": "e", "result_type": "village"},
        {"region": "f", "result_type": "village"},
    ])
    dist = result_type_distribution(df)
    assert dist["result_type"].to_list() == ["administrative", "village", "vineyard"]
    assert dist["n"].to_list() == [3, 2, 1]
    assert dist["example_region"].to_list() == ["b", "e", "a"]


def test_result_type_distribution_flags_blacklisted() -> None:
    bad = next(iter(GEOCODE_BAD_RESULT_TYPES))
    df = _geocode_df([
        {"region": "ok",  "result_type": "administrative"},
        {"region": "bad", "result_type": bad},
    ])
    dist = result_type_distribution(df)
    by_type = {r["result_type"]: r["blacklisted"] for r in dist.iter_rows(named=True)}
    assert by_type["administrative"] is False
    assert by_type[bad] is True


def test_result_type_distribution_only_counts_ok_rows() -> None:
    df = _geocode_df([
        {"region": "a", "result_type": "village", "status": "ok"},
        {"region": "b", "result_type": "village", "status": "not_found"},
    ])
    dist = result_type_distribution(df)
    assert dist["n"].to_list() == [1]
