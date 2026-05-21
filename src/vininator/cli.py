"""Typer CLI entrypoint.

`vininator` is the single orchestration surface — phases (`data`, `features`,
`train`, `eval`, `api`) hang off subcommand groups. Phase 1 ships the `data`
group; later phases add their own without touching this file's structure.
"""

from __future__ import annotations

import json

import polars as pl
import typer

from vininator.data.geocode import (
    filter_to_usable,
    geocode_regions,
    result_type_distribution,
    scan_geocode,
)
from vininator.data.load import download_xwines, xwines_info
from vininator.features.soil import build_soil_table

app = typer.Typer(
    name="vininator",
    help="Vininator 3000 — wine rating and tasting-notes predictor.",
    no_args_is_help=True,
)

data_app = typer.Typer(help="Dataset acquisition and inspection.", no_args_is_help=True)
app.add_typer(data_app, name="data")

features_app = typer.Typer(
    help="Feature engineering and terroir pipeline.", no_args_is_help=True
)
app.add_typer(features_app, name="features")


@data_app.command("download")
def data_download(
    force: bool = typer.Option(
        False, "--force", help="Re-fetch and re-normalize even if the parquets already exist."
    ),
) -> None:
    """Fetch the X-Wines variant's CSVs (test = auto, slim/full = manual drop)."""
    paths = download_xwines(force=force)
    for name, path in paths.items():
        typer.echo(f"X-Wines {name:8s} parquet: {path}")


@data_app.command("info")
def data_info() -> None:
    """Print row counts, schemas, and missingness for both X-Wines parquets."""
    typer.echo(json.dumps(xwines_info(), indent=2, default=str))


@features_app.command("geocode")
def features_geocode(
    force: bool = typer.Option(
        False, "--force", help="Discard the existing geocode cache and re-fetch everything."
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Process at most N regions this run (resumable — re-run to cover the rest).",
    ),
) -> None:
    """Geocode unique RegionName values to (lat, lon) via Nominatim.

    Rate-limited to 1 req/sec per Nominatim's TOS. ~2,160 regions on the full
    variant => ~35 minutes for a cold run; subsequent runs are no-ops. Progress
    is printed at every checkpoint flush; Ctrl+C is safe — partial results are
    persisted before the process exits.
    """

    def progress(done: int, total: int) -> None:
        pct = 100.0 * done / total if total else 100.0
        typer.echo(f"... geocoded {done}/{total} ({pct:.1f}%)")

    path = geocode_regions(
        force=force, limit=limit, progress_fn=progress, notify_fn=typer.echo
    )
    typer.echo(f"Geocode cache: {path}")


@features_app.command("geocode-audit")
def features_geocode_audit() -> None:
    """Print the `result_type` distribution from `geocode.parquet`.

    Helps tune the blacklist in `config.GEOCODE_BAD_RESULT_TYPES` against the
    real data — Nominatim tags wine regions with a wildly varied set of
    `result_type` values, so a tight whitelist drops real regions.
    """
    import unicodedata

    def _ascii(s: str | None) -> str:
        """Fold to ASCII so Windows cp1252 consoles don't crash on accents."""
        if s is None:
            return ""
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    df = scan_geocode().collect()
    n_total = df.height
    n_ok = df.filter(pl.col("status") == "ok").height
    n_kept = filter_to_usable(df).height
    typer.echo(f"total rows:           {n_total}")
    typer.echo(f"status='ok':          {n_ok}")
    typer.echo(f"after blacklist:      {n_kept}  (drops {n_ok - n_kept} rows)")
    typer.echo("")
    typer.echo(f"{'result_type':25s} {'n':>5s}  {'blacklisted':>11s}  example_region")
    typer.echo("-" * 80)
    for row in result_type_distribution(df).iter_rows(named=True):
        rt = row["result_type"] if row["result_type"] is not None else "(null)"
        mark = "yes" if row["blacklisted"] else ""
        typer.echo(f"{rt:25s} {row['n']:>5d}  {mark:>11s}  {_ascii(row['example_region'])}")


@features_app.command("soil")
def features_soil(
    force: bool = typer.Option(
        False, "--force", help="Discard the existing soil cache and re-fetch everything."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Process at most N regions this run (resumable)."
    ),
) -> None:
    """Pull SoilGrids 0-30cm profile + Open-Elevation DEM for every geocoded region.

    Reads `data/interim/geocode.parquet` (status='ok' rows only). Each region
    costs ~9 SoilGrids + 1 Open-Elevation HTTP calls; results land in
    `data/interim/soil.parquet`. Resumable: re-runs only process missing rows.
    """

    def progress(done: int, total: int) -> None:
        pct = 100.0 * done / total if total else 100.0
        typer.echo(f"... soil {done}/{total} ({pct:.1f}%)")

    path = build_soil_table(
        force=force, limit=limit, progress_fn=progress, notify_fn=typer.echo
    )
    typer.echo(f"Soil table: {path}")


if __name__ == "__main__":
    app()
