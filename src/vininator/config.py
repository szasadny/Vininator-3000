"""Single source of truth for filesystem paths and external-service config.

All other modules import the resolved `settings` singleton — they never
construct paths or read environment variables themselves. This keeps the
layout decision in one place and makes tests trivial: override `DATA_DIR`
via env or fixture and everything downstream follows.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

XWINES_GITHUB_RAW = (
    "https://raw.githubusercontent.com/rogerioxavier/X-Wines/main/Dataset/last"
)

# Filename templates per variant. The slim/full variants live on the project's
# Google Drive (not in-repo); the test variant ships directly from GitHub.
XWINES_VARIANTS: dict[str, dict[str, str]] = {
    "test": {
        "wines_csv":   "XWines_Test_100_wines.csv",
        "ratings_csv": "XWines_Test_1K_ratings.csv",
    },
    "slim": {
        "wines_csv":   "XWines_Slim_1K_wines.csv",
        "ratings_csv": "XWines_Slim_150K_ratings.csv",
    },
    "full": {
        "wines_csv":   "XWines_Full_100K_wines.csv",
        "ratings_csv": "XWines_Full_21M_ratings.csv",
    },
}

XWINES_WINES_PARQUET = "xwines_wines.parquet"
XWINES_RATINGS_PARQUET = "xwines_ratings.parquet"

# Geocoding — Nominatim asks for a contactable user agent in their TOS and caps
# unauthenticated traffic at 1 req/sec; both are non-negotiable. The default
# socket timeout (1s) is too aggressive for the free tier — bump to 10s so a
# slow response cleanly exhausts the RateLimiter's retries instead of failing
# the whole row.
#
# The contact string is supplied via env (`VININATOR_NOMINATIM_CONTACT`) — never
# hardcoded here, because this file is committed and a personal email leaking
# into git history is a one-way mistake. The default is the project repo URL,
# which satisfies Nominatim's "contactable operator" requirement without
# exposing any individual.
NOMINATIM_USER_AGENT_TEMPLATE = "vininator-3000/0.1 ({contact})"
NOMINATIM_DEFAULT_CONTACT = "+https://github.com/vininator-3000/vininator-3000"
NOMINATIM_RATE_LIMIT_SEC = 1.0
NOMINATIM_TIMEOUT_SEC = 10.0

# Flush the geocode cache every N rows so a Ctrl+C 30 minutes into a 36-minute
# cold run loses at most N rows of work, not the whole run.
GEOCODE_CHECKPOINT_EVERY = 25

# Outer backoff when Nominatim returns 429 and the RateLimiter exhausts its own
# retries. Pause increasingly long between offending rows so we stop hammering
# the server while still making progress when the cool-down clears.
GEOCODE_BACKOFF_BASE_SEC = 30.0
GEOCODE_BACKOFF_CAP_SEC = 300.0

GEOCODE_PARQUET = "geocode.parquet"

# Geocode `result_type` blacklist — applied at read time by `filter_to_usable`.
# Audit of the real 2160-region geocode showed that Nominatim's `result_type`
# is wildly inconsistent for wine regions: many real appellations come back
# tagged as "restaurant", "residential", "river", "volcano", "peak", etc. A
# whitelist would drop hundreds of real wine regions. So we blacklist only
# entity types that are unambiguously non-region (transport stops, postal
# infrastructure, financial POIs, healthcare, fuel/parking) and keep the long
# tail. Some bad rows (Buenos Aires-style city centroids) still slip through;
# the downstream null detection in `compute_soil_features` catches them.
GEOCODE_BAD_RESULT_TYPES: frozenset[str] = frozenset({
    "bus_stop", "station", "platform", "halt",
    "post_office", "post_box",
    "atm", "bank", "fuel",
    "hospital", "clinic", "school", "college",
    "parking", "elevator", "fire_station",
})

# ---------------------------------------------------------------------------
# Soil + DEM (PR2)
# ---------------------------------------------------------------------------
#
# SoilGrids v2 (ISRIC) is free, no auth, but the public endpoint is flaky on
# single-pixel queries. We buffer the centroid into a 3x3 grid (~500 m square)
# and average the per-property `mean` values across the 9 points so that one
# bad pixel can't flip drainage_class between runs.

SOILGRIDS_BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# SoilGrids does NOT ship a 0-30cm aggregate. It exposes three topsoil bands
# that we depth-weight-average to get a single value per property:
#   0-5cm   thickness 5  → weight 5/30
#   5-15cm  thickness 10 → weight 10/30
#   15-30cm thickness 15 → weight 15/30
SOILGRIDS_DEPTHS: tuple[str, ...] = ("0-5cm", "5-15cm", "15-30cm")
SOILGRIDS_DEPTH_WEIGHTS: dict[str, float] = {
    "0-5cm":  5.0 / 30.0,
    "5-15cm": 10.0 / 30.0,
    "15-30cm": 15.0 / 30.0,
}

SOILGRIDS_PROPERTIES: tuple[str, ...] = (
    "phh2o", "cec", "clay", "sand", "silt", "soc", "bdod", "cfvo",
)
SOILGRIDS_BUFFER_DEG = 0.005
SOILGRIDS_RETRIES = 3
SOILGRIDS_BACKOFF_SEC: tuple[float, ...] = (1.0, 4.0, 16.0)
SOILGRIDS_TIMEOUT_SEC = 30.0
SOILGRIDS_RATE_LIMIT_SEC = 1.0  # be polite to ISRIC

# SoilGrids v2 returns d-factored integers — `unit_measure.d_factor` in the
# response. Divide the raw `mean` by this number to land directly in the layer's
# `target_units` (NOT an intermediate `mapped_units`). Values verified against
# the live response metadata for Tuscany (43.46, 11.04).
#
# clay / sand / silt:  d_factor=10  → %
# phh2o / cec / soc:   d_factor=10  → pH / cmol(c)/kg / g/kg
# bdod:                d_factor=100 → kg/dm³
# cfvo:                d_factor=10  → cm³/100cm³ (vol %)
#
# Tuple is (final_divisor, target_unit_label) per property.
SOILGRIDS_UNIT_CONVERSIONS: dict[str, tuple[float, str]] = {
    "phh2o": (10.0,  "pH"),
    "clay":  (10.0,  "%"),
    "sand":  (10.0,  "%"),
    "silt":  (10.0,  "%"),
    "soc":   (10.0,  "g/kg"),
    "cec":   (10.0,  "cmol(c)/kg"),
    "bdod":  (100.0, "kg/dm³"),
    "cfvo":  (10.0,  "vol %"),
}

# Open-Elevation: free, no auth, intermittently down. One POST per region
# returns elevations for our 3x3 grid; slope is then derived in-process via
# Horn's algorithm. The grid spacing (~300 m at the equator) matches SRTM's
# 30 m source resolution closely enough that slope is meaningful but not noisy.
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
OPEN_ELEVATION_TIMEOUT_SEC = 30.0
OPEN_ELEVATION_RATE_LIMIT_SEC = 1.0
DEM_GRID_DEG = 0.0027
DEM_LAT_METERS_PER_DEG = 111_320.0

# Soil feature derivations.
CALCAREOUS_PH_THRESHOLD = 7.5
DRAINAGE_CLAY_PCT = 40.0
DRAINAGE_SAND_PCT = 60.0

SOIL_RAW_DIRNAME = "soil_raw"
DEM_RAW_DIRNAME = "dem"
SOIL_PARQUET = "soil.parquet"


def _project_root() -> Path:
    """Walk up from this file until we find the repo's `pyproject.toml`.

    Anchoring relative paths to the repo root (instead of CWD) is what lets
    notebooks, scripts, and tests resolve `data/raw/...` the same way no
    matter where they were launched from.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


class Settings(BaseSettings):
    """Environment-driven configuration with sensible local-dev defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VININATOR_",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    xwines_variant: Literal["test", "slim", "full"] = Field(default="test")
    nominatim_contact: str = Field(default=NOMINATIM_DEFAULT_CONTACT)

    @field_validator("data_dir", mode="after")
    @classmethod
    def _anchor_to_project_root(cls, value: Path) -> Path:
        """Relative `data_dir` resolves against the repo root, not CWD."""
        return value if value.is_absolute() else (_project_root() / value).resolve()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def xwines_wines_parquet(self) -> Path:
        """Per-wine canonical parquet (WineID + attrs)."""
        return self.raw_dir / XWINES_WINES_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def xwines_ratings_parquet(self) -> Path:
        """Per-rating canonical parquet (RatingID, WineID, Vintage, Rating, Date)."""
        return self.raw_dir / XWINES_RATINGS_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def xwines_wines_csv(self) -> Path:
        """Source CSV the variant expects to find under `data/raw/`."""
        return self.raw_dir / XWINES_VARIANTS[self.xwines_variant]["wines_csv"]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def xwines_ratings_csv(self) -> Path:
        return self.raw_dir / XWINES_VARIANTS[self.xwines_variant]["ratings_csv"]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def geocode_parquet(self) -> Path:
        """Cache of `RegionName` → `(lat, lon)` results from Nominatim."""
        return self.interim_dir / GEOCODE_PARQUET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def nominatim_user_agent(self) -> str:
        """User agent sent to Nominatim, built from the env-driven contact."""
        return NOMINATIM_USER_AGENT_TEMPLATE.format(contact=self.nominatim_contact)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def soil_raw_dir(self) -> Path:
        """One JSON per region: the raw SoilGrids 3x3 buffer response."""
        return self.interim_dir / SOIL_RAW_DIRNAME

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dem_raw_dir(self) -> Path:
        """One JSON per region: the raw Open-Elevation 3x3 grid response."""
        return self.interim_dir / DEM_RAW_DIRNAME

    @computed_field  # type: ignore[prop-decorator]
    @property
    def soil_parquet(self) -> Path:
        """Aggregated per-region soil + terrain features."""
        return self.interim_dir / SOIL_PARQUET

    def ensure_dirs(self) -> None:
        """Create the data layout if missing. Idempotent."""
        for d in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.soil_raw_dir,
            self.dem_raw_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
