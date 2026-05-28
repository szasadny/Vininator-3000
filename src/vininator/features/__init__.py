"""Feature-engineering layer.

Pure functions over polars frames + cached fetchers for external data
sources (SoilGrids, NASA POWER, DEM). Imports from `data/` but never the
other way around; the API layer imports from here at inference time so
batch and online code paths share the same feature definitions.
"""

from __future__ import annotations
