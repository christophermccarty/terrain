"""Reusable regional climate validation primitives.

Named geographic boxes were historically duplicated across manual scripts and
overnight notebooks.  Keeping their definitions and target ranges here lets
tests, diagnostics, and the GUI evaluate the same regions consistently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClimateRegion:
    name: str
    lat_n: float
    lat_s: float
    lon_w: float
    lon_e: float
    group: str
    precip_min_mm_year: float
    precip_max_mm_year: float


EARTH_PRECIP_REGIONS: tuple[ClimateRegion, ...] = (
    ClimateRegion("Sahara", 30.0, 15.0, -10.0, 30.0, "desert", 0.0, 200.0),
    ClimateRegion("Kalahari", -20.0, -28.0, 15.0, 25.0, "desert", 0.0, 200.0),
    ClimateRegion("Atacama", -20.0, -28.0, -71.0, -68.0, "desert", 0.0, 50.0),
    ClimateRegion(
        "Canadian Prairies", 55.0, 50.0, -110.0, -100.0,
        "continental", 400.0, 500.0,
    ),
    ClimateRegion(
        "US Midwest", 45.0, 38.0, -100.0, -90.0,
        "continental", 800.0, 1000.0,
    ),
    ClimateRegion(
        "Central Europe", 53.0, 47.0, 5.0, 20.0,
        "continental", 550.0, 750.0,
    ),
    # Monsoon/eastern-margin humid subtropics: sit inside the same
    # DRYBELT_CENTER_DEG~28 latitude window as Sahara/Kalahari/Atacama
    # above, but escape the subtropical high in reality via monsoon/warm-
    # current moisture -- see se-us-east-asia-drybelt-latitude-bug memory.
    # Real Koppen for all three is Cfa (humid subtropical); target ranges
    # are approximate regional climatology, not precise station data.
    ClimateRegion(
        "SE US", 33.0, 27.0, -90.0, -80.0,
        "monsoon_subtropical", 1100.0, 1500.0,
    ),
    ClimateRegion(
        "East China", 30.0, 23.0, 110.0, 120.0,
        "monsoon_subtropical", 1300.0, 1800.0,
    ),
    ClimateRegion(
        "S Japan", 34.0, 31.0, 130.0, 134.0,
        "monsoon_subtropical", 1600.0, 2200.0,
    ),
)


def latitude_longitude_centers(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column center coordinates in degrees."""
    lat = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * 180.0
    lon = ((np.arange(W, dtype=np.float64) + 0.5) / W) * 360.0 - 180.0
    return lat, lon


def region_mask(
    shape: tuple[int, int],
    region: ClimateRegion,
    *,
    cell_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return cells whose centers fall inside ``region``.

    Longitude boxes that cross the dateline are supported by allowing
    ``lon_w > lon_e``.  ``cell_mask`` can restrict the result to land/ocean.
    """
    H, W = shape
    lat, lon = latitude_longitude_centers(H, W)
    lat_sel = (lat <= region.lat_n) & (lat >= region.lat_s)
    if region.lon_w <= region.lon_e:
        lon_sel = (lon >= region.lon_w) & (lon <= region.lon_e)
    else:
        lon_sel = (lon >= region.lon_w) | (lon <= region.lon_e)
    selected = lat_sel[:, None] & lon_sel[None, :]
    if cell_mask is not None:
        if cell_mask.shape != shape:
            raise ValueError(
                f"cell_mask shape {cell_mask.shape} does not match field shape {shape}"
            )
        selected &= np.asarray(cell_mask, dtype=bool)
    return selected


def region_mean(
    field: np.ndarray,
    region: ClimateRegion,
    *,
    cell_mask: np.ndarray | None = None,
) -> float | None:
    """Return a regional cell mean, or ``None`` when no selected cells exist."""
    values = np.asarray(field)
    selected = region_mask(values.shape, region, cell_mask=cell_mask)
    if not np.any(selected):
        return None
    return float(np.mean(values[selected], dtype=np.float64))


def precipitation_by_region_mm_year(
    precipitation_mm_day: np.ndarray,
    *,
    land_mask: np.ndarray,
    days_per_year: float,
    regions: tuple[ClimateRegion, ...] = EARTH_PRECIP_REGIONS,
) -> dict[str, float | None]:
    """Evaluate daily precipitation over named land regions in mm/year."""
    if days_per_year <= 0.0:
        raise ValueError("days_per_year must be positive")
    return {
        region.name: (
            None
            if (daily := region_mean(
                precipitation_mm_day, region, cell_mask=land_mask
            )) is None
            else daily * float(days_per_year)
        )
        for region in regions
    }


def target_error_fraction(value: float, region: ClimateRegion) -> float:
    """Distance outside a target interval, normalized by its nonzero midpoint."""
    midpoint = 0.5 * (region.precip_min_mm_year + region.precip_max_mm_year)
    scale = max(midpoint, region.precip_max_mm_year, 1.0)
    if value < region.precip_min_mm_year:
        return (region.precip_min_mm_year - value) / scale
    if value > region.precip_max_mm_year:
        return (value - region.precip_max_mm_year) / scale
    return 0.0
