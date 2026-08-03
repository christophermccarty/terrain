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


@dataclass(frozen=True)
class OrographicPair:
    """A windward/leeward box pair straddling one mountain range.

    Added 2026-08-02 for ACCURACY_AUDIT.md A5's orographic calibration, and
    specifically to answer the audit's own process note 11: *none* of the nine
    ``EARTH_PRECIP_REGIONS`` boxes is mountainous, so an orographic change reads
    as "no effect" against them by construction. A mechanism can only be judged
    by an instrument that resolves it.

    ``ratio_min``/``ratio_max`` bracket the real-world windward:leeward annual
    precipitation ratio for these *boxes* (not station extremes, which are much
    larger -- a box average over a few degrees of longitude necessarily includes
    slope and foothill cells that dilute the crestline contrast). Approximate
    regional climatology, same standing as the monsoon boxes above.

    Box placement was verified against the bundled DEM at 512x1024: each
    ``windward``/``leeward`` pair sits on opposite flanks of an actual resolved
    crest. These pairs need >= ~256x512 to be meaningful; at the tracked 64x128
    fixture a 2.8 deg cell spans the whole range and the two boxes collapse
    onto the same cell.
    """

    name: str
    windward: ClimateRegion
    leeward: ClimateRegion
    ratio_min: float
    ratio_max: float
    note: str = ""


def _orographic_box(name: str, lat_n: float, lat_s: float,
                    lon_w: float, lon_e: float) -> ClimateRegion:
    # Precip bounds are unused for these boxes -- contrast is a ratio between
    # the two flanks, not a per-box target -- so they are left wide open.
    return ClimateRegion(name, lat_n, lat_s, lon_w, lon_e, "orographic", 0.0, 1.0e9)


OROGRAPHIC_PAIRS: tuple[OrographicPair, ...] = (
    OrographicPair(
        "Cascades",
        _orographic_box("Cascades windward", 48.5, 44.5, -124.0, -122.3),
        _orographic_box("Cascades leeward", 48.5, 44.5, -120.5, -118.5),
        3.0, 6.0,
        "mid-latitude westerlies; crest resolved at -121.5",
    ),
    OrographicPair(
        "Sierra Nevada",
        _orographic_box("Sierra windward", 40.0, 36.0, -121.5, -120.0),
        _orographic_box("Sierra leeward", 40.0, 36.0, -118.3, -116.8),
        2.5, 5.0,
        "westerlies; crest resolved at -119.5, Owens Valley rain shadow",
    ),
    OrographicPair(
        "S Andes",
        _orographic_box("S Andes windward", -41.0, -48.0, -74.5, -73.0),
        _orographic_box("S Andes leeward", -41.0, -48.0, -70.5, -69.0),
        5.0, 15.0,
        "strongest westerlies on Earth; Patagonian steppe in the lee",
    ),
    OrographicPair(
        "Southern Alps",
        _orographic_box("S Alps windward", -42.5, -45.5, 169.8, 170.9),
        _orographic_box("S Alps leeward", -42.5, -45.5, 171.4, 172.6),
        4.0, 12.0,
        "narrow range: only ~2-3 cells per box even at 512x1024",
    ),
    OrographicPair(
        "Scandinavia",
        _orographic_box("Scandes windward", 63.0, 59.5, 5.0, 6.8),
        _orographic_box("Scandes leeward", 63.0, 59.5, 9.0, 11.5),
        2.0, 4.0,
        "westerlies; Bergen vs the Swedish interior",
    ),
    OrographicPair(
        "Himalaya",
        _orographic_box("Himalaya windward", 28.0, 25.5, 84.0, 92.0),
        _orographic_box("Himalaya leeward", 33.0, 30.5, 84.0, 92.0),
        5.0, 20.0,
        "meridional pair: summer monsoon from the south, Tibet in the lee",
    ),
)


def orographic_contrast(
    precipitation_mm_day: np.ndarray,
    pair: OrographicPair,
    *,
    land_mask: np.ndarray,
) -> dict[str, float] | None:
    """Windward:leeward mean ratio for one range, or ``None`` if unresolved."""
    windward = region_mean(precipitation_mm_day, pair.windward, cell_mask=land_mask)
    leeward = region_mean(precipitation_mm_day, pair.leeward, cell_mask=land_mask)
    if windward is None or leeward is None:
        return None
    return {
        "windward": float(windward),
        "leeward": float(leeward),
        "ratio": float(windward / leeward) if leeward > 1e-9 else float("inf"),
    }


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
