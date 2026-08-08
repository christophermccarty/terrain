"""Low-order force-restore and Penman--Monteith land surface closure."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class LandSurfaceStep(NamedTuple):
    temperature: np.ndarray
    deep_temperature: np.ndarray
    latent_heat_w_m2: np.ndarray
    sensible_heat_w_m2: np.ndarray


# Dry-to-saturated volumetric soil heat-capacity ratio.  Standard boundary-layer
# climatology tabulations (e.g. Oke, *Boundary Layer Climates*, 2nd ed., Table
# 2.1) give dry mineral soil ~1.3 MJ m-3 K-1 and water-saturated soil ~3.0
# MJ m-3 K-1 -- water's own volumetric heat capacity (~4.18 MJ m-3 K-1) mixed
# into typical ~40% porosity.  That is a ~2.3x range from dry to saturated, not
# the fixed scalar this module previously used regardless of wetness.
_SOIL_HEAT_CAPACITY_WET_DRY_RATIO = 2.3

# Wetness at which the scaling below reproduces the caller's heat-capacity
# argument exactly, chosen to match `simulate.py`'s own default land-moisture
# fallback (`soil_for_land = 0.55` when no soil-moisture field exists yet) so
# a run without soil-moisture history is unaffected, and so the existing
# `land_surface_heat_capacity_j_m2_k` / `land_deep_heat_capacity_j_m2_k`
# defaults keep their previously calibrated meaning at that reference wetness
# rather than silently drifting when this function is first switched on.
_SOIL_HEAT_CAPACITY_REFERENCE_WETNESS = 0.55

# Root-zone weighting of the fast/shallow versus slow/deep soil-moisture
# buckets for the Penman--Monteith dry-surface resistance term. Jackson et al.
# (1996, *Oecologia*, "A global analysis of root distributions for terrestrial
# biomes") finds most fine-root biomass across biomes concentrated in roughly
# the top 30 cm, with a real but minority fraction extending into deeper soil
# that is tapped once the shallow layer is depleted. This model carries only
# two lumped buckets (no per-biome root profile), so a fixed 70/30 shallow/deep
# split is the simplest defensible stand-in for that general pattern, not a
# fit to any single biome's coefficient.
_ROOT_ZONE_SHALLOW_WEIGHT = 0.7


def _moisture_scaled_heat_capacity(
    base_j_m2_k: float, wetness: np.ndarray
) -> np.ndarray:
    """Scale a base reservoir heat capacity by soil wetness.

    Linear (rule-of-mixtures) interpolation between a dry and a saturated
    multiplier of `base_j_m2_k`, calibrated so that at
    `_SOIL_HEAT_CAPACITY_REFERENCE_WETNESS` the result equals `base_j_m2_k`
    exactly -- see that constant's docstring.  Wet soil then acquires more
    thermal inertia than dry soil, and dry soil less, spanning the
    `_SOIL_HEAT_CAPACITY_WET_DRY_RATIO` literature range end to end.
    """
    denom = 1.0 + _SOIL_HEAT_CAPACITY_REFERENCE_WETNESS * (
        _SOIL_HEAT_CAPACITY_WET_DRY_RATIO - 1.0
    )
    dry_multiplier = 1.0 / denom
    wet_multiplier = _SOIL_HEAT_CAPACITY_WET_DRY_RATIO * dry_multiplier
    multiplier = dry_multiplier + np.clip(wetness, 0.0, 1.0) * (
        wet_multiplier - dry_multiplier
    )
    return base_j_m2_k * multiplier


def force_restore_penman_monteith(
    surface_temperature_k: np.ndarray,
    air_temperature_k: np.ndarray,
    deep_temperature_k: np.ndarray | None,
    soil_moisture: np.ndarray,
    wind_speed_m_s: np.ndarray,
    net_radiation_w_m2: np.ndarray,
    land_mask: np.ndarray,
    *,
    dt_days: float,
    surface_heat_capacity_j_m2_k: float,
    deep_heat_capacity_j_m2_k: float,
    restore_days: float,
    surface_resistance_min_s_m: float,
    surface_resistance_dry_s_m: float,
    soil_moisture_deep: np.ndarray | None = None,
) -> LandSurfaceStep:
    """Advance a two-reservoir land surface one explicit, bounded timestep.

    The available energy is partitioned by the FAO-56 Penman--Monteith form.
    Soil moisture sets surface resistance, so wet land preferentially cools
    through latent heat while dry land partitions more energy into sensible
    heat.  The surface-to-deep exchange is the restore term.

    `soil_moisture` is the fast/shallow bucket and `soil_moisture_deep` the
    slow/root-zone bucket (see `atmosphere.generate_precipitation`'s docstring
    for the same two-bucket split).  Both now feed real physical terms rather
    than only the shallow bucket feeding resistance and neither feeding
    thermal inertia: dry-surface resistance uses a root-zone-weighted blend of
    the two (`_ROOT_ZONE_SHALLOW_WEIGHT`), and each reservoir's own heat
    capacity is scaled by its own wetness (`_moisture_scaled_heat_capacity`).
    `soil_moisture_deep=None` falls back to the shallow bucket for both uses,
    exactly reproducing this function's previous behaviour.
    """
    if dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    if min(surface_heat_capacity_j_m2_k, deep_heat_capacity_j_m2_k,
           restore_days, surface_resistance_min_s_m, surface_resistance_dry_s_m) <= 0.0:
        raise ValueError("force-restore parameters must be positive")
    ts = np.asarray(surface_temperature_k, dtype=np.float64)
    ta = np.asarray(air_temperature_k, dtype=np.float64)
    soil = np.clip(np.asarray(soil_moisture, dtype=np.float64), 0.0, 1.0)
    wind = np.maximum(np.asarray(wind_speed_m_s, dtype=np.float64), 0.2)
    rn = np.asarray(net_radiation_w_m2, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    if not (ts.shape == ta.shape == soil.shape == wind.shape == rn.shape == land.shape):
        raise ValueError("land surface inputs must share a shape")
    if soil_moisture_deep is None:
        soil_deep = soil
    else:
        soil_deep = np.clip(np.asarray(soil_moisture_deep, dtype=np.float64), 0.0, 1.0)
        if soil_deep.shape != ts.shape:
            raise ValueError("soil_moisture_deep must match surface temperature")
    # A newly enabled prognostic reservoir has no history.  Initializing it to
    # the surface avoids manufacturing a restore flux merely because the old
    # diagnostic atmosphere and the new land path disagree on their first
    # timestep; it then acquires physical memory through the restore equation.
    td = ts.copy() if deep_temperature_k is None else np.asarray(deep_temperature_k, dtype=np.float64).copy()
    if td.shape != ts.shape:
        raise ValueError("deep_temperature_k must match surface temperature")

    # Saturation-vapour slope (Pa K-1) and a standard psychrometric constant.
    tc = np.clip(ts - 273.15, -70.0, 70.0)
    es_pa = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    delta = 4098.0 * es_pa / (tc + 243.5) ** 2
    gamma = 66.0
    rho_air, cp_air = 1.2, 1004.0
    ra = 1.0 / (1.3e-3 * wind)
    root_zone_wetness = (
        _ROOT_ZONE_SHALLOW_WEIGHT * soil + (1.0 - _ROOT_ZONE_SHALLOW_WEIGHT) * soil_deep
    )
    rs = (
        surface_resistance_min_s_m
        + (1.0 - root_zone_wetness) ** 2 * surface_resistance_dry_s_m
    )
    # A humidity proxy is intentionally not supplied here; the resolved
    # air--surface temperature difference provides a stable vapour-deficit
    # closure while the main moisture model remains authoritative for water.
    vpd_pa = np.maximum(0.0, 120.0 * (ts - ta))
    latent = (delta * rn + rho_air * cp_air * vpd_pa / ra) / (
        delta + gamma * (1.0 + rs / ra)
    )
    latent = np.clip(latent, 0.0, np.maximum(rn + 350.0, 0.0))
    sensible = rho_air * cp_air * (ts - ta) / ra
    # Each reservoir's own heat capacity now varies with its own wetness
    # (`_moisture_scaled_heat_capacity`) instead of using the caller's scalar
    # everywhere: wet soil carries more thermal inertia than dry soil, per
    # `_SOIL_HEAT_CAPACITY_WET_DRY_RATIO`. `surface_capacity` also sets the
    # restore flux's conductance term, since force-restore theory ties the
    # surface-to-deep exchange to the same shallow-reservoir thermal mass.
    surface_capacity = _moisture_scaled_heat_capacity(surface_heat_capacity_j_m2_k, soil)
    deep_capacity = _moisture_scaled_heat_capacity(deep_heat_capacity_j_m2_k, soil_deep)
    restore_flux = surface_capacity * (ts - td) / (restore_days * 86400.0)
    tendency = (rn - latent - sensible - restore_flux) * dt_days * 86400.0 / surface_capacity
    ts_next = ts + np.clip(tendency, -5.0, 5.0)
    td_next = td + restore_flux * dt_days * 86400.0 / deep_capacity
    ts_next = np.where(land, ts_next, ts)
    td_next = np.where(land, td_next, td)
    return LandSurfaceStep(
        ts_next.astype(np.float32), td_next.astype(np.float32),
        latent.astype(np.float32), sensible.astype(np.float32),
    )
