"""Time simulation for planet conditions.

Advances atmospheric systems (temperature, wind, precipitation) forward in time
with configurable time scales. Default unit is one day.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import NamedTuple
from pathlib import Path
from datetime import datetime
from enum import Enum

LOG = logging.getLogger("planetsim")


class TimeScaleMode(Enum):
    """Time integration strategy for different simulation speeds.

    DAILY   — 1 day per UI frame, full physics (highest accuracy).
    WEEKLY  — 7 days per UI frame, 7 × 1-day sub-steps (same physics).
    MONTHLY — ~30 days per UI frame, 5 × 6-day sub-steps, no wind evolution
              (faster; weather variability is parameterized away).
    ANNUAL  — ~365 days per UI frame, 52 × 7-day sub-steps, no wind or storms
              (fastest; only slow climate variables evolve accurately).
    """
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ANNUAL = "annual"
import pickle
from atmosphere import (
    generate_wind_field, generate_precipitation,
    evolve_wind, evolve_wind_aloft, _upsample_bilinear_many,
    _update_jet_index, _update_jet_blocking,
)
from temperature import temperature_kelvin_for_lat, elevation_to_alt_km
from ocean import calculate_ocean_heat_transport, update_sea_ice, compute_ekman_transport
from carbon_cycle import (
    carbon_cycle_step, co2_temperature_response, CO2_PREINDUSTRIAL,
    co2_radiative_forcing, vegetation_albedo,
)
from climate_averages import (
    update_climate_averages, compute_stable_biomes,
    update_monthly_statistics, classify_koppen, koppen_to_legacy_biome,
)
from masks import get_masks
from planet_params import PlanetParams, EARTH

# Numba JIT compilation for performance
try:
    from numba import jit, prange  # pyright: ignore[reportMissingImports]
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback: create dummy decorators if Numba not installed
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    NUMBA_AVAILABLE = False

# Cache for diagnostic/relaxation wind to avoid recomputing every step.
_RELAX_CACHE = {"key": None, "u": None, "v": None}


def _coarsen(arr: np.ndarray, Hc: int, Wc: int, bs: int) -> np.ndarray:
    """Downsample (H,W) → (Hc,Wc) by block-averaging.

    Avoids np.pad when the grid divides evenly (the common case for standard
    block sizes), which eliminates a full-array copy per field per step.
    """
    a: np.ndarray = arr if arr.dtype == np.float32 else arr.astype(np.float32)
    H, W = a.shape
    ph, pw = Hc * bs - H, Wc * bs - W
    if ph > 0 or pw > 0:
        a = np.pad(a, ((0, ph), (0, pw)), mode="edge")
    return a.reshape(Hc, bs, Wc, bs).mean(axis=(1, 3)).astype(np.float32, copy=False)


def _coarsen_many(fields: dict[str, np.ndarray], Hc: int, Wc: int, bs: int) -> dict[str, np.ndarray]:
    """Batched `_coarsen`: downsample multiple same-shape (H,W) fields in one pass.

    Stacks inputs into one (K,H,W) array and does a single pad+reshape+mean instead
    of K separate `_coarsen` calls — mirrors `_upsample_bilinear_many`'s batching
    approach for the opposite (upsample) direction. Every field must share the same
    (H, W) shape; callers with mixed shapes should group fields accordingly first.
    """
    if not fields:
        return {}
    keys = list(fields.keys())
    stack = np.stack(
        [f if f.dtype == np.float32 else f.astype(np.float32) for f in fields.values()],
        axis=0,
    )
    K, H, W = stack.shape
    ph, pw = Hc * bs - H, Wc * bs - W
    if ph > 0 or pw > 0:
        stack = np.pad(stack, ((0, 0), (0, ph), (0, pw)), mode="edge")
    out = stack.reshape(K, Hc, bs, Wc, bs).mean(axis=(2, 4)).astype(np.float32, copy=False)
    return {k: out[i] for i, k in enumerate(keys)}


# A single generate_precipitation() call can only "rain out" the moisture
# reservoir once, regardless of how many days `dt_days` spans. At DAILY dt=1
# that's fine (365 independent calls/year, each free to evaporate-then-rain).
# At MONTHLY dt~30 a single call applies one snapshot of wind/humidity as if
# constant for the whole month and rains it out once, so precipitation doesn't
# scale up the way a real month's worth of independent weather events would --
# while the soil-moisture drain term (evaporation * dt) scales linearly with
# dt with no such ceiling. That mismatch was driving continental-interior soil
# moisture to its floor within a few decades of MONTHLY-mode spinup (observed:
# Canadian-Prairies-latitude precip collapsing to ~12 mm/yr vs Earth's
# 350-450 mm/yr), even though the underlying replenish/drain calibration is
# sound at dt=1. Sub-stepping in ~1-week chunks lets humidity evaporate and
# rain out multiple times per outer call, closing the gap without touching
# per-call physics or DAILY/WEEKLY-mode behavior (n_sub=1 there, so this is a
# no-op below the threshold).
_PRECIP_SUBSTEP_DAYS = 8.0


def _generate_precipitation_substepped(H, W, elev, *, temperature, wind_u, wind_v,
                                        humidity, soil_moisture, soil_moisture_deep,
                                        cloud_fraction,
                                        day_of_year, dt_days,
                                        surface_pressure_hpa=1013.25,
                                        planet_params=None):
    dt_days = float(dt_days)
    if dt_days <= _PRECIP_SUBSTEP_DAYS:
        return generate_precipitation(
            H, W, elev, temperature=temperature, wind_u=wind_u, wind_v=wind_v,
            humidity=humidity, soil_moisture=soil_moisture,
            soil_moisture_deep=soil_moisture_deep,
            cloud_fraction=cloud_fraction, day_of_year=day_of_year, dt_days=dt_days,
            surface_pressure_hpa=surface_pressure_hpa, planet_params=planet_params,
        )
    n_sub = max(1, int(round(dt_days / _PRECIP_SUBSTEP_DAYS)))
    sub_dt = dt_days / n_sub
    hum, soil, soil_deep = humidity, soil_moisture, soil_moisture_deep
    P_accum = None
    for _ in range(n_sub):
        P_i, hum, soil, soil_deep = generate_precipitation(
            H, W, elev, temperature=temperature, wind_u=wind_u, wind_v=wind_v,
            humidity=hum, soil_moisture=soil, soil_moisture_deep=soil_deep,
            cloud_fraction=cloud_fraction,
            day_of_year=day_of_year, dt_days=sub_dt,
            surface_pressure_hpa=surface_pressure_hpa, planet_params=planet_params,
        )
        P_accum = P_i.astype(np.float32) if P_accum is None else P_accum + P_i
    return (P_accum / n_sub).astype(np.float32, copy=False), hum, soil, soil_deep


# Cache for coarsened elevation. Elevation is static terrain — the same array
# object is threaded unchanged through every simulate_step call for the life
# of a run (see `new_state = PlanetState(..., elevation=state.elevation, ...)`
# at the end of simulate_step) — but it was being re-coarsened from scratch up
# to 3x per step (temp/precip block_size, wind_block_size, precip block_size),
# a measurable share of profiled per-step cost at production resolution.
# Mirrors masks.py's id()+content-fingerprint cache pattern (same risk: id()
# can be reused after garbage collection, so a cheap fingerprint guards against
# stale hits) rather than inventing a new scheme.
_ELEV_COARSEN_CACHE: dict[tuple[int, int, int, int], np.ndarray] = {}
_ELEV_COARSEN_CACHE_FP: dict[tuple[int, int, int, int], tuple[float, float, float]] = {}


def _coarsen_elevation_cached(elevation: np.ndarray, Hc: int, Wc: int, bs: int) -> np.ndarray:
    """Like `_coarsen`, but cached for the (elevation, Hc, Wc, bs) combination.

    Only safe for arrays whose Python object identity is stable across calls
    (i.e. `state.elevation`) — never use this for per-step-mutated fields.
    """
    key = (id(elevation), Hc, Wc, bs)
    elev_r = np.asarray(elevation, dtype=np.float32).ravel()
    n = elev_r.size
    fp = (float(elev_r[0]), float(elev_r[-1]), float(elev_r.sum())) if n >= 2 else (0.0, 0.0, 0.0)
    cached = _ELEV_COARSEN_CACHE.get(key)
    if cached is not None and cached.shape == (Hc, Wc) and _ELEV_COARSEN_CACHE_FP.get(key) == fp:
        return cached
    result = _coarsen(elevation, Hc, Wc, bs)
    result.flags.writeable = False  # catch accidental in-place mutation of the cached array
    _ELEV_COARSEN_CACHE[key] = result
    _ELEV_COARSEN_CACHE_FP[key] = fp
    return result

# Cache for ocean heat transport adjustment.
# Ocean dynamics are slow (decorrelation time ~30 days), so we recompute
# only once per ocean_update_interval_days and reuse the cached ΔT array.
_OCEAN_ADJ_CACHE: dict = {"adj": None, "last_update_day": -9999.0}

# Cache for the slow-changing half of the carbon cycle: wildfire, permafrost
# thaw, wetland CH4 emission, and the biome classification that feeds
# vegetation NPP. These are all genuinely slow processes (fire risk, thaw, and
# biome shifts don't meaningfully change day to day) that were nonetheless
# being recomputed as full-resolution array passes every single step —
# ~22% of profiled per-step cost at production resolution (512x1024). Applied
# in every TimeScaleMode, including DAILY, unlike the Phase 1 "DAILY = full
# per-day physics" convention used elsewhere (see PLAN.md) — this is the one
# deliberate exception, because these four processes don't actually have
# meaningful per-day dynamics to resolve even in DAILY mode.
# Mirrors _OCEAN_ADJ_CACHE's pattern: recompute+apply a lump update every
# CARBON_SLOW_UPDATE_INTERVAL_DAYS (with dt_days = the accumulated interval,
# not the per-step days), hold state constant in between. Ocean CO2 exchange
# and vegetation NPP/growth are NOT included here — they stay fully per-step
# via carbon_cycle_step (fast-responding, and cheap relative to the four
# processes above).
CARBON_SLOW_UPDATE_INTERVAL_DAYS = 4.0
_CARBON_SLOW_CACHE: dict = {"key": None, "last_update_day": -9999.0, "biome": None}


# ============================================================================
# Numba-accelerated compute kernels for temperature evolution
# These provide 5-20x speedup for advection and diffusion operations
# ============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _advect_temperature_x_numba(T: np.ndarray, u: np.ndarray,
                                u_cfl: np.ndarray) -> np.ndarray:
    """Advect temperature in x-direction (periodic boundaries).

    u_cfl is the pre-computed CFL number |u|*dt/dx, clipped to [0, 0.5].
    Sign of u selects upwind direction.
    Returns updated temperature field.
    """
    H, W = T.shape
    T_out = T.copy()

    for i in prange(H):
        for j in range(W):
            # Periodic wrap in x
            j_east = (j + 1) % W
            j_west = (j - 1 + W) % W

            # Upwind advection: u>0 means westward flow in grid coords
            if u[i, j] >= 0:
                T_x = T[i, j_west]
            else:
                T_x = T[i, j_east]

            # Temperature difference with manual clipping (Numba compatible)
            T_diff = T_x - T[i, j]
            if T_diff > 12.0:
                T_diff = 12.0
            elif T_diff < -12.0:
                T_diff = -12.0

            # Apply CFL-correct advection
            T_out[i, j] = T[i, j] + u_cfl[i, j] * T_diff

    return T_out


@jit(nopython=True, parallel=True, cache=True)
def _advect_temperature_y_numba(T: np.ndarray, v: np.ndarray,
                                v_cfl: np.ndarray) -> np.ndarray:
    """Advect temperature in y-direction (edge boundaries).

    v_cfl is the pre-computed CFL number |v|*dt/dy, clipped to [0, 0.5].
    Sign of v selects upwind direction.
    Returns updated temperature field.
    """
    H, W = T.shape
    T_out = T.copy()

    for i in prange(1, H-1):  # Skip poles (edges)
        for j in range(W):
            # Upwind advection
            if v[i, j] >= 0:
                T_y = T[i + 1, j]  # Southward (positive v)
            else:
                T_y = T[i - 1, j]  # Northward (negative v)

            # Temperature difference with manual clipping (Numba compatible)
            T_diff = T_y - T[i, j]
            if T_diff > 12.0:
                T_diff = 12.0
            elif T_diff < -12.0:
                T_diff = -12.0

            # Apply CFL-correct advection
            T_out[i, j] = T[i, j] + v_cfl[i, j] * T_diff

    return T_out


@jit(nopython=True, parallel=True, cache=True)
def _apply_diffusion_numba(T: np.ndarray, thermal_diff: float, days: float,
                          iterations: int = 3) -> np.ndarray:
    """Apply Laplacian diffusion to temperature field.

    Returns updated temperature field after specified iterations.
    """
    H, W = T.shape
    T_curr = T.copy()

    for _ in range(iterations):
        T_new = T_curr.copy()

        for i in prange(1, H-1):  # Skip edges
            for j in range(W):
                # Periodic in x
                j_east = (j + 1) % W
                j_west = (j - 1 + W) % W

                # Laplacian (5-point stencil)
                c = T_curr[i, j]
                n = T_curr[i - 1, j]
                s = T_curr[i + 1, j]
                e = T_curr[i, j_east]
                w = T_curr[i, j_west]

                T_lap = n + s + e + w - 4.0 * c

                # Clamp laplacian to prevent extreme smoothing (manual clip for Numba)
                if T_lap > 30.0:
                    T_lap = 30.0
                elif T_lap < -30.0:
                    T_lap = -30.0

                # Apply diffusion
                T_new[i, j] = c + thermal_diff * 1.2 * T_lap * days

        T_curr = T_new

    return T_curr


class PlanetState(NamedTuple):
    """Current planet state snapshot."""
    day_of_year: float  # Fractional day (0-365.2422)
    elevation: np.ndarray  # (H, W) terrain elevation [0,1]
    total_days: float = 0.0  # Unwrapped simulation time (days since start)
    temperature: np.ndarray | None = None  # (H, W) surface temperature (K)
    air_temperature: np.ndarray | None = None  # (H, W) troposphere temperature (K)
    wind_u: np.ndarray | None = None  # (H, W) eastward wind (m/s)
    wind_v: np.ndarray | None = None  # (H, W) northward wind (m/s)
    precipitation: np.ndarray | None = None  # (H, W) precipitation (mm/day)
    humidity: np.ndarray | None = None  # (H, W) specific humidity [kg/kg]
    soil_moisture: np.ndarray | None = None  # (H, W) bucket soil moisture [0,1]
    cloud_cover: np.ndarray | None = None  # (H, W) cloud fraction [0,1]
    cloud_water: np.ndarray | None = None  # (H, W) cloud liquid water [kg/kg]
    snow_depth: np.ndarray | None = None  # (H, W) snow depth [m]
    ice_cover: np.ndarray | None = None  # (H, W) sea ice fraction [0,1]
    # Carbon cycle fields (Phase 3)
    co2_atmosphere: float = 400.0  # Atmospheric CO2 concentration [ppm] (global mean)
    co2_ocean: np.ndarray | None = None  # (H, W) dissolved CO2 in ocean [ppm equivalent]
    vegetation_biomass: np.ndarray | None = None  # (H, W) carbon in vegetation [kg C/m²]
    # Climate averaging fields (Phase 1 - Biome Stability)
    climate_temp_avg: np.ndarray | None = None  # (H, W) 10-year rolling average temperature [K]
    climate_precip_avg: np.ndarray | None = None  # (H, W) 10-year rolling average precip [mm/day]
    climate_sample_days: float = 0.0  # Days accumulated in climate average
    biome_type: np.ndarray | None = None  # (H, W) stable biome classification (0-4: ocean, desert, grass, forest, tundra)
    biome_last_update_day: float = 0.0  # Total days when biomes were last reclassified
    # Monthly climate statistics for Köppen classification
    monthly_temp: np.ndarray | None = None  # (12, H, W) monthly mean temperature [K]
    monthly_precip: np.ndarray | None = None  # (12, H, W) monthly mean precipitation [mm/day]
    monthly_sample_count: np.ndarray | None = None  # (12,) sample count per month
    koppen_type: np.ndarray | None = None  # (H, W) Köppen climate classification (0-19)
    ice_sheet_age: np.ndarray | None = None  # (H, W) days each land cell has continuously met EF criteria
    # Feature 1: cloud radiative feedback
    # (cloud_cover already exists above — used as prev_cloud_cover each step)
    # Feature 3: salinity / AMOC freshwater
    salinity: np.ndarray | None = None  # (H, W) practical salinity [PSU]; ocean only
    # Feature 4: CH4 / permafrost carbon
    ch4_atmosphere: float = 1900.0  # global mean atmospheric CH4 [ppb]
    permafrost_carbon: np.ndarray | None = None  # (H, W) frozen soil carbon [kgC/m²]
    # Feature 5: deep ocean 2-layer
    T_deep_ocean: np.ndarray | None = None  # (H, W) abyssal ocean temperature [K]
    # Feature 6: sea ice thickness
    ice_thickness: np.ndarray | None = None  # (H, W) sea ice thickness [m]; 0 on land/open ocean
    # Feature 7: jet stream dynamics (persistent meander index + blocking events)
    jet_index_nh: float = 0.0   # NH persistent meander/waviness index, roughly [-2, 2]
    jet_index_sh: float = 0.0   # SH persistent meander/waviness index
    jet_block_lon_nh: float = -1.0        # active NH blocking ridge longitude [deg]; -1 = inactive
    jet_block_days_left_nh: float = 0.0   # days remaining in the active NH block
    jet_block_total_days_nh: float = 0.0  # total drawn duration of the active NH block (for ramp envelope)
    jet_block_lon_sh: float = -1.0
    jet_block_days_left_sh: float = 0.0
    jet_block_total_days_sh: float = 0.0
    # Feature 8: 1.5-layer atmosphere (real prognostic upper-level wind)
    wind_u_aloft: np.ndarray | None = None  # (H, W) upper-level eastward wind (m/s)
    wind_v_aloft: np.ndarray | None = None  # (H, W) upper-level northward wind (m/s)
    # Planet identity this save belongs to (None = EARTH, for saves predating this field)
    planet_params: PlanetParams | None = None
    # 2-layer soil-moisture bucket (Feature: soil desiccation-bistability fix, Jul 2026):
    # soil_moisture (above) is now the fast-draining surface layer; this is the
    # slow-draining deep/root-zone reservoir. See atmosphere.generate_precipitation.
    soil_moisture_deep: np.ndarray | None = None


def simulate_step(
    state: PlanetState,
    days: float = 1.0,
    *,
    block_size: int = 3,
    wind_block_size: int | None = None,
    update_wind: bool = True,
    # Small relaxation toward a diagnostic wind (includes tropical trades + mid-lat storm-track structure).
    # This helps recover easterly trades, westerly mid-lats, and calmer doldrums in a single-layer model.
    wind_relax: float = 0.0,
    wind_target_weather_amp: float = 0.35,
    wind_target_zonal_pressure: float = 0.85,
    wind_target_terrain_pressure_amp: float = 0.25,
    wind_target_terrain_flow_amp: float = 0.25,
    wind_pgf_temp_scale: float = 450.0,
    wind_pgf_terrain_scale: float = 900.0,
    wind_drag_base: float = 2.0e-7,
    wind_drag_elev_scale: float = 6.0e-7,
    wind_damping: float = 0.50,  # PGF scaling: 0.25 halved PGF causing 3-7× weak winds; 0.5 is better
    wind_vmax_clip: float = 50.0,  # Phase 4 fix: Realistic maximum wind speed (strong jet stream)
    # Baroclinic eddy / vertical momentum coupling to the upper-level (1.5-layer)
    # wind (evolve_wind_aloft). Dimensionless coupling-strength multiplier
    # (1.0 = nominal full-strength coupling on the wind_baroclinic_mix-day
    # relaxation timescale) -- NOTE: prior to the 1.5-layer atmosphere upgrade
    # this parameter scaled a magnitude-only `|dT/dy|` proxy directly (hence
    # the old ~1e6 default); its meaning changed with the real upper layer,
    # so the default was recalibrated rather than reused.
    wind_baroclinic_jet_amp: float = 1.0,
    wind_baroclinic_mix: float = 2.0,
    # Increased to 3.0 days to prevent oscillations from over-relaxation
    wind_cell_relax_days: float = 3.0,
    ocean_transport_coeff: float | None = None,  # None → pp.ocean_transport_coeff
    # Deprecated no-ops (kept for config compatibility): these were never read
    # by ocean.calculate_ocean_heat_transport.
    ocean_exchange_floor: float = 0.65,
    ocean_exchange_span: float = 0.35,
    # Ocean-atmosphere restoring rate [K/day]. Default matches the value that
    # was previously hardcoded inside ocean.py (0.03) — the old default here
    # (0.08) was silently ignored, so 0.03 preserves actual behavior now that
    # the parameter is wired through.
    ocean_exchange_coeff: float = 0.03,
    ocean_exchange_inertia: float = 0.35,
    epsilon_equator: float | None = None,
    epsilon_pole: float | None = None,
    polar_cooling_scale: float | None = None,  # None → pp.polar_cooling_scale
    ice_freeze_temp: float = 269.9,  # Require colder water before new sea ice forms
    ice_melt_temp: float = 271.4,    # Preserve some hysteresis without locking in subpolar ice
    ice_freeze_rate: float = 0.045,
    ice_melt_rate: float = 0.19,
    ice_albedo_strength: float | None = None,  # None → pp.ice_albedo_strength
    thermal_diffusion: float | None = None,    # None → pp.thermal_diffusivity
    latent_cooling_coeff: float = 0.015,  # Deprecated no-op (found by test_param_wiring.py, 2026-07-04): accepted but never read anywhere in the codebase; kept for config compatibility
    enable_carbon_cycle: bool = True,
    co2_climate_feedback: float | None = None,  # None → pp.co2_climate_feedback
    debug_log: bool = False,
    track_components: bool = False,
    planet_params: PlanetParams | None = None,
    time_scale: TimeScaleMode = TimeScaleMode.DAILY,
    feedback_flags: dict[str, bool] | None = None,
) -> tuple[PlanetState, dict]:
    # Supported flags (all default True when absent):
    #   'ice_albedo'        — sea ice effect on surface albedo + latent heat
    #   'snow_albedo'       — snow pack effect on surface albedo
    #   'amoc_acc'          — dynamic AMOC/ACC circulation weakening with ice
    #   'co2_greenhouse'    — CO2 temperature offset applied to T_base
    #   'vegetation_albedo' — biome/Köppen-based land albedo
    #   'ocean_transport'   — ocean heat transport ΔT calculation
    """Advance planet state forward by `days`.

    Updates temperature, wind, and precipitation based on new day_of_year.
    Interactions:
    - Temperature depends on insolation (day_of_year), wind advection, land-sea effects
    - Wind depends on temperature gradients
    - Precipitation depends on wind (advection/convergence), temperature (evaporation),
      and elevation (orographic uplift)
    
    Temperature now includes:
    - Longitudinal variation (coastal effects: land-sea contrast)
    - Meridional heat transport (winds carry heat poleward/equatorward)
    - Diurnal variation (approximate day/night cycle via longitude)

    Args:
        state: Current planet state
        days: Time step in days (default 1.0)
        block_size: Coarse resolution for simulation (larger = faster, less accurate)
        wind_block_size: Coarse resolution used for wind evolution. If None, uses `block_size`.
        update_wind: Whether to recompute wind field

    Returns:
        New state with updated day_of_year and atmospheric fields
    """
    pp = planet_params if planet_params is not None else EARTH

    # Resolve parameters that default to None → pp.<field> (allows per-planet tuning
    # while still permitting explicit overrides from the optimizer or tests).
    if ocean_transport_coeff is None:
        ocean_transport_coeff = pp.ocean_transport_coeff
    if polar_cooling_scale is None:
        polar_cooling_scale = pp.polar_cooling_scale
    if ice_albedo_strength is None:
        ice_albedo_strength = pp.ice_albedo_strength
    if thermal_diffusion is None:
        thermal_diffusion = pp.thermal_diffusivity
    if co2_climate_feedback is None:
        co2_climate_feedback = pp.co2_climate_feedback

    # Detect whether elevation is a loaded real-world DEM (vs procedural noise).
    # Loaded DEMs have a large fraction of exactly-zero ocean cells; procedural
    # terrain uses continuous noise values that rarely land on exactly 0.0.
    _zeros_frac = float(np.sum(state.elevation == 0.0)) / max(1, state.elevation.size)
    _is_loaded_dem = bool(_zeros_frac > 0.05)

    # Dynamic AMOC / ACC feedback — scale the parameterised ocean heat bonus by
    # the actual polar sea-ice state.
    #
    # Physics basis: AMOC strength is driven by the density contrast between warm
    # salty Atlantic water and cold dense deep water sinking in the Nordic Seas.
    # When extensive sea ice covers 60-75°N it freshens the surface layer via
    # meltwater and suppresses thermohaline sinking, weakening AMOC on multi-year
    # timescales (Dansgaard-Oeschger stadials).  Conversely, a nearly ice-free
    # sub-polar gyre allows full AMOC and strong poleward heat delivery.
    #
    # Reference values (Northern Hemisphere, 60-75°N band):
    #   <5 % ice  → amoc_factor = 1.00  (full strength)
    #   35 % ice  → amoc_factor = 0.30  (reduced to 30 % of nominal)
    #
    # ACC (Antarctic Circumpolar Current) is less variable, so a more conservative
    # range is used (minimum factor 0.50, reference threshold 15 % ice at 60-75°S).
    if state.ice_cover is not None:
        _H_ice = state.ice_cover.shape[0]
        _lat_ice = (0.5 - (np.arange(_H_ice, dtype=np.float32) + 0.5) / _H_ice) * 180.0
        _nh_rows = (_lat_ice >= 60.0) & (_lat_ice <= 75.0)
        _sh_rows = (_lat_ice >= -75.0) & (_lat_ice <= -60.0)
        _nh_ice_frac = float(np.mean(state.ice_cover[_nh_rows])) if np.any(_nh_rows) else 0.0
        _sh_ice_frac = float(np.mean(state.ice_cover[_sh_rows])) if np.any(_sh_rows) else 0.0
        amoc_factor = float(np.clip(1.0 - (_nh_ice_frac - 0.05) / 0.30, 0.30, 1.0))
        acc_factor  = float(np.clip(1.0 - (_sh_ice_frac - 0.15) / 0.30, 0.50, 1.0))
    else:
        amoc_factor = 1.0
        acc_factor  = 1.0
        _lat_ice = np.array([], dtype=np.float32)

    # Feature 3: salinity modulates AMOC strength.
    # Fresher N.Atlantic surface water (lower density) reduces thermohaline sinking.
    if state.salinity is not None and pp.has_liquid_water_ocean and pp.salinity_amoc_scale > 0.0:
        _na_rows_sal = (_lat_ice >= 50.0) & (_lat_ice <= 75.0)
        if np.any(_na_rows_sal):
            _na_sal = float(np.mean(state.salinity[_na_rows_sal]))
            sal_anomaly = _na_sal - pp.salinity_reference_psu
            sal_amoc = float(np.clip(1.0 + 0.15 * sal_anomaly * pp.salinity_amoc_scale, 0.15, 1.5))
            amoc_factor = float(np.clip(amoc_factor * sal_amoc, 0.15, 1.0))

    # Apply feedback flags — freeze individual feedback loops at neutral state for testing.
    # Planet-level disables (has_liquid_water_ocean=False) are merged in as flag overrides.
    _fb = dict(feedback_flags) if feedback_flags else {}
    if not pp.has_liquid_water_ocean:
        _fb.setdefault('ocean_transport', False)
        _fb.setdefault('ice_albedo', False)
    if not _fb.get('amoc_acc', True):
        amoc_factor = 1.0
        acc_factor = 1.0

    eps_eq = float(pp.epsilon_equator if epsilon_equator is None else epsilon_equator)
    eps_pole = float(pp.epsilon_pole if epsilon_pole is None else epsilon_pole)
    new_day = (state.day_of_year + days) % pp.orbital_period_days
    new_total_days = float(state.total_days) + float(days)
    H, W = state.elevation.shape
    Hc, Wc = (max(1, (H + block_size - 1) // block_size),
              max(1, (W + block_size - 1) // block_size))
    wind_bs = max(1, int(block_size if wind_block_size is None else wind_block_size))
    Hcw, Wcw = (max(1, (H + wind_bs - 1) // wind_bs),
                max(1, (W + wind_bs - 1) // wind_bs))

    # ------------------------------------------------------
    # Climate Averaging and Köppen Classification
    # ------------------------------------------------------
    # Update 10-year rolling climate averages for general smoothing
    # Update monthly statistics for Köppen seasonality detection
    # Reclassify Köppen climate zones every 30 days

    # Update climate averages (exponential moving average)
    temp_avg, precip_avg, sample_days = update_climate_averages(
        state, days, window_days=10.0 * 365.2422  # 10-year averaging window
    )

    # Update monthly statistics for Köppen classification
    monthly_temp, monthly_precip, monthly_sample_count = update_monthly_statistics(
        state, days, window_years=1.0  # 1-year rolling average per month
    )

    # Initialize biome/Köppen variables
    biome_new = state.biome_type
    koppen_new = state.koppen_type
    biome_last_update = state.biome_last_update_day

    # Reclassify Köppen climate zones every 30 days
    BIOME_UPDATE_INTERVAL = 30.0  # 30 days (was 3 years)
    days_since_biome_update = new_total_days - state.biome_last_update_day

    # Compute Köppen classification if monthly data is available
    if monthly_temp is not None and monthly_precip is not None:
        if days_since_biome_update >= BIOME_UPDATE_INTERVAL or state.koppen_type is None:
            # Time to update Köppen classification
            _, land_mask_for_biomes = get_masks(state.elevation)
            # Coarse (block-averaged) elevation, upsampled back to full resolution -
            # this is the elevation baseline the temperature physics already used for
            # its own orographic lapse-rate cooling (_evolve_temperature). Passing it
            # lets classify_koppen apply only the *additional* fine-grained delta
            # (peaks colder / valleys warmer than their block average) instead of
            # double-applying the full lapse rate on an already-cooled input.
            elev_c_for_biomes = _coarsen_elevation_cached(state.elevation, Hc, Wc, block_size)
            elev_baseline_for_biomes = (
                _upsample_bilinear_many({"elev": elev_c_for_biomes}, H, W, block_size)["elev"]
                if block_size > 1 else elev_c_for_biomes
            )
            koppen_new = classify_koppen(
                monthly_temp, monthly_precip, land_mask_for_biomes,
                elevation=state.elevation,
                elevation_baseline=elev_baseline_for_biomes,
            )
            # Convert Köppen to legacy biome for backward compatibility
            biome_new = koppen_to_legacy_biome(koppen_new)
            biome_last_update = new_total_days
            if debug_log:
                LOG.info(f"[Köppen Update] Day {new_total_days:.0f} - Climate zones reclassified from monthly data")
        else:
            # Keep existing classification
            koppen_new = state.koppen_type
            biome_new = state.biome_type
            biome_last_update = state.biome_last_update_day
    else:
        # Monthly data not yet initialized - will be computed on next step
        koppen_new = state.koppen_type
        biome_new = state.biome_type
        biome_last_update = state.biome_last_update_day

    # ------------------------------------------------------
    # Antarctic Ice Sheet Initial Seeding (loaded real-earth DEM only)
    # ------------------------------------------------------
    # The Antarctic ice sheet is ~26.5 million km³ of land ice accumulated over
    # millions of years — a boundary condition that must be imposed explicitly for
    # the real Earth, not spun up from scratch in a few simulated years.
    #
    # Rule: land tiles south of -60° are seeded as EF (ice cap) ONCE on the very
    # first step when a loaded DEM is detected.  After that, the normal 30-day
    # Köppen reclassification takes over — if a tile warms enough (T_warmest ≥ 0°C)
    # it will naturally transition away from EF.
    #
    # All OTHER land tiles (non-Antarctic or procedural terrain) must earn EF
    # classification through the natural growth mechanism tracked by `ice_sheet_age`
    # (see below).
    _is_first_init = (state.ice_sheet_age is None)
    if _is_loaded_dem and _is_first_init:
        lat_1d_seed = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0
        _, land_mask_seed = get_masks(state.elevation)
        # Cells south of -60° that are land
        antarctic_ice_seed = (lat_1d_seed[:, None] < -60.0) & land_mask_seed  # (H, W)
        if koppen_new is None:
            # First step with no monthly data — initialize all land/ocean to ET/ocean
            koppen_new = np.where(land_mask_seed, 18, 0).astype(np.int32)  # 18=ET, 0=ocean
        else:
            koppen_new = koppen_new.copy()
        koppen_new[antarctic_ice_seed] = 19  # KOPPEN_EF (ice cap)
        biome_new = koppen_to_legacy_biome(koppen_new)
        biome_last_update = new_total_days

    # ------------------------------------------------------
    # Ice Sheet Age Tracking
    # ------------------------------------------------------
    # Ice sheets form over centuries/millennia from sustained snow accumulation in
    # areas where ablation < accumulation. In this model the proxy for "mature ice
    # sheet" is: a land cell must be continuously classified as EF (T_warmest < 0°C
    # from monthly averages) for ICE_SHEET_THRESHOLD_DAYS before it receives the
    # high ice-sheet albedo (0.80).  Until that threshold is reached the cell is
    # physically treated as tundra (albedo 0.25) even though Köppen labels it EF.
    #
    # Seeded Antarctic cells start with age = threshold (already mature at t=0).
    # All other cells start at age = 0 and must grow naturally.
    ICE_SHEET_THRESHOLD_DAYS = 3.0 * pp.orbital_period_days  # 3 years of sustained EF conditions

    if _is_first_init:
        ice_sheet_age_new = np.zeros((H, W), dtype=np.float32)
        if _is_loaded_dem:
            # Seeded Antarctic cells are already at threshold — full albedo from step 1
            lat_1d_age = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0
            _, land_mask_age = get_masks(state.elevation)
            antarctic_mask_age = (lat_1d_age[:, None] < -60.0) & land_mask_age
            ice_sheet_age_new[antarctic_mask_age] = ICE_SHEET_THRESHOLD_DAYS
    else:
        ice_sheet_age_new = state.ice_sheet_age.copy().astype(np.float32, copy=False)

    # Update age each step: EF-classified land cells accumulate toward threshold;
    # non-EF land cells lose age at half the gain rate (hysteresis — ice sheets
    # are slow to melt even after conditions warm slightly).
    if koppen_new is not None:
        _, _land_mask_upd = get_masks(state.elevation)
        _ef_land = (koppen_new == 19) & _land_mask_upd
        ice_sheet_age_new = np.where(
            _ef_land,
            np.minimum(ice_sheet_age_new + float(days), ICE_SHEET_THRESHOLD_DAYS),
            np.maximum(ice_sheet_age_new - float(days) * 0.5, 0.0),
        ).astype(np.float32, copy=False)

    # ------------------------------------------------------
    # CO2 Greenhouse Forcing (if carbon cycle enabled)
    # ------------------------------------------------------
    # CRITICAL FIX: CO2 forcing must be applied to BASE TEMPERATURE before temperature evolution,
    # not added to final temperature afterward (which would cause runaway warming).
    # The forcing represents the equilibrium temperature offset that the simulation should relax toward.
    co2_temp_offset = 0.0
    _co2_ref = pp.co2_baseline_ppm if pp.co2_baseline_ppm > 1.0 else CO2_PREINDUSTRIAL
    if enable_carbon_cycle and _fb.get('co2_greenhouse', True):
        co2_forcing = co2_radiative_forcing(state.co2_atmosphere, _co2_ref)
        co2_temp_offset = co2_temperature_response(co2_forcing, co2_climate_feedback)
        # Feature 4: CH4 radiative forcing added to equilibrium temperature offset
        if pp.ch4_baseline_ppb > 0.0:
            from carbon_cycle import ch4_radiative_forcing as _ch4_rf
            _ch4_forcing = _ch4_rf(state.ch4_atmosphere, pp.ch4_baseline_ppb)
            co2_temp_offset += co2_climate_feedback * _ch4_forcing
        if debug_log:
            LOG.info(f"CO2={state.co2_atmosphere:.1f} ppm, forcing={co2_forcing:.2f} W/m², T_offset={co2_temp_offset:.2f}K")

    # Get base insolation temperature (latitude-dependent) + CO2 offset
    # Ocean: seasonal lag of ~50 days (1.5 months) due to high heat capacity
    # Land: immediate response to current insolation
    lat = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi

    # Calculate temperature for current day (land response)
    T_lat_land = temperature_kelvin_for_lat(
        lat,
        day_of_year=int(new_day),
        polar_cooling_scale=polar_cooling_scale,
        planet_params=pp,
    )
    T_base_land = np.repeat(T_lat_land[:, None], Wc, axis=1).astype(np.float32, copy=False) + co2_temp_offset
    # Land summer temperature cap: temperature_kelvin_for_lat gives radiative equilibrium,
    # but real land surfaces never reach those peaks because latent heat (ET), soil heat
    # capacity, and turbulent exchange all limit summer temperatures.
    # At 55-65°N the formula returns ~305 K (32°C) in summer — unrealistically hot —
    # pulling the simulation 12-13°C above Earth via the land_blend.
    # Cap: 301 K (~28°C) held through 0-45°, tapering linearly to 286 K (13°C) at 60°+.
    # This matches observed mid-latitude summer maxima for mean daily temperatures.
    #
    # Taper start moved from 0deg to 45deg (2026-07, wind/precip-model revisit): the old
    # linear-from-the-equator taper capped land at only ~290.6K (17.5C) at ~41.5N in
    # summer -- measured directly on real terrain to be *below* the ocean's own summer
    # temperature at the same latitude (~296-298K) year-round, including at the summer
    # peak. Real land is warmer than adjacent ocean in summer (lower heat capacity, faster
    # response) -- that's the thermal-low mechanism that drives monsoon-style onshore
    # moisture inflow into continental interiors. With the cap capping land *colder* than
    # the ocean even at peak summer, that sign was backwards, and wind divergence over
    # continental-interior boxes measured persistently positive (divergent, not
    # convergent) with no seasonal signal at all -- see moisture-transport-investigation
    # memory. Holding 301K through 45deg (same ceiling already accepted at the equator,
    # just extended) restores the correct summer land > ocean sign at mid-latitudes while
    # leaving the original 55-65N fix (this taper's whole reason for existing) untouched:
    # the endpoint is still 286K at 60deg+.
    _abs_lat_deg_land = np.abs(np.rad2deg(lat))  # (Hc,)
    _land_cap_1d = 301.0 - 15.0 * np.clip((_abs_lat_deg_land - 45.0) / 15.0, 0.0, 1.0)
    # Atmospheric meridional heat transport warms high-latitude land.
    # The Ferrel cell and synoptic eddies carry ~60% of the ocean transport value
    # poleward even over the Antarctic continent.  Without this, Antarctic winter
    # equilibrium falls to ~185 K; the 200 K floor then dominates the annual mean,
    # giving ~203 K (vs Earth 224 K at the South Pole).  Adding 0.60 × 40 K = 24 K
    # at the pole lifts the winter minimum to ~210 K and the annual mean to ~220-226 K.
    # Symmetric for both poles (no AMOC; that term is ocean-only and applied separately).
    # Atmospheric meridional heat transport to high-latitude LAND.
    # Increased coefficient from 0.42 → 0.65: synoptic eddies, the Ferrel cell, and
    # frontal systems deliver substantially more heat to high-latitude continental
    # interiors than the previous value implied.  At 65°N the Ferrel/eddy transport
    # raises winter land T_base by ~18K (vs ~11K before), which is the primary driver
    # of the NH zonal-mean cold bias: when winter T_base_land(65-85°N) was too cold,
    # land cells dragged the zonal mean ~10°C below Earth reference.
    # The formula only ramps above 42°, so latitudes ≤50°N change by <1K. ✓
    _atm_land_transport_1d = (
        0.65 * 34.0 * np.clip((_abs_lat_deg_land - 42.0) / 28.0, 0.0, 1.0) ** 1.5
    )
    # Mid-latitude storm-track heat transport (22-50°): winter cyclones and frontal
    # systems deliver substantial poleward heat well before the 42° ramp above kicks
    # in. Without this, zonal-mean land coldest-month temperatures at 30-55° come out
    # 15-30K too cold (observed: -37 to -40°C at 45-55°N vs Earth's -5 to -15°C),
    # which spuriously classifies most of Canada/Siberia/Central Asia as Dwd (extreme
    # continental, requires coldest month < -38°C) instead of Dfb. Modeled as a
    # trapezoid (ramps 22°→42°, flat to 42°, decays 42°→50° handing off to the ramp
    # above) rather than a Gaussian so it doesn't reopen the 60°+ balance already
    # tuned by the ramp above. Cuts off by 50° (not 60°) so it doesn't dampen the
    # sea-ice/CO2 sensitivity signal at ice-forming latitudes (test_2x_co2_less_ice).
    _midlat_rise = np.clip((_abs_lat_deg_land - 22.0) / 20.0, 0.0, 1.0)
    _midlat_fall = np.clip((50.0 - _abs_lat_deg_land) / 8.0, 0.0, 1.0)
    _midlat_storm_bonus_1d = 27.0 * _midlat_rise * _midlat_fall
    T_base_land = T_base_land + (_atm_land_transport_1d + _midlat_storm_bonus_1d)[:, None].astype(np.float32, copy=False)

    # Evapotranspiration/convective cooling (2026-07, root-cause fix for the summer
    # overheating _land_cap_1d above only ever patched post-hoc): real land loses
    # substantial absorbed energy evaporating soil/plant moisture rather than raising
    # sensible temperature, especially at high summer insolation -- and critically,
    # this depends on how much moisture is actually *available*, not just latitude.
    # Measured directly: temperature_kelvin_for_lat's own radiative-equilibrium calc
    # reaches ~305-314K (33-41C) essentially UNIFORMLY across 20-70N at NH summer
    # solstice (no evapotranspiration physics at all below that function's own
    # is_polar>55deg threshold, and even above it the mechanism is tuned very weak
    # via polar_cooling_scale) -- a flat, moisture-blind profile that _land_cap_1d
    # was reshaping into a realistic latitude-dependent ceiling via a hard, seasonally
    # -discontinuous, moisture-blind clamp. This adds the missing physical mechanism
    # instead: cooling scales with (a) how far the pre-cooling temperature exceeds a
    # reference (more excess energy = more evaporative demand), (b) local soil
    # moisture (deserts get little cooling and stay realistically hot -- e.g. Sahara
    # summer means are genuinely very high -- while moist continental interior/boreal
    # land gets strong cooling), and (c) the local hemisphere's own seasonal cycle via
    # solar declination (near-zero in winter, peaking at the local summer solstice) so
    # the transition is smooth in time rather than an instant on/off clamp.
    # _land_cap_1d is kept below as a safety-net backstop for any remaining extreme
    # case, not the primary mechanism.
    if state.soil_moisture is not None:
        _soil_2d = _coarsen_many({"soil": state.soil_moisture}, Hc, Wc, block_size)["soil"]
    else:
        _soil_2d = np.full((Hc, Wc), 0.55, dtype=np.float32)
    if state.elevation is not None:
        _elev_c_early = _coarsen_elevation_cached(state.elevation, Hc, Wc, block_size)
        _, _land_mask_early = get_masks(_elev_c_early)
    else:
        _land_mask_early = np.zeros((Hc, Wc), dtype=bool)
    _gamma_season = 2.0 * np.pi * (float(new_day) - 80.0) / float(pp.orbital_period_days)
    _delta_season = float(np.arcsin(np.clip(np.sin(pp.obliquity_rad) * np.sin(_gamma_season), -1.0, 1.0)))
    _summer_signal_1d = np.sign(lat) * (_delta_season / max(pp.obliquity_rad, 1e-6))
    _summer_factor_1d = np.clip(_summer_signal_1d, 0.0, 1.0).astype(np.float32)  # 0 in winter, 1 at local summer peak
    _EVAP_COOL_THRESHOLD_K = 290.0
    _EVAP_COOL_COEFF_MAX = 0.85
    _evap_excess_2d = np.maximum(T_base_land - _EVAP_COOL_THRESHOLD_K, 0.0)
    _evap_cooling_2d = (
        _summer_factor_1d[:, None] * _EVAP_COOL_COEFF_MAX * np.clip(_soil_2d, 0.0, 1.0) * _evap_excess_2d
    ) * _land_mask_early.astype(np.float32)
    T_base_land = (T_base_land - _evap_cooling_2d).astype(np.float32, copy=False)

    # Re-apply summer cap: atmospheric transport can only raise winter/polar-night
    # temperatures; it must not push summer land above observed peak means.
    # Now a rarely-binding safety net (see evapotranspiration cooling above), not the
    # primary mechanism.
    T_base_land = np.minimum(T_base_land, _land_cap_1d[:, None].astype(np.float32, copy=False))

    # Calculate temperature for lagged day (ocean response with 1.5 month delay)
    lag_days = pp.ocean_lag_days * (float(pp.orbital_period_days) / 365.2422)  # scale thermal lag with year length
    lagged_day = (new_day - lag_days) % float(pp.orbital_period_days)
    T_lat_ocean_lagged = temperature_kelvin_for_lat(
        lat,
        day_of_year=int(lagged_day),
        polar_cooling_scale=polar_cooling_scale,
        planet_params=pp,
    )

    # CRITICAL: Two corrections to ocean base temperature:
    #
    # 1) SEASONAL AMPLITUDE DAMPING: The ocean's thermal time constant is ~1-3 YEARS,
    #    far longer than a season. SST barely oscillates around the annual mean.
    #    The 50-day lag shifts phase but doesn't damp amplitude. Without damping,
    #    winter T_base at 55-60N drops to 210-230K causing unrealistic ice.
    #
    # 2) MERIDIONAL HEAT TRANSPORT WARMING: temperature_kelvin_for_lat computes LOCAL
    #    radiative equilibrium, which ignores the ~2 PW of poleward ocean heat transport.
    #    Real SST at 55-70N is 12-42K warmer than radiative equilibrium due to Gulf
    #    Stream, Kuroshio, and thermohaline circulation. This offset is standard in
    #    energy balance climate models (Budyko 1969, Sellers 1969).

    T_lat_annual_mean = temperature_kelvin_for_lat(
        lat,
        day_of_year=(80.0 / 365.2422) * float(pp.orbital_period_days),  # Earth-relative spring equinox proxy
        polar_cooling_scale=polar_cooling_scale,
        planet_params=pp,
    )

    # Meridional heat transport warming: concentrated at high latitudes
    # 0K below 40°, ramping to 40K at 70°+ (matches observed SST - radiative eq deficit)
    # Profile: steep ramp starting at 40° prevents over-warming subtropics
    # The explicit ocean transport function handles finer redistribution (western
    # boundary currents, seasonal variation, east-west asymmetry)
    #
    # AMOC asymmetry (physically motivated): The Atlantic Meridional Overturning
    # Circulation transports ~1.2 PW northward into the Arctic with no SH equivalent.
    # This is why Earth's Arctic Ocean is ~10-15°C warmer than the Southern Ocean at
    # the same latitude.  Without this term both poles get identical base temperatures,
    # the NH Arctic falls into an ice-albedo runaway while the SH warms excessively.
    # Fix: NH latitudes > 50° receive a bonus +18 K (AMOC-driven warming), ramping
    # over 50-75°N where AMOC heat delivery is strongest.
    # Start at 50° (not 60°): the T_base_land summer cap already prevents mid-latitude
    # overheating, so we no longer need to restrict the AMOC ramp to avoid it.
    # Starting at 60° (Round 2) gave 0 K at 60°N and only 11 K at 70°N, which was
    # insufficient to keep Arctic SSTs above the ice melt threshold (NH edge regressed
    # from 70°N to 54°N).  50° start gives 7 K at 60°N and 14 K at 70°N — same as
    # the successful Round 1 ramp but with a slightly lower peak (18 vs 20 K).
    # SH transport raised from 50% to 65% of base: the Southern Ocean needs more warmth
    # to avoid its own cold runaway, while still remaining cooler than AMOC-warmed NH.
    lat_deg_1d = np.abs(np.rad2deg(lat))
    _ocean_scale = float(pp.has_liquid_water_ocean)
    # Scale AMOC/ACC with planet rotation rate and ocean fraction.
    # AMOC strength ∝ ω^0.4 (Coriolis drives western boundary currents; weaker on slow rotators).
    # AMOC suppressed entirely for retrograde rotators (Coriolis deflects opposite → no WBC).
    # ACC is primarily wind-driven so scales only with ocean_fraction, not rotation.
    _EARTH_OMEGA = 7.2921e-5  # rad/s  (2π / 23.9345 h)
    _rotation_scale = float(np.clip((pp.omega / _EARTH_OMEGA) ** 0.4, 0.05, 2.0))
    _ocean_frac_scale = float(pp.ocean_fraction / 0.71)
    _amoc_scale = _ocean_scale * _rotation_scale * _ocean_frac_scale * float(pp.rotation_direction > 0)
    _acc_scale  = _ocean_scale * _ocean_frac_scale
    _transport_base = _acc_scale * 34.0 * np.clip((lat_deg_1d - 42.0) / 28.0, 0.0, 1.0) ** 1.5
    # AMOC bonus: steep ramp from 65-75°N (3K at 65°N → 18K at 75°N+).
    # Scaled by dynamic feedback factor (amoc_factor: 0.30–1.00) and planet rotation/ocean params.
    # Geographic taper: bonus tapers to zero above pp.amoc_cutoff_lat to prevent NH pole over-warming.
    _amoc_taper = np.clip((pp.amoc_cutoff_lat - lat_deg_1d) / 10.0, 0.0, 1.0)
    _amoc_bonus = _amoc_scale * amoc_factor * _amoc_taper * np.where(
        lat > 0,
        pp.amoc_bonus_near * np.clip((lat_deg_1d - 42.0) / 23.0, 0.0, 1.0)
        + pp.amoc_bonus_far * np.clip((lat_deg_1d - 65.0) / 10.0, 0.0, 1.0),
        0.0,
    )  # NH only; tapers to 0 above amoc_cutoff_lat
    # ACC (Antarctic Circumpolar Current) bonus scaled by acc_factor (0.50–1.00).
    # Extensive Antarctic sea ice partially blocks CDW upwelling and reduces
    # the net poleward heat delivery by the ACC.
    _acc_bonus = _acc_scale * acc_factor * np.where(
        lat < 0,
        pp.acc_bonus_near * np.clip((lat_deg_1d - 55.0) / 10.0, 0.0, 1.0)
        + pp.acc_bonus_far * np.clip((lat_deg_1d - 65.0) / 10.0, 0.0, 1.0),
        0.0,
    )  # SH only; at 75-85°S total = acc_bonus_near+acc_bonus_far at full strength
    _sh_factor = np.where(lat > 0, 1.0, 0.58)   # SH gets weaker baseline transport than NH (no AMOC)
    # _transport_base's ramp only depends on |lat|, so it stays flat at its 34K max
    # all the way to the exact pole — unlike amoc_bonus, which already tapers to
    # zero above amoc_cutoff_lat. That left the NH pole cell ~30K warmer than
    # intended (the dominant cause of the too-small NH equator-pole gradient) and
    # made amoc_bonus_near/far tuning ineffective, since the metric samples the
    # exact pole row that amoc_bonus never reaches. Taper the NH share of the
    # generic transport too, so basin-average heat delivery also falls off near
    # the pole. Uses its own narrower (5°) taper rather than _amoc_taper's 10°:
    # widening it to overlap 60-70°N measurably fought the eddy-heat-flux
    # Laplacian smoothing (which acts over 20-70°) and *increased* zonal-mean
    # variance instead of reducing it once eddies were enabled (tested — 20°
    # width made the interaction worse, not better). Keeping the ramp entirely
    # above 70°N (75-85°N here) avoids overlapping the eddy band at all. SH
    # (ACC) side is left untouched — out of scope for the NH gradient fix.
    _nh_transport_taper = np.clip((pp.amoc_cutoff_lat - lat_deg_1d) / 5.0, 0.0, 1.0)
    _nh_transport_taper = np.where(lat > 0, _nh_transport_taper, 1.0)
    transport_warming = _transport_base * _sh_factor * _nh_transport_taper + _amoc_bonus + _acc_bonus

    # Seasonal fraction: what fraction of the radiative swing the ocean actually feels
    # Based on ΔT/ΔT_rad ≈ 1/sqrt(1 + (2π τ/P)²) where τ ~ 1-3 years, P = 1 year
    # Equator (τ~0.7yr): ~24%, Mid-lat (τ~1.5yr): ~10%, Polar (τ~3yr): ~5%
    obliq_ratio = float(pp.obliquity_deg) / 23.44
    obliq_factor = np.clip(obliq_ratio, 0.6, 2.0) ** 0.5
    polar_lat_boost = np.sin(np.deg2rad(lat_deg_1d)) ** 2
    high_obliq_boost = max(obliq_ratio - 1.0, 0.0)
    ocean_seasonal_frac = (
        (0.05 + 0.20 * np.cos(np.deg2rad(lat_deg_1d))) * obliq_factor
        + 0.60 * high_obliq_boost * polar_lat_boost
    )
    # High-obliquity planets have larger polar insolation swings; allow a proportionally
    # higher seasonal fraction rather than capping at Earth's 0.45.
    _seasonal_cap = float(min(0.45 * obliq_factor, 0.85))
    ocean_seasonal_frac = np.clip(ocean_seasonal_frac, 0.03, _seasonal_cap)

    # Final ocean base: annual mean + transport warming + small seasonal oscillation
    T_lat_ocean = (T_lat_annual_mean + transport_warming
                   + ocean_seasonal_frac * (T_lat_ocean_lagged - T_lat_annual_mean))

    T_base_ocean = np.repeat(T_lat_ocean[:, None], Wc, axis=1).astype(np.float32, copy=False) + co2_temp_offset

    def _compute_T_base_ocean_full() -> np.ndarray:
        """Full-resolution fallback base temperature.

        Only needed when wind evolves at full resolution (wind_bs <= 1) before
        `state.temperature` is initialized — computed lazily because the
        production path (wind_bs > 1 with initialized temperature) never uses
        it, and this block costs three full-resolution
        temperature_kelvin_for_lat calls plus the transport math per step.
        """
        lat_full = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
        T_lat_ocean_full_lagged = temperature_kelvin_for_lat(
            lat_full,
            day_of_year=int(lagged_day),
            polar_cooling_scale=polar_cooling_scale,
            planet_params=pp,
        )
        T_lat_annual_mean_full = temperature_kelvin_for_lat(
            lat_full,
            day_of_year=(80.0 / 365.2422) * float(pp.orbital_period_days),
            polar_cooling_scale=polar_cooling_scale,
            planet_params=pp,
        )
        lat_deg_full = np.abs(np.rad2deg(lat_full))
        _transport_base_full = _acc_scale * 34.0 * np.clip((lat_deg_full - 42.0) / 28.0, 0.0, 1.0) ** 1.5
        _amoc_taper_full = np.clip((pp.amoc_cutoff_lat - lat_deg_full) / 10.0, 0.0, 1.0)
        _amoc_bonus_full = _amoc_scale * amoc_factor * _amoc_taper_full * np.where(
            lat_full > 0,
            pp.amoc_bonus_near * np.clip((lat_deg_full - 42.0) / 23.0, 0.0, 1.0)
            + pp.amoc_bonus_far * np.clip((lat_deg_full - 65.0) / 10.0, 0.0, 1.0),
            0.0,
        )
        _acc_bonus_full = _acc_scale * acc_factor * np.where(
            lat_full < 0,
            pp.acc_bonus_near * np.clip((lat_deg_full - 55.0) / 10.0, 0.0, 1.0)
            + pp.acc_bonus_far * np.clip((lat_deg_full - 65.0) / 10.0, 0.0, 1.0),
            0.0,
        )
        _sh_factor_full = np.where(lat_full > 0, 1.0, 0.58)
        _nh_transport_taper_full = np.clip((pp.amoc_cutoff_lat - lat_deg_full) / 5.0, 0.0, 1.0)
        _nh_transport_taper_full = np.where(lat_full > 0, _nh_transport_taper_full, 1.0)
        transport_warming_full = (
            _transport_base_full * _sh_factor_full * _nh_transport_taper_full
            + _amoc_bonus_full + _acc_bonus_full
        )
        polar_lat_boost_full = np.sin(np.deg2rad(lat_deg_full)) ** 2
        ocean_seasonal_frac_full = (
            (0.05 + 0.20 * np.cos(np.deg2rad(lat_deg_full))) * obliq_factor
            + 0.60 * high_obliq_boost * polar_lat_boost_full
        )
        ocean_seasonal_frac_full = np.clip(ocean_seasonal_frac_full, 0.03, _seasonal_cap)
        T_lat_ocean_full = (T_lat_annual_mean_full + transport_warming_full
                            + ocean_seasonal_frac_full * (T_lat_ocean_full_lagged - T_lat_annual_mean_full))
        return np.repeat(T_lat_ocean_full[:, None], W, axis=1).astype(np.float32, copy=False) + co2_temp_offset

    
    # Blend based on land fraction (will be calculated in _evolve_temperature)
    # Use ocean-lagged temperature as base; _evolve_temperature will handle land/ocean mixing
    T_base = T_base_ocean  # Start with ocean (lagged), land will be corrected in evolution
    
    # Compute coarse elevation grid for wind/temperature evolution
    elev_c = _coarsen_elevation_cached(state.elevation, Hc, Wc, block_size) if state.elevation is not None else None

    # Update temperature with wind advection and land-sea effects.
    # Batched (see `_coarsen_many`): T_prev/T_air/ice/ice_thickness all share the
    # same (Hc, Wc, block_size) coarsening. Fallbacks (T_base.copy(), etc.) are
    # applied after the batch, same as the original per-field logic — a fallback
    # never depends on a value the batch itself needed to produce, except
    # T_air_coarse's fallback to T_prev_coarse, which is resolved first below.
    _group_a_in: dict[str, np.ndarray] = {}
    if state.temperature is not None:
        _group_a_in["T_prev"] = state.temperature
    # Downsample T_air; initialize from T_sst if not yet present (first step or old save)
    _T_air_src = state.air_temperature if state.air_temperature is not None else state.temperature
    if _T_air_src is not None:
        _group_a_in["T_air"] = _T_air_src
    if state.ice_cover is not None:
        _group_a_in["ice"] = state.ice_cover
    # Feature 6: sea ice thickness — initialize to 1 m where ice exists on first step
    _ice_thick_src = state.ice_thickness
    if _ice_thick_src is None and state.ice_cover is not None:
        _ice_thick_src = np.where(state.ice_cover > 0, 1.0, 0.0).astype(np.float32, copy=False)
    if _ice_thick_src is not None:
        _group_a_in["ice_thick"] = _ice_thick_src

    _group_a_out = _coarsen_many(_group_a_in, Hc, Wc, block_size)
    T_prev_coarse = _group_a_out["T_prev"] if "T_prev" in _group_a_out else T_base.copy()
    T_air_coarse = _group_a_out["T_air"] if "T_air" in _group_a_out else T_prev_coarse.copy()
    ice_prev_coarse = _group_a_out.get("ice")
    ice_thick_prev_coarse = _group_a_out.get("ice_thick")
    
    # ------------------------------------------------------
    # Jet stream dynamics: persistent meander index + blocking events
    # (see atmosphere._update_jet_index / _update_jet_blocking). Computed once
    # per step from the actual simulated temperature field -- weaker
    # pole-equator gradient nudges the index toward "wavy" -- then fed into
    # whichever evolve_wind() call below executes.
    # ------------------------------------------------------
    lat_c_deg_1d = np.rad2deg((0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi)
    T_zm = np.mean(T_prev_coarse, axis=1).astype(np.float64, copy=False)

    def _pole_eq_gradient(hemi_sign: float) -> float:
        trop_mask = (lat_c_deg_1d * hemi_sign >= 0.0) & (np.abs(lat_c_deg_1d) <= 30.0)
        polar_mask = (lat_c_deg_1d * hemi_sign >= 0.0) & (np.abs(lat_c_deg_1d) >= 60.0)
        if not np.any(trop_mask) or not np.any(polar_mask):
            return float(pp.jet_gradient_ref_k)
        return float(np.mean(T_zm[trop_mask]) - np.mean(T_zm[polar_mask]))

    jet_index_nh_new = _update_jet_index(
        state.jet_index_nh, _pole_eq_gradient(1.0), days, new_total_days, hemisphere_seed=1,
        tau_days=float(pp.jet_meander_tau_days), noise_amp=float(pp.jet_meander_noise_amp),
        gradient_ref_k=float(pp.jet_gradient_ref_k),
    )
    jet_index_sh_new = _update_jet_index(
        state.jet_index_sh, _pole_eq_gradient(-1.0), days, new_total_days, hemisphere_seed=2,
        tau_days=float(pp.jet_meander_tau_days), noise_amp=float(pp.jet_meander_noise_amp),
        gradient_ref_k=float(pp.jet_gradient_ref_k),
    )
    jet_block_lon_nh_new, jet_block_days_left_nh_new, jet_block_total_nh_new = _update_jet_blocking(
        state.jet_block_lon_nh, state.jet_block_days_left_nh, state.jet_block_total_days_nh,
        jet_index_nh_new, days, new_total_days, hemisphere_seed=1,
        trigger_rate_per_day=float(pp.jet_block_trigger_rate_per_day),
        duration_range_days=pp.jet_block_duration_range_days,
    )
    jet_block_lon_sh_new, jet_block_days_left_sh_new, jet_block_total_sh_new = _update_jet_blocking(
        state.jet_block_lon_sh, state.jet_block_days_left_sh, state.jet_block_total_days_sh,
        jet_index_sh_new, days, new_total_days, hemisphere_seed=2,
        trigger_rate_per_day=float(pp.jet_block_trigger_rate_per_day),
        duration_range_days=pp.jet_block_duration_range_days,
    )
    _jet_block_nh = (jet_block_lon_nh_new, jet_block_days_left_nh_new, jet_block_total_nh_new)
    _jet_block_sh = (jet_block_lon_sh_new, jet_block_days_left_sh_new, jet_block_total_sh_new)

    # ------------------------------------------------------
    # NEW: Prognostic Wind Evolution (Physics Items 16-33)
    # ------------------------------------------------------
    # If wind is None, initialize it near-rest (small noise) so circulation spins up
    # from pressure gradients (Hadley-like overturning) rather than a synthetic target.
    if state.wind_u is None or state.wind_v is None:
        rng = np.random.default_rng(12345)
        u_full = rng.normal(0.0, 0.15, size=(H, W)).astype(np.float32, copy=False)
        v_full = rng.normal(0.0, 0.15, size=(H, W)).astype(np.float32, copy=False)
    else:
        u_full, v_full = state.wind_u, state.wind_v

    # 1.5-layer atmosphere: upper-level prognostic wind (Feature 8). Lazy-init
    # as a copy of the surface wind (a physically-reasonable warm start) if
    # this is the first step or an old save predates this field.
    if state.wind_u_aloft is None or state.wind_v_aloft is None:
        u2_full, v2_full = u_full.copy(), v_full.copy()
    else:
        u2_full, v2_full = state.wind_u_aloft, state.wind_v_aloft

    # Evolve wind at `wind_block_size` resolution (can differ from temperature/precip `block_size`)
    # Then upsample to full resolution for precipitation
    # Cached diagnostic wind for relaxation (once per day/shape/params).
    def _diag_wind_cached(h: int, w: int, temp_field: np.ndarray, elev_field: np.ndarray):
        key = (
            h,
            w,
            # `new_day` wraps every orbital period (0..~365), so keying on it
            # alone made this cache reuse year-1's storm/Rossby-wave snapshot
            # for every later year's same calendar day at MONTHLY/ANNUAL speed
            # -- freezing the diagnostic wind's weather into a single repeating
            # year instead of continuing to evolve with `new_total_days`. Key
            # on the monotonic day count instead; still only varies once per
            # simulated day, which was the actual perf intent.
            int(new_total_days),
            float(wind_target_weather_amp),
            float(wind_target_zonal_pressure),
            float(wind_target_terrain_pressure_amp),
            float(wind_target_terrain_flow_amp),
            round(float(jet_index_nh_new), 3),
            round(float(jet_index_sh_new), 3),
            tuple(round(float(x), 2) for x in _jet_block_nh),
            tuple(round(float(x), 2) for x in _jet_block_sh),
            round(float(pp.solar_constant), 4),
            round(float(pp.obliquity_deg), 4),
            round(float(pp.sidereal_day_hours), 4),
            round(float(pp.radius_m), 1),
            round(float(pp.pgf_continentality_amp), 4),
        )
        cache = _RELAX_CACHE
        if cache["key"] == key and cache["u"] is not None and cache["v"] is not None:
            return cache["u"], cache["v"]
        u_diag, v_diag = generate_wind_field(
            h,
            w,
            day_of_year=int(new_day),
            block_size=1,
            temperature=temp_field,
            elevation=elev_field,
            weather_amp=float(wind_target_weather_amp),
            zonal_pressure=float(wind_target_zonal_pressure),
            terrain_pressure_amp=float(wind_target_terrain_pressure_amp),
            terrain_flow_amp=float(wind_target_terrain_flow_amp),
            time_days=new_total_days,
            planet_params=pp,
            jet_index_nh=jet_index_nh_new,
            jet_index_sh=jet_index_sh_new,
            jet_block_nh=_jet_block_nh,
            jet_block_sh=_jet_block_sh,
        )
        cache.update({"key": key, "u": u_diag, "v": v_diag})
        return u_diag, v_diag

    # Honor `update_wind`: MONTHLY/ANNUAL substeps pass update_wind=False by
    # design (PLAN.md Open Question 1: "cached relaxation target, chosen for
    # speed"). This flag was silently ignored — wind (including storm systems
    # and jet dynamics) evolved prognostically every step in every mode,
    # costing the exact work those modes were designed to skip. When
    # update_wind=False, wind now follows the cached *diagnostic* wind
    # (generate_wind_field's seasonal climatology, refreshed once per
    # simulated day via _RELAX_CACHE) instead of either full prognostic
    # evolution (old behavior, expensive) or a permanently frozen field
    # (which would lose the seasonal wind cycle entirely on long
    # MONTHLY/ANNUAL runs). Wind still evolves prognostically when no wind
    # exists yet (first step).
    _do_evolve_wind = bool(update_wind) or state.wind_u is None or state.wind_v is None

    if not _do_evolve_wind:
        # Diagnostic/climatological wind on the wind grid, upsampled to full res.
        if wind_bs > 1:
            _elev_c_w = _coarsen_elevation_cached(state.elevation, Hcw, Wcw, wind_bs)
            if state.temperature is not None:
                _T_w = _coarsen(state.temperature, Hcw, Wcw, wind_bs)
            else:
                _T_w = None
            u_diag, v_diag = _diag_wind_cached(Hcw, Wcw, _T_w, _elev_c_w)
            uv = _upsample_bilinear_many({"u": u_diag, "v": v_diag}, H, W, wind_bs)
            u_full, v_full = uv["u"], uv["v"]
        else:
            u_full, v_full = _diag_wind_cached(H, W, state.temperature, state.elevation)
    elif wind_bs > 1:
        # Downsample wind/temperature/elevation/ice for evolution on the wind grid (batched).
        _group_b_in: dict[str, np.ndarray] = {
            "u": u_full, "v": v_full, "u2": u2_full, "v2": v2_full,
        }
        if state.temperature is not None:
            _group_b_in["T"] = state.temperature
        if state.ice_cover is not None:
            _group_b_in["ice"] = state.ice_cover
        _group_b_out = _coarsen_many(_group_b_in, Hcw, Wcw, wind_bs)
        u_coarse_evol = _group_b_out["u"]
        v_coarse_evol = _group_b_out["v"]
        u2_coarse_evol = _group_b_out["u2"]
        v2_coarse_evol = _group_b_out["v2"]
        ice_c_w = _group_b_out.get("ice")
        elev_c_w = _coarsen_elevation_cached(state.elevation, Hcw, Wcw, wind_bs)

        if state.temperature is not None:
            T_for_wind = _group_b_out["T"]
        else:
            # When temperature is not yet initialized, use the same lagged-ocean base but on the wind grid.
            lat_w = (0.5 - (np.arange(Hcw, dtype=np.float32) + 0.5) / Hcw) * np.pi
            T_lat_ocean_w = temperature_kelvin_for_lat(
                lat_w,
                day_of_year=int(lagged_day),
                polar_cooling_scale=polar_cooling_scale,
                planet_params=pp,
            )
            T_for_wind = np.repeat(T_lat_ocean_w[:, None], Wcw, axis=1).astype(np.float32, copy=False)

        # 1.5-layer atmosphere: evolve the upper-level wind first, at the same
        # wind-grid resolution, so evolve_wind's baroclinic mixing term below
        # can relax the surface toward this step's freshly-updated aloft wind.
        u2_coarse_evol, v2_coarse_evol = evolve_wind_aloft(
            u2_coarse_evol, v2_coarse_evol,
            temperature=T_for_wind,
            dt_days=days,
            pgf_temp_scale=float(wind_pgf_temp_scale),
            upper_pgf_amp=float(pp.wind_upper_pgf_amp),
            damping_rate=float(pp.wind_upper_damping),
            vmax_clip=float(wind_vmax_clip),
            planet_params=pp,
            hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
        )

        # Evolve at wind-grid resolution.
        u_coarse_evol, v_coarse_evol = evolve_wind(
            u_coarse_evol, v_coarse_evol,
            temperature=T_for_wind,
            pressure=None,
            elevation=elev_c_w,
            dt_days=days,
            damping=float(wind_damping),
            pgf_temp_scale=float(wind_pgf_temp_scale),
            pgf_terrain_scale=float(wind_pgf_terrain_scale),
            drag_base=float(wind_drag_base),
            drag_elev_scale=float(wind_drag_elev_scale),
            vmax_clip=float(wind_vmax_clip),
            baroclinic_jet_amp=float(wind_baroclinic_jet_amp),
            baroclinic_mix=float(wind_baroclinic_mix),
            cell_relax_days=float(wind_cell_relax_days),
            time_days=float(new_total_days),
            planet_params=pp,
            ice_cover=ice_c_w,
            jet_index_nh=jet_index_nh_new,
            jet_index_sh=jet_index_sh_new,
            jet_block_nh=_jet_block_nh,
            jet_block_sh=_jet_block_sh,
            u_aloft=u2_coarse_evol,
            v_aloft=v2_coarse_evol,
        )

        # Keep winds energized + seasonally varying by weakly relaxing toward a diagnostic wind
        # (generate_wind_field injects synoptic-scale "weather systems" seeded by day_of_year).
        if wind_relax > 0.0:
            u_diag, v_diag = _diag_wind_cached(Hcw, Wcw, T_for_wind, elev_c_w)
            a = float(np.clip(wind_relax, 0.0, 1.0))
            u_coarse_evol = (1.0 - a) * u_coarse_evol + a * u_diag
            v_coarse_evol = (1.0 - a) * v_coarse_evol + a * v_diag

        # Upsample back to full resolution using bilinear interpolation
        uv = _upsample_bilinear_many(
            {"u": u_coarse_evol, "v": v_coarse_evol, "u2": u2_coarse_evol, "v2": v2_coarse_evol},
            H, W, wind_bs,
        )
        u_full, v_full = uv["u"], uv["v"]
        u2_full, v2_full = uv["u2"], uv["v2"]
    else:
        # Full resolution evolution
        # If wind evolves at higher resolution than the temperature solver, drive it with the
        # coarse temperature field upsampled to full resolution. This avoids injecting
        # grid-scale temperature noise into the wind solver (which can blow up speeds),
        # while still allowing the wind numerics to run on the fine grid.
        T_wind_full = state.temperature if state.temperature is not None else _compute_T_base_ocean_full()
        elev_wind_full = state.elevation
        if wind_bs < block_size and block_size > 1:
            to_up = {}
            to_up["T"] = T_prev_coarse if state.temperature is not None else T_base
            if elev_c is not None:
                to_up["elev"] = elev_c
            up = _upsample_bilinear_many(to_up, H, W, block_size)
            T_wind_full = up["T"]
            if "elev" in up:
                elev_wind_full = up["elev"]

        # 1.5-layer atmosphere: evolve the upper-level wind first (same
        # full-resolution grid), so evolve_wind's baroclinic mixing term below
        # can relax the surface toward this step's freshly-updated aloft wind.
        u2_full, v2_full = evolve_wind_aloft(
            u2_full, v2_full,
            temperature=T_wind_full,
            dt_days=days,
            pgf_temp_scale=float(wind_pgf_temp_scale),
            upper_pgf_amp=float(pp.wind_upper_pgf_amp),
            damping_rate=float(pp.wind_upper_damping),
            vmax_clip=float(wind_vmax_clip),
            planet_params=pp,
            hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
        )

        u_full, v_full = evolve_wind(
            u_full, v_full,
            temperature=T_wind_full,
            pressure=None,
            elevation=elev_wind_full,
            dt_days=days,
            damping=float(wind_damping),
            pgf_temp_scale=float(wind_pgf_temp_scale),
            pgf_terrain_scale=float(wind_pgf_terrain_scale),
            drag_base=float(wind_drag_base),
            drag_elev_scale=float(wind_drag_elev_scale),
            vmax_clip=float(wind_vmax_clip),
            baroclinic_jet_amp=float(wind_baroclinic_jet_amp),
            baroclinic_mix=float(wind_baroclinic_mix),
            cell_relax_days=float(wind_cell_relax_days),
            time_days=float(new_total_days),
            planet_params=pp,
            ice_cover=state.ice_cover,
            jet_index_nh=jet_index_nh_new,
            jet_index_sh=jet_index_sh_new,
            jet_block_nh=_jet_block_nh,
            jet_block_sh=_jet_block_sh,
            u_aloft=u2_full,
            v_aloft=v2_full,
        )
        if wind_relax > 0.0:
            T_for_wind = T_wind_full
            u_diag, v_diag = _diag_wind_cached(H, W, T_for_wind, elev_wind_full)
            a = float(np.clip(wind_relax, 0.0, 1.0))
            u_full = (1.0 - a) * u_full + a * u_diag
            v_full = (1.0 - a) * v_full + a * v_diag

    # Winds to couple into temperature evolution operate on the temperature grid (Hc,Wc).
    if block_size > 1:
        _uv_c = _coarsen_many({"u": u_full, "v": v_full}, Hc, Wc, block_size)
        u_coarse = _uv_c["u"]
        v_coarse = _uv_c["v"]
    else:
        u_coarse = u_full
        v_coarse = v_full

    # Apply temperature evolution with advection and radiation.
    # Batched: these four fields share the same (Hc, Wc, block_size) coarsening and
    # are all independently optional, so one stacked pad+reshape+mean replaces four
    # separate `_coarsen` calls (see `_coarsen_many`).
    _group_c_in: dict[str, np.ndarray] = {}
    if state.humidity is not None:
        _group_c_in["humidity"] = state.humidity
    if state.snow_depth is not None:
        _group_c_in["snow_depth"] = state.snow_depth
    if state.precipitation is not None:
        _group_c_in["precipitation"] = state.precipitation
    if state.vegetation_biomass is not None:
        _group_c_in["biomass"] = state.vegetation_biomass
    _group_c_out = _coarsen_many(_group_c_in, Hc, Wc, block_size)
    humidity_coarse = _group_c_out.get("humidity")
    snow_depth_coarse = _group_c_out.get("snow_depth")
    precipitation_coarse = _group_c_out.get("precipitation")
    biomass_coarse = _group_c_out.get("biomass")

    # Downsample biomes / Köppen to coarse resolution (center-of-block sample, not average).
    _mid = block_size // 2
    if biome_new is not None:
        _bs = block_size
        _bh, _bw = Hc * _bs - H, Wc * _bs - W
        _bp = biome_new.astype(np.int32)
        if _bh > 0 or _bw > 0:
            _bp = np.pad(_bp, ((0, _bh), (0, _bw)), mode="edge")
        biome_coarse: np.ndarray | None = _bp.reshape(Hc, _bs, Wc, _bs)[:, _mid, :, _mid]
    else:
        biome_coarse = None

    if koppen_new is not None:
        _kp = koppen_new.astype(np.int32)
        _bh, _bw = Hc * block_size - H, Wc * block_size - W
        if _bh > 0 or _bw > 0:
            _kp = np.pad(_kp, ((0, _bh), (0, _bw)), mode="edge")
        koppen_coarse: np.ndarray | None = _kp.reshape(Hc, block_size, Wc, block_size)[:, _mid, :, _mid]
    else:
        koppen_coarse = None

    # Albedo-effective Köppen: cells classified as EF (ice cap, code 19) but not yet
    # mature (ice_sheet_age < threshold) are physically treated as ET (tundra, code 18)
    # for albedo purposes.  This prevents newly-cold cells from jumping straight to the
    # 0.80 ice-sheet albedo before they have accumulated enough ice to warrant it.
    # The displayed/stored koppen_type is unchanged — only the albedo computation differs.
    if koppen_coarse is not None:
        _isa_coarse = _coarsen(ice_sheet_age_new, Hc, Wc, block_size)
        _immature_ef = (koppen_coarse == 19) & (_isa_coarse < ICE_SHEET_THRESHOLD_DAYS)
        koppen_phys_coarse = koppen_coarse.copy()
        koppen_phys_coarse[_immature_ef] = 18  # treat as ET (tundra) albedo
    else:
        koppen_phys_coarse = None

    # Coarsen new fields for _evolve_temperature (Features 1, 5, 6) — batched (see above).
    _group_c2_in: dict[str, np.ndarray] = {}
    if state.cloud_cover is not None:
        _group_c2_in["cloud_cover"] = state.cloud_cover
    if state.T_deep_ocean is not None:
        _group_c2_in["T_deep"] = state.T_deep_ocean
    _group_c2_out = _coarsen_many(_group_c2_in, Hc, Wc, block_size)
    cloud_cover_coarse = _group_c2_out.get("cloud_cover")
    T_deep_coarse = _group_c2_out.get("T_deep")
    # ice_thick_prev_coarse already computed above

    # Always track components for diagnostics (minimal overhead)
    T_sst_coarse, T_air_coarse_new, cloud_c, snow_c, temp_components, T_deep_coarse_new = _evolve_temperature(
        T_prev_coarse, T_base, state.elevation, Hc, Wc, block_size, H, W,
        day_of_year=int(new_day), days=days,
        T_air_prev=T_air_coarse,
        wind_u=u_coarse, wind_v=v_coarse,
        T_base_land=T_base_land,
        ice_cover=ice_prev_coarse,
        thermal_diffusion=thermal_diffusion,
        ocean_transport_coeff=ocean_transport_coeff,
        ocean_exchange_floor=ocean_exchange_floor,
        ocean_exchange_span=ocean_exchange_span,
        ocean_exchange_coeff=ocean_exchange_coeff,
        ocean_exchange_inertia=ocean_exchange_inertia,
        epsilon_equator=eps_eq,
        epsilon_pole=eps_pole,
        ice_albedo_strength=ice_albedo_strength,
        humidity=humidity_coarse,
        track_components=track_components,
        precipitation=precipitation_coarse,
        vegetation_biomass=biomass_coarse,
        biome=biome_coarse,
        koppen_type=koppen_phys_coarse,  # albedo-effective: EF→ET for immature ice sheets
        planet_params=pp,
        elev_c=elev_c,
        snow_depth=snow_depth_coarse,
        # Pass the resolved flags (_fb), not the raw caller dict: _fb includes
        # planet-level auto-disables (e.g. ocean_transport/ice_albedo off for
        # worlds without a liquid ocean) that were previously dropped here.
        feedback_flags=_fb,
        total_days=new_total_days,
        prev_cloud_cover=cloud_cover_coarse,  # Feature 1: cloud persistence
        T_deep_ocean=T_deep_coarse,            # Feature 5: deep ocean layer
        ice_thickness=ice_thick_prev_coarse,   # Feature 6: thickness-dependent albedo
    )
    T_coarse = T_sst_coarse  # alias: T_coarse continues to mean T_sst going forward
    
    # Upsample components to full resolution if needed
    if block_size > 1 and temp_components:
        temp_components_full = {}
        to_up = {k: v for k, v in temp_components.items() if isinstance(v, np.ndarray) and v.shape == (Hc, Wc)}
        if to_up:
            up = _upsample_bilinear_many(to_up, H, W, block_size)
            temp_components_full.update(up)
        for name, field in temp_components.items():
            if name not in temp_components_full:
                # Scalar or already full resolution
                temp_components_full[name] = field
        temp_components = temp_components_full
    
    if block_size > 1:
        _up_fields: dict[str, np.ndarray] = {"T": T_coarse, "cloud": cloud_c, "T_air": T_air_coarse_new}
        if T_deep_coarse_new is not None:
            _up_fields["T_deep"] = T_deep_coarse_new
        up = _upsample_bilinear_many(_up_fields, H, W, block_size)
        T_full, cloud_full, T_air_full = up["T"], up["cloud"], up["T_air"]
        T_deep_full: np.ndarray | None = up.get("T_deep")
    else:
        T_full = T_coarse
        cloud_full = cloud_c
        T_air_full = T_air_coarse_new
        T_deep_full = T_deep_coarse_new

    # Feature 5: initialize deep ocean on first step (SST - 15K, clamped to 271-285K).
    # Use the same value for land and ocean so coarsening never produces unphysical averages.
    # The physics exchange is gated by sea_mask inside _evolve_temperature, so land values
    # never feed back into T_sst.
    if T_deep_full is None and pp.has_liquid_water_ocean and T_full is not None:
        T_deep_full = np.clip(T_full - 15.0, 271.0, 285.0).astype(np.float32, copy=False)

    
    if state.elevation is not None:
        # Run precipitation at half resolution (block_size=2) for large grids where
        # the half-resolution cell size (~0.7°) still resolves the subtropical dry belt
        # adequately. For small grids (H < 256) the subtropical band spans too few rows
        # at half resolution, so full-resolution precipitation is used instead.
        _pbs = 2 if H >= 256 else 1
        _Hcp = max(1, H // _pbs)
        _Wcp = max(1, W // _pbs)
        if _pbs > 1 and H >= 4 and W >= 4:
            _elev_p = _coarsen_elevation_cached(state.elevation, _Hcp, _Wcp, _pbs)
            # Batched (see `_coarsen_many`): T/u/v are unconditional, humidity/soil/cloud
            # are independently optional, all sharing the same (_Hcp, _Wcp, _pbs) grid.
            _group_p_in: dict[str, np.ndarray] = {"T": T_full, "u": u_full, "v": v_full}
            if state.humidity is not None:
                _group_p_in["hum"] = state.humidity
            if state.soil_moisture is not None:
                _group_p_in["soil"] = state.soil_moisture
            if state.soil_moisture_deep is not None:
                _group_p_in["soil_deep"] = state.soil_moisture_deep
            if cloud_full is not None:
                _group_p_in["cloud"] = cloud_full
            _group_p_out = _coarsen_many(_group_p_in, _Hcp, _Wcp, _pbs)
            _T_p = _group_p_out["T"]
            _u_p = _group_p_out["u"]
            _v_p = _group_p_out["v"]
            _hum_p = _group_p_out.get("hum")
            _soil_p = _group_p_out.get("soil")
            _soil_deep_p = _group_p_out.get("soil_deep")
            _cloud_p = _group_p_out.get("cloud")
            P_p, hum_p_next, soil_p_next, soil_deep_p_next = _generate_precipitation_substepped(
                _Hcp, _Wcp, _elev_p,
                temperature=_T_p, wind_u=_u_p, wind_v=_v_p,
                humidity=_hum_p, soil_moisture=_soil_p, soil_moisture_deep=_soil_deep_p,
                cloud_fraction=_cloud_p,
                day_of_year=int(new_day), dt_days=float(days),
                surface_pressure_hpa=pp.surface_pressure_pa / 100.0,
                planet_params=pp,
            )
            _up = _upsample_bilinear_many(
                {"P": P_p, "q": hum_p_next, "soil": soil_p_next, "soil_deep": soil_deep_p_next}, H, W, _pbs
            )
            P_full: np.ndarray | None = _up["P"]
            humidity_next: np.ndarray | None = _up["q"]
            soil_next: np.ndarray | None = _up["soil"]
            soil_deep_next: np.ndarray | None = _up["soil_deep"]
        else:
            P_full, humidity_next, soil_next, soil_deep_next = _generate_precipitation_substepped(
                H, W, state.elevation,
                temperature=T_full, wind_u=u_full, wind_v=v_full,
                humidity=state.humidity, soil_moisture=state.soil_moisture,
                soil_moisture_deep=state.soil_moisture_deep,
                cloud_fraction=cloud_full,
                day_of_year=int(new_day), dt_days=float(days),
                surface_pressure_hpa=pp.surface_pressure_pa / 100.0,
                planet_params=pp,
            )
    else:
        P_full = None
        humidity_next = None
        soil_next = None
        soil_deep_next = None
    # NOTE: latent cooling from precipitation is already applied inside
    # _evolve_temperature (via evaporation) and generate_precipitation.
    # Applying it again here was a double-count and has been removed.
    if T_full is not None:
        T_full = np.clip(T_full, 150.0, 330.0)
    if T_full is not None and pp.has_liquid_water_ocean:
        ice_full, delta_ice, ice_thick_full = update_sea_ice(
            T_full, state.elevation, state.ice_cover, days,
            _ice_thick_src,  # prev_thickness (already initialized above)
            freeze_temp=ice_freeze_temp,
            melt_temp=ice_melt_temp,
            freeze_rate=ice_freeze_rate,
            melt_rate=ice_melt_rate,
        )
        # Ice-ocean latent heat feedback: freezing releases heat, melting absorbs heat
        # L_f=334 kJ/kg, rho_ice=917 kg/m³, ~1m effective thickness, ~100m mixed layer
        # gives ~3K per unit ice fraction change
        if _fb.get('ice_albedo', True):
            latent_scale = 3.0  # K per unit ice fraction change
            is_ocean_full, _ = get_masks(state.elevation)
            T_full = T_full + delta_ice * latent_scale * is_ocean_full.astype(np.float32)
    else:
        ice_full = state.ice_cover  # preserve existing (e.g. dry planet polar CO2 ice)
        delta_ice = np.zeros((H, W), dtype=np.float32)
        ice_thick_full = _ice_thick_src  # no thickness evolution on dry planets

    # Feature 3: salinity evolution (after sea-ice so delta_ice is available)
    from ocean import evolve_salinity  # imported here to avoid circular-import risk
    if pp.has_liquid_water_ocean:
        _sal_prev = state.salinity
        if _sal_prev is None:
            # First-step initialization: uniform ocean salinity
            _sea_sal, _ = get_masks(state.elevation)
            _sal_prev = np.where(_sea_sal, pp.salinity_reference_psu, 0.0).astype(np.float32, copy=False)
        salinity_new = evolve_salinity(
            _sal_prev, T_full, state.elevation,
            P_full, delta_ice, dt_days=float(days), pp=pp,
        )
    else:
        salinity_new: np.ndarray | None = state.salinity

    # Snow depth evolution (degree-day accumulation / melt model)
    # Only over land; ocean has sea ice instead of snow pack.
    # Physics:
    #   Accumulation  — fraction of precipitation that falls as snow (T-dependent)
    #   Melt          — degree-day factor: 3 mm SWE per °C per day above freezing
    #   Sublimation   — 0.1 % of current pack per day (slow but persistent)
    #   Cap at 10 m SWE (realistic maximum for land ice / deep snow pack)
    _T_air_for_snow = T_air_full if T_air_full is not None else T_full
    _snow_prev = state.snow_depth if state.snow_depth is not None else np.zeros((H, W), dtype=np.float32)
    if P_full is not None and _T_air_for_snow is not None:
        _, _land_snow = get_masks(state.elevation)
        _T_air_c = _T_air_for_snow - 273.15  # °C
        # Snow fraction: 1 at ≤−3°C, 0 at ≥+2°C (linear ramp through the mixed-phase zone)
        _snow_frac = np.clip((-_T_air_c + 2.0) / 5.0, 0.0, 1.0).astype(np.float32, copy=False)
        # Snowfall in m SWE/day (P_full is mm/day liquid-water equiv; 1 mm = 0.001 m SWE)
        _snowfall = P_full * _snow_frac * 1e-3
        # Melt: 3 mm SWE per °C per day (standard degree-day factor for temperate/polar snow)
        _melt = np.clip(_T_air_c, 0.0, None).astype(np.float32, copy=False) * 3.0e-3
        # Sublimation: 0.1% of pack per day
        _sublim = _snow_prev * 0.001
        _snow_new = _snow_prev + (_snowfall - _melt - _sublim) * float(days)
        snow_depth_new = np.where(_land_snow, np.clip(_snow_new, 0.0, 10.0), 0.0).astype(np.float32, copy=False)
    else:
        snow_depth_new = _snow_prev

    # Debug logging if requested
    if debug_log:
        if T_full is not None:
            T_stats = {
                'min': float(np.min(T_full)),
                'mean': float(np.mean(T_full)),
                'max': float(np.max(T_full)),
                'p25': float(np.percentile(T_full, 25)),
                'p50': float(np.percentile(T_full, 50)),
                'p75': float(np.percentile(T_full, 75)),
            }
            LOG.info(f"[Simulation Day {new_day:.1f}] T: min={T_stats['min']:.1f}K ({T_stats['min']-273.15:.1f}°C), "
                     f"mean={T_stats['mean']:.1f}K ({T_stats['mean']-273.15:.1f}°C), "
                     f"max={T_stats['max']:.1f}K ({T_stats['max']-273.15:.1f}°C), "
                     f"median={T_stats['p50']:.1f}K ({T_stats['p50']-273.15:.1f}°C)")
            
            # Additional diagnostics: temperature by latitude bands
            H_full = T_full.shape[0]
            eq_idx = H_full // 2
            arctic_idx = int(H_full * 0.15)  # ~66°N
            tropics_idx = int(H_full * 0.4)  # ~23°N
            T_arctic = np.mean(T_full[arctic_idx, :])
            T_tropics = np.mean(T_full[tropics_idx, :])
            T_equator = np.mean(T_full[eq_idx, :])
            LOG.info(f"  By latitude: Arctic(66°N)={float(T_arctic):.1f}K ({float(T_arctic-273.15):.1f}°C), "
                     f"Tropics(23°N)={float(T_tropics):.1f}K ({float(T_tropics-273.15):.1f}°C), "
                     f"Equator={float(T_equator):.1f}K ({float(T_equator-273.15):.1f}°C)")
            
            # Temperature vs elevation analysis
            if state.elevation is not None:
                high_elev_mask = state.elevation > 0.4  # High altitude areas
                if np.any(high_elev_mask):
                    T_high_elev = T_full[high_elev_mask]
                    LOG.info(f"  High altitude (>0.4 elev): T_mean={float(np.mean(T_high_elev)):.1f}K ({float(np.mean(T_high_elev)-273.15):.1f}°C), "
                             f"T_max={float(np.max(T_high_elev)):.1f}K ({float(np.max(T_high_elev)-273.15):.1f}°C)")

    # ------------------------------------------------------
    # Carbon Cycle (Phase 3)
    # ------------------------------------------------------
    if enable_carbon_cycle:
        from carbon_cycle import (
            wetland_ch4_emissions as _wetland_ch4,
            permafrost_thaw_step as _pfc_thaw,
            ch4_oxidation_step as _ch4_oxidize,
            permafrost_init as _pfc_init,
            wildfire_dynamics as _wildfire,
            compute_biome_type as _compute_biome_type,
        )
        from masks import get_masks as _get_masks_cc

        _sea_cc, _land_cc = _get_masks_cc(state.elevation)
        _P_for_carbon = P_full if P_full is not None else np.ones_like(T_air_full) * 3.0

        # --- Slow carbon-cycle bundle: biome classification, wildfire, permafrost
        # thaw, wetland CH4. See CARBON_SLOW_UPDATE_INTERVAL_DAYS above for why.
        _cs = _CARBON_SLOW_CACHE
        _cs_key = (H, W)
        _elapsed_carbon = new_total_days - _cs["last_update_day"]
        # Implausibly large gap (>60d) means a stale cross-run cache (e.g. a
        # previous simulation in this process, or a loaded save with very
        # different total_days) rather than real elapsed simulated time —
        # treat it like a fresh start (dt=days) instead of lump-applying years
        # of accumulated flux in one call.
        _is_first_or_reset = (
            _cs["last_update_day"] <= -9000.0
            or _cs["key"] != _cs_key
            or abs(_elapsed_carbon) > 60.0
        )
        _do_carbon_slow = _is_first_or_reset or abs(_elapsed_carbon) >= CARBON_SLOW_UPDATE_INTERVAL_DAYS
        if _do_carbon_slow:
            _carbon_dt = float(days) if _is_first_or_reset else float(abs(_elapsed_carbon))
            cached_biome = _compute_biome_type(T_air_full, _P_for_carbon, _land_cc)
            _cs["biome"] = cached_biome
            _cs["key"] = _cs_key
            _cs["last_update_day"] = new_total_days
        else:
            cached_biome = _cs["biome"]
            _carbon_dt = 0.0  # unused unless _do_carbon_slow

        # Create temporary state for carbon cycle computation
        temp_state_for_carbon = PlanetState(
            day_of_year=new_day,
            total_days=new_total_days,
            elevation=state.elevation,
            temperature=T_air_full,  # vegetation NPP responds to air temperature
            wind_u=u_full,
            wind_v=v_full,
            precipitation=P_full,
            co2_atmosphere=state.co2_atmosphere,
            co2_ocean=state.co2_ocean,
            vegetation_biomass=state.vegetation_biomass,
        )

        # Evolve the per-step half of the carbon cycle: ocean CO2 exchange +
        # vegetation NPP/growth (fast-responding; stays per-step every mode).
        co2_atm_new, co2_ocean_new, biomass_new, co2_forcing_result = carbon_cycle_step(
            temp_state_for_carbon, days, biome=cached_biome
        )

        # CO2 greenhouse feedback is now applied to T_base (equilibrium temperature) above,
        # not added to final temperature here. This prevents runaway warming.

        # Initialize permafrost on first step
        _pfc = state.permafrost_carbon
        if _pfc is None and T_air_full is not None:
            _pfc = _pfc_init(state.elevation, T_air_full)

        ch4_ppb = state.ch4_atmosphere
        pfc_new = _pfc

        if _do_carbon_slow:
            # Wildfire: applies _carbon_dt days worth of fire risk in one lump
            # (moved out of carbon_cycle_step so it can share this cache).
            biomass_new, co2_from_fire = _wildfire(
                biomass_new, T_air_full, _P_for_carbon, state.soil_moisture, _carbon_dt
            )
            co2_atm_new = float(np.clip(co2_atm_new + co2_from_fire, 100.0, 10000.0))

            if _pfc is not None and T_full is not None:
                pfc_new, d_co2_pfc, d_ch4_pfc = _pfc_thaw(_pfc, T_full, snow_depth_new, _carbon_dt)
                ch4_ppb += d_ch4_pfc
                co2_atm_new += d_co2_pfc

            # Wetland emissions
            if T_full is not None:
                ch4_ppb += _wetland_ch4(T_full, state.soil_moisture, _land_cc, _carbon_dt)

        # Background natural CH4 source balancing OH oxidation at the planetary
        # baseline (see carbon_cycle.ch4_natural_source): without it CH4 decayed
        # from 1900 ppb toward zero over multi-decade runs (τ=9yr), injecting a
        # spurious ~-1 W/m² forcing drift no modeled source could offset.
        if pp.ch4_baseline_ppb > 0.0:
            from carbon_cycle import ch4_natural_source as _ch4_source
            ch4_ppb += _ch4_source(pp.ch4_baseline_ppb, float(days))

        # Atmospheric oxidation (9-yr lifetime) — cheap scalar op, stays per-step.
        ch4_atm_new = float(np.clip(_ch4_oxidize(ch4_ppb, float(days)), 100.0, 50_000.0))

        if debug_log:
            LOG.info(f"Carbon cycle: CO2={co2_atm_new:.1f} ppm, forcing={co2_forcing_result:.2f} W/m², CH4={ch4_atm_new:.0f} ppb, "
                     f"slow_update={_do_carbon_slow}")
    else:
        co2_atm_new = state.co2_atmosphere
        co2_ocean_new = state.co2_ocean
        biomass_new = state.vegetation_biomass
        ch4_atm_new = state.ch4_atmosphere
        pfc_new = state.permafrost_carbon

    new_state = PlanetState(
        day_of_year=new_day,
        total_days=new_total_days,
        elevation=state.elevation,
        temperature=T_full,         # T_sst: sea surface / land surface temperature
        air_temperature=T_air_full, # T_air: 2m air temperature
        wind_u=u_full,
        wind_v=v_full,
        precipitation=P_full,
        humidity=humidity_next,
        soil_moisture=soil_next,
        soil_moisture_deep=soil_deep_next,
        cloud_cover=cloud_full,
        snow_depth=snow_depth_new,
        ice_cover=ice_full,
        co2_atmosphere=co2_atm_new,
        co2_ocean=co2_ocean_new,
        vegetation_biomass=biomass_new,
        # Phase 1: Climate averaging and stable biomes
        climate_temp_avg=temp_avg,
        climate_precip_avg=precip_avg,
        climate_sample_days=sample_days,
        biome_type=biome_new,
        biome_last_update_day=biome_last_update,
        # Monthly statistics and Köppen classification
        monthly_temp=monthly_temp,
        monthly_precip=monthly_precip,
        monthly_sample_count=monthly_sample_count,
        koppen_type=koppen_new,
        ice_sheet_age=ice_sheet_age_new,
        # Feature 3: salinity
        salinity=salinity_new,
        # Feature 4: CH4 / permafrost
        ch4_atmosphere=ch4_atm_new,
        permafrost_carbon=pfc_new,
        # Feature 5: deep ocean
        T_deep_ocean=T_deep_full,
        # Feature 6: sea ice thickness
        ice_thickness=ice_thick_full,
        # Feature 7: jet stream dynamics
        jet_index_nh=jet_index_nh_new,
        jet_index_sh=jet_index_sh_new,
        jet_block_lon_nh=jet_block_lon_nh_new,
        jet_block_days_left_nh=jet_block_days_left_nh_new,
        jet_block_total_days_nh=jet_block_total_nh_new,
        jet_block_lon_sh=jet_block_lon_sh_new,
        jet_block_days_left_sh=jet_block_days_left_sh_new,
        jet_block_total_days_sh=jet_block_total_sh_new,
        # Feature 8: 1.5-layer atmosphere upper-level wind
        wind_u_aloft=u2_full,
        wind_v_aloft=v2_full,
        planet_params=pp,
    )

    # Return state and components (empty dict if not tracking)
    return new_state, temp_components if temp_components else {}


def simulate_multiple_steps(
    initial_state: PlanetState,
    total_days: float,
    step_days: float = 1.0,
    **kwargs,
) -> tuple[list[PlanetState], list[dict]]:
    """Simulate multiple steps, returning intermediate states.

    Args:
        initial_state: Starting state
        total_days: Total simulation time
        step_days: Time per step
        **kwargs: Passed to simulate_step

    Returns:
        List of states at each step (including initial)
    """
    states = [initial_state]
    components_list = [{}]  # Empty dict for initial state
    current = initial_state
    n_steps = int(np.ceil(total_days / step_days))
    for _ in range(n_steps):
        dt = min(step_days, total_days - (len(states) - 1) * step_days)
        if dt <= 0:
            break
        current, comps = simulate_step(current, days=dt, **kwargs)
        states.append(current)
        components_list.append(comps)
    return states, components_list


def create_initial_state(
    elevation: np.ndarray,
    day_of_year: float = 80.0,
    **kwargs,
) -> PlanetState:
    """Create initial planet state from elevation map.

    Args:
        elevation: (H, W) terrain elevation [0,1]
        day_of_year: Starting day (0-365.2422)
        **kwargs: Passed to simulate_step for initial computation

    Returns:
        Initialized state with all fields computed
    """
    # Seed atmospheric composition from the planet's parameters. Previously
    # only the optimizer's headless runner applied co2_initial_ppm, so GUI and
    # test runs silently started every planet at the PlanetState defaults
    # (Earth-2020 values) regardless of PlanetParams.
    _pp_init = kwargs.get("planet_params") or EARTH
    state = PlanetState(
        day_of_year=day_of_year,
        total_days=0.0,
        elevation=elevation,
        temperature=None,
        wind_u=None,
        wind_v=None,
        precipitation=None,
        humidity=None,
        co2_atmosphere=float(_pp_init.co2_initial_ppm),
        ch4_atmosphere=float(_pp_init.ch4_initial_ppb),
        planet_params=_pp_init,
    )
    new_state, _ = simulate_step(state, days=0.0, **kwargs)
    return new_state


def _evolve_temperature(
    T_prev: np.ndarray,
    T_base: np.ndarray,
    elevation: np.ndarray,
    Hc: int,
    Wc: int,
    block_size: int,
    H: int,
    W: int,
    day_of_year: int,
    days: float,
    *,
    T_air_prev: np.ndarray | None = None,
    wind_u: np.ndarray | None = None,
    wind_v: np.ndarray | None = None,
    land_sea_contrast: float = 0.0,
    thermal_diffusion: float = 0.04,
    T_base_land: np.ndarray | None = None,
    ice_cover: np.ndarray | None = None,
    ocean_transport_coeff: float = 0.5,
    ocean_exchange_floor: float = 0.65,   # deprecated no-op (never read downstream)
    ocean_exchange_span: float = 0.35,    # deprecated no-op (never read downstream)
    ocean_exchange_coeff: float = 0.03,
    ocean_exchange_inertia: float = 0.0,
    epsilon_equator: float = 0.72,
    epsilon_pole: float = 0.50,
    ice_albedo_strength: float = 1.0,
    humidity: np.ndarray | None = None,
    track_components: bool = False,
    precipitation: np.ndarray | None = None,
    vegetation_biomass: np.ndarray | None = None,
    biome: np.ndarray | None = None,
    koppen_type: np.ndarray | None = None,
    planet_params: PlanetParams | None = None,
    elev_c: np.ndarray | None = None,
    snow_depth: np.ndarray | None = None,
    feedback_flags: dict[str, bool] | None = None,
    total_days: float | None = None,  # monotonic sim time for the ocean-update cache
    prev_cloud_cover: np.ndarray | None = None,  # Feature 1: cloud persistence
    T_deep_ocean: np.ndarray | None = None,       # Feature 5: deep ocean layer
    ice_thickness: np.ndarray | None = None,      # Feature 6: thickness-dependent albedo
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray | None]:
    """Evolve temperature with FULL physics: Radiation, Advection, Latent Heat.

    Physics upgrades (Items 1-15):
    - Cloud-radiation feedback (Albedo + Greenhouse)
    - Snow/Ice albedo feedback
    - Latent heat of phase changes (Evap/Condensation)
    - Sensible heat flux
    - Longwave radiation emission
    - Surface heat capacity variations
    """
    # Validate expected shapes
    assert T_prev.shape == (Hc, Wc)
    
    # 1. Prepare Surface Properties
    # Downsample elevation (skip if caller already provides coarse elev)
    if elev_c is None:
        elev_pad = np.pad(elevation.astype(np.float32, copy=False), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        elev_c = elev_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    # Land/Sea Masks — elev_c is a transient coarse array; skip cache to avoid stale hits
    sea_mask, land_mask = get_masks(elev_c, use_cache=False)
    land_fraction = land_mask.astype(np.float32) # Simplified for now
    
    _pp = planet_params if planet_params is not None else EARTH

    # 2. Wind Field (Prognostic or Diagnostic)
    if wind_u is None or wind_v is None:
        u, v = generate_wind_field(
            Hc,
            Wc,
            day_of_year=day_of_year,
            block_size=1,
            elevation=elev_c,
            planet_params=_pp,
        )
    else:
        u, v = wind_u, wind_v
        
    dt_sec = days * 86400.0

    # --- Two prognostic fields ---
    # T_sst: sea surface temperature (ocean) / land surface temperature.
    #        Driven by radiative balance, ocean transport, and land-surface physics.
    #        NOT advected by wind — the atmosphere blows over it, not with it.
    # T_air: 2-metre air temperature.
    #        Advected by wind, diffused, and coupled to T_sst through surface exchange.
    T_sst = T_prev.copy().astype(np.float32, copy=False)
    T_air = (T_air_prev.copy() if T_air_prev is not None else T_prev.copy()).astype(np.float32, copy=False)

    # === PASS 1: T_air dynamics — advection, diffusion, surface exchange ===
    lat_1d = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    cos_lat = np.cos(lat_1d).clip(0.05, 1.0)
    dx_lat = (2.0 * np.pi * _pp.radius_m * cos_lat / Wc).astype(np.float32, copy=False)
    dy = float(np.pi * _pp.radius_m / Hc)
    dx_2d = dx_lat[:, None]
    u_cfl = np.clip(np.abs(u) * dt_sec / dx_2d, 0.0, 0.5).astype(np.float32, copy=False)
    v_cfl = np.clip(np.abs(v) * dt_sec / dy, 0.0, 0.5).astype(np.float32, copy=False)

    T_before_advection = T_air.copy()  # for component tracking

    # Advect T_air with wind (atmosphere moves horizontally)
    if NUMBA_AVAILABLE:
        T_air = _advect_temperature_x_numba(T_air, u.astype(np.float32, copy=False), u_cfl)
        T_air = _advect_temperature_y_numba(T_air, v.astype(np.float32, copy=False), v_cfl)
    else:
        T_east = np.roll(T_air, -1, axis=1)
        T_west = np.roll(T_air, 1, axis=1)
        T_x = np.where(u >= 0, T_west, T_east)
        T_air = T_air + u_cfl * (T_x - T_air)
        T_north = np.roll(T_air, -1, axis=0)
        T_south = np.roll(T_air, 1, axis=0)
        T_y = np.where(v >= 0, T_south, T_north)
        T_air = T_air + v_cfl * (T_y - T_air)

    # Diffuse T_air (atmospheric mixing).
    # Explicit Laplacian diffusion is only stable for r = coeff*1.2*days below
    # ~0.5 per application (same forward-difference CFL bound handled for the
    # ocean eddy flux below); production substeps (days ≤ 7) are fine at the
    # 0.04 default, but direct calls with large `days` need sub-stepping.
    T_before_diffusion = T_air.copy() if track_components else None
    _r_diff = float(thermal_diffusion) * 1.2 * float(days)
    _n_diff_sub = max(1, int(np.ceil(_r_diff / 0.4)))
    _days_diff_sub = float(days) / _n_diff_sub
    for _ in range(_n_diff_sub):
        if NUMBA_AVAILABLE:
            T_air = _apply_diffusion_numba(T_air, float(thermal_diffusion), _days_diff_sub, iterations=2)
        else:
            for _ in range(2):
                T_pad = np.pad(T_air, ((1, 1), (0, 0)), mode="edge")
                c = T_pad[1:-1, :]
                n = T_pad[0:-2, :]
                s = T_pad[2:, :]
                e = np.roll(c, -1, axis=1)
                w = np.roll(c, 1, axis=1)
                T_lap = n + s + e + w - 4.0 * c
                T_air = T_air + thermal_diffusion * 1.2 * np.clip(T_lap, -30.0, 30.0) * _days_diff_sub

    # T_air relaxes toward surface temperature (T_sst).
    # Over ocean: ~4-day time constant (efficient sensible heat flux at ocean surface).
    # Over land: ~2-day time constant (land surface heats/cools overlying air quickly).
    # Fraction capped at 0.5 so relaxation is stable for any dt (no overshoot).
    k_air_surface = np.where(sea_mask, 0.25, 0.50).astype(np.float32, copy=False)
    _air_frac = np.minimum(k_air_surface * float(days), 0.5).astype(np.float32, copy=False)
    T_air = (T_air + _air_frac * (T_sst - T_air)).astype(np.float32, copy=False)
        
    # --- Radiative Balance (Physics Item 1, 2, 10, 12) ---
    # Incoming Solar (S_in) - Albedo (A)
    # A depends on: Land/Ocean, Snow/Ice, Cloud
    
    # Approx Latitude
    lat = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    lat_2d = np.repeat(lat[:, None], Wc, axis=1)
    
    # === PASS 2: T_sst dynamics — radiation, relaxation, ocean transport ===

    _fb_t = feedback_flags or {}
    # Snow cover — use physically-tracked snow depth when available.
    # Full cover at ≥0.1 m SWE (≈0.3 m fresh snow); falls off linearly below.
    # Fallback to temperature-derived estimate when snow_depth not yet tracked.
    if snow_depth is not None:
        snow_cover = np.clip(snow_depth / 0.1, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        snow_cover = np.clip((273.15 - T_sst) / 10.0, 0.0, 1.0)
    if not _fb_t.get('snow_albedo', True):
        snow_cover = np.zeros_like(T_sst, dtype=np.float32)
    sea_ice = np.zeros_like(T_sst, dtype=np.float32) if ice_cover is None else np.clip(ice_cover.astype(np.float32, copy=False), 0.0, 1.0)
    sea_ice = np.where(sea_mask, sea_ice, 0.0)
    if ice_albedo_strength != 1.0:
        sea_ice = np.clip(sea_ice * float(ice_albedo_strength), 0.0, 1.0)
    if not _fb_t.get('ice_albedo', True):
        sea_ice = np.zeros_like(T_sst, dtype=np.float32)

    # Cloud cover — humidity lives in the atmosphere, so use T_air for Clausius-Clapeyron
    Tc = np.clip(T_air - 273.15, -60.0, 60.0)
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    qsat = np.clip(0.622 * es / (_pp.surface_pressure_pa / 100.0), 1e-6, 0.035).astype(np.float32, copy=False)
    if humidity is not None:
        q = np.clip(humidity.astype(np.float32, copy=False), 0.0, qsat)
    else:
        temp_norm = np.clip((T_air - 255.0) / 45.0, 0.0, 1.0)
        base_q = np.where(sea_mask, 0.012, 0.008).astype(np.float32, copy=False)
        q = base_q * (0.5 + 0.7 * temp_norm)
    rh = np.clip(q / qsat, 0.0, 1.5)
    div = 0.5 * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) + np.gradient(v, axis=0)
    ascent = np.clip(-div, 0.0, None)
    subsidence = np.clip(div, 0.0, None)
    ascent = ascent / (np.mean(ascent) + 1e-6)
    subsidence = subsidence / (np.mean(subsidence) + 1e-6)
    gx = 0.5 * (np.roll(elev_c, -1, axis=1) - np.roll(elev_c, 1, axis=1))
    gy = np.gradient(elev_c, axis=0)
    orog = np.clip(gx * u + gy * v, 0.0, None)
    orog = orog / (np.mean(orog) + 1e-6)
    rh_core = np.clip((rh - 0.65) * 2.0, 0.0, 1.0)
    ascent_term = np.clip(0.6 + 0.6 * ascent, 0.0, 1.4)
    cloud_fraction = rh_core * ascent_term
    cloud_fraction = np.clip(cloud_fraction + 0.25 * rh_core * orog, 0.0, 1.0)
    cloud_fraction = np.clip(cloud_fraction * (1.0 - 0.6 * np.clip(subsidence, 0.0, 1.0)), 0.0, 1.0)

    # Feature 1: cloud temporal persistence (~3-day memory).
    # Blend freshly-diagnosed cloud_fraction toward the previous-step value so
    # clouds don't jump discontinuously between steps.
    if prev_cloud_cover is not None and _fb_t.get('cloud_feedback', True):
        tau_cloud_days = 3.0
        alpha = float(np.clip(days / tau_cloud_days, 0.0, 0.5))
        cloud_fraction = (
            alpha * cloud_fraction + (1.0 - alpha) * prev_cloud_cover.astype(np.float32, copy=False)
        )
        cloud_fraction = cloud_fraction.astype(np.float32, copy=False)

    # Cloud <-> precipitation feedback: heavy rain rains a cloud out, so it shouldn't
    # persist at full cover into the next step. `precipitation` is last step's
    # coarsened rain rate [mm/day]; it was previously accepted here but unused,
    # leaving cloud_fraction and precipitation diagnosed independently even though
    # they share the same RH/ascent drivers (a gap noted in PLAN.md). 20 mm/day is a
    # heavy rain rate; deplete_frac saturates at 30% so persistent stratiform cloud
    # sheets don't get wiped out by their own drizzle (kept gentle deliberately —
    # this is a secondary coupling, not the primary cloud-cover driver).
    if precipitation is not None and _fb_t.get('cloud_feedback', True):
        rain_deplete = np.clip(precipitation.astype(np.float32, copy=False) / 20.0, 0.0, 0.30)
        cloud_fraction = np.clip(cloud_fraction * (1.0 - rain_deplete), 0.0, 1.0).astype(np.float32, copy=False)

    # Albedo (with vegetation feedback - Phase 4)
    # Ocean: 0.06, Sea Ice: 0.75, Snow: 0.8, Cloud: 0.5
    # Land albedo now depends on vegetation/biome type

    # Compute biome-based vegetation albedo (if biomass field exists)
    # Phase 1 improvement: Use stable biomes from long-term climate averages (not daily weather)
    # Köppen classification provides more detailed albedo values
    # Feature 6: thickness-dependent ocean albedo.
    # At h≥0.5m (thick ice): same as old formula — no regression.
    # At h<0.5m (thin/new ice): lower albedo prevents summer ice-albedo runaway.
    # alpha_ice(h) = 0.06 + 0.59 * min(h / 0.5, 1.0)  [0.06 open water → 0.65 thick ice]
    if ice_thickness is not None:
        _alpha_ice = (0.06 + 0.59 * np.minimum(ice_thickness / 0.5, 1.0)).astype(np.float32, copy=False)
        _alpha_sea = ((1.0 - sea_ice) * 0.06 + sea_ice * _alpha_ice).astype(np.float32, copy=False)
    else:
        _alpha_sea = (0.06 * (1.0 - sea_ice) + 0.65 * sea_ice).astype(np.float32, copy=False)

    if vegetation_biomass is not None and biome is not None and _fb_t.get('vegetation_albedo', True):
        albedo_veg = vegetation_albedo(biome, base_land_albedo=0.2, koppen_type=koppen_type)
        albedo_sfc = np.where(sea_mask, _alpha_sea, albedo_veg)
    else:
        albedo_sfc = np.where(sea_mask, _alpha_sea, 0.2)

    # Snow albedo overrides vegetation (snow is brighter)
    snow_cover_land = snow_cover * land_mask.astype(np.float32)
    albedo_sfc = np.where(land_mask, albedo_sfc * (1.0 - snow_cover_land) + 0.8 * snow_cover_land, albedo_sfc)

    # Total albedo including clouds
    albedo_total = albedo_sfc * (1 - cloud_fraction) + 0.5 * cloud_fraction
    
    # Insolation Q (Daily mean) - use proper astronomical calculation
    _pp = planet_params if planet_params is not None else EARTH
    obliq = _pp.obliquity_rad
    gamma = 2.0 * np.pi * (day_of_year - 80.0) / _pp.orbital_period_days
    decl = np.arcsin(np.sin(obliq) * np.sin(gamma))
    S0 = _pp.effective_solar_constant(day_of_year)

    # Clamp to avoid domain errors in polar regions
    lat_safe = np.clip(lat_2d, -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
    cos_h = np.clip(-np.tan(lat_safe) * np.tan(decl), -1.0, 1.0)
    h = np.arccos(cos_h) # hour angle radians (0 to pi)
    h = np.where(cos_h <= -1.0, np.pi, h)  # 24h daylight
    h = np.where(cos_h >= 1.0, 0.0, h)     # polar night

    Q = S0 * (1.0/np.pi) * (h * np.sin(lat_safe)*np.sin(decl) + np.cos(lat_safe)*np.cos(decl)*np.sin(h))
    Q = np.maximum(0.0, Q)
    
    S_absorbed = np.maximum(0.0, Q * (1.0 - albedo_total) + _pp.aerosol_forcing_w_m2)
    
    # Outgoing Longwave (L_out) = sigma * T^4 * epsilon
    # Greenhouse effect reduces OLR - use latitude-dependent epsilon (like temperature.py)
    abs_lat_deg = np.rad2deg(np.abs(lat_2d))
    epsilon_equator = float(epsilon_equator)  # Increased from 0.75 to match temperature.py and warm global mean
    epsilon_pole = float(epsilon_pole)     # Increased from 0.50 to reduce polar extremes
    lat_factor = np.cos(np.deg2rad(abs_lat_deg))  # 1.0 at equator, 0.0 at poles
    epsilon = epsilon_pole + (epsilon_equator - epsilon_pole) * lat_factor
    
    # Feature 1: cloud greenhouse on OLR.
    # High clouds (cold tops) trap outgoing longwave; low/warm clouds do not.
    # T_air proxy: colder air column → higher cloud tops → stronger LW trapping.
    if _fb_t.get('cloud_feedback', True) and _pp.cloud_greenhouse_factor > 0.0:
        cloud_high_weight = np.clip((265.0 - T_air) / 20.0, 0.0, 1.0).astype(np.float32, copy=False)
        epsilon_cloud_ghg = _pp.cloud_greenhouse_factor * cloud_fraction * cloud_high_weight
        epsilon = np.clip(epsilon - epsilon_cloud_ghg, 0.30, 0.95).astype(np.float32, copy=False)

    # Feature 2: water vapour greenhouse on OLR (applied after cloud term).
    if _fb_t.get('water_vapor_feedback', True) and _pp.wv_greenhouse_factor > 0.0 and humidity is not None:
        # rh already computed above; higher RH → more WV → lower effective epsilon
        wv_reduction = _pp.wv_greenhouse_factor * np.clip(rh - 0.5, 0.0, 1.0).astype(np.float32, copy=False)
        epsilon = np.clip(epsilon - wv_reduction, 0.30, 0.95).astype(np.float32, copy=False)

    sigma = 5.67e-8
    # Longwave emission is from the surface (T_sst drives outgoing radiation)
    L_out = epsilon * sigma * (T_sst ** 4)

    R_net = S_absorbed - L_out  # W/m²

    # Radiative equilibrium temperature for the surface
    T_eq_rad = (S_absorbed / (epsilon * sigma + 1e-9)) ** 0.25
    T_eq_rad = np.clip(T_eq_rad, 150.0, 350.0)
    
    # CRITICAL FIX: Blend radiation equilibrium with base temperature
    # T_base comes from temperature_kelvin_for_lat which has proper polar cooling physics
    # We should trust it more, especially at poles where radiation-only calculation fails
    # Use a mostly base-driven blend, but let radiation pull more strongly than before
    # so broad imposed warmth in T_base does not dominate the equilibrium.
    T_eq = 0.90 * T_base + 0.10 * T_eq_rad

    # --- Orographic cooling (lapse rate) ---
    # Previously missing: high terrain never cooled as a function of altitude.
    # Apply to equilibrium temperature so radiation relaxes toward a colder state aloft.
    lapse_rate = 6.5  # K/km
    alt_km = elevation_to_alt_km(elev_c)
    T_eq = T_eq - lapse_rate * alt_km

    # --- Ocean Temperature Bounds (SST cap + freeze-point floor) ---
    # These bounds parameterize Earth's liquid-water ocean physics; suppress for dry planets.
    if _pp.has_liquid_water_ocean:
        # Upper cap (SST max ~29°C = 302K): evaporative cooling prevents ocean from exceeding
        # this in practice. The radiative equilibrium formula ignores latent heat and would
        # otherwise push subtropical SSTs to 43-48°C.  Earth's tropical zonal-mean SST is
        # 28-29°C; 302K matches this while preventing runaway tropical heating.
        T_eq = np.where(sea_mask, np.minimum(T_eq, 302.0), T_eq)
        # Latitude-dependent T_eq ocean floor — prevents ice-albedo runaway at mid-latitudes
        # while still allowing polar sea ice:
        #
        # At mid-latitudes (|lat| ≤ 60°): floor = 271K (above ice_freeze_temp=269.9K).
        #   Without this, ice forming at 55°N (albedo→0.65) drives T_eq_rad→140K, so
        #   T_eq = 0.9×278.75 + 0.1×140 = 265.9K < freeze → ice-albedo runaway to 52°N.
        #   With floor=271K the equilibrium always pulls T_sst above freezing → ice melts. ✓
        #
        # At high latitudes (|lat| ≥ 75°): floor = 266K (below freeze_temp).
        #   Allows genuine Arctic/Antarctic sea ice to form and persist. ✓
        #
        # Linear ramp between 60° and 75° so there is no sharp boundary.
        _abs_lat_floor = np.abs(np.rad2deg(lat_2d))
        _ramp = np.clip((_abs_lat_floor - 60.0) / 15.0, 0.0, 1.0)   # 0 at 60°, 1 at 75°+
        T_eq_floor = (271.0 * (1.0 - _ramp) + 266.0 * _ramp).astype(np.float32, copy=False)
        T_eq = np.where(sea_mask, np.maximum(T_eq, T_eq_floor), T_eq)

    # Relaxation rate k (1/days) based on mixed-layer depth
    # Real oceans have latitude-dependent mixed layer depth:
    #   Tropics: ~30-50m (thin thermocline, trade winds)
    #   Mid-latitudes: ~50-150m (seasonal deepening)
    #   High latitudes: ~200-500m (deep convective mixing in winter)
    # Deeper mixed layers = more thermal inertia = slower response to forcing
    abs_lat_1d = np.abs(np.rad2deg(lat))  # lat computed at line 948
    abs_lat_2d_relax = np.repeat(abs_lat_1d[:, None], Wc, axis=1)
    mld = 30.0 + 170.0 * (abs_lat_2d_relax / 90.0) ** 1.5  # 30m tropical, ~200m polar
    # Seasonal polar MLD reduction: Arctic/Antarctic meltwater halocline creates a
    # shallow warm layer (~20-30m) in summer, allowing rapid surface warming → ice melt.
    # Without this, the ~186m polar MLD gives a 93-day thermal time constant — too slow
    # for summer T to reach ice_melt_temp=260K during the 90-day Arctic summer.
    _summer_solstice_day = (172.0 / 365.2422) * float(_pp.orbital_period_days)
    _gamma_mld = 2.0 * np.pi * (float(day_of_year) - _summer_solstice_day) / float(_pp.orbital_period_days)
    _nh_summer = float(0.5 * (1.0 + np.cos(_gamma_mld)))   # 1.0 at NH summer, 0 at NH winter
    _sh_summer = float(0.5 * (1.0 - np.cos(_gamma_mld)))   # 1.0 at SH summer (day ~355)
    _hemi_summer = np.where(lat_2d >= 0, _nh_summer, _sh_summer)  # (Hc, Wc)
    _polar_ramp = np.clip((abs_lat_2d_relax - 55.0) / 30.0, 0.0, 1.0)  # 0 at 55°, 1 at 85°+
    # Up to 50% MLD reduction at poles in polar summer → ~94m → 47-day time constant
    # 80% was too aggressive: T85N reached +15°C in summer (unrealistic).
    # 50% allows meaningful summer warming (3 time constants over 90-day Arctic summer)
    # without overshooting. Target: summer T85N near 0°C, not +15°C.
    mld = mld * (1.0 - 0.50 * _polar_ramp * _hemi_summer)
    k_ocean = np.clip(1.0 / (mld * 0.5), 0.005, 0.07)  # 14-200 day time constants

    # Ice insulates T_sst against cooling (ice-ocean decoupling in winter) but not warming
    cooling_direction = T_eq < T_sst
    ice_insulation = np.where(
        cooling_direction,
        1.0 - 0.7 * sea_ice,
        1.0,
    )
    k_relax = np.where(sea_mask, k_ocean * ice_insulation, 0.1)

    # Relax T_sst toward radiative+transport equilibrium.
    # Fraction capped at 0.5 for unconditional stability at any dt (large-step modes
    # like MONTHLY/ANNUAL use dt=6-7 days where k_relax*dt can exceed 1 without this).
    _sst_frac = np.minimum(k_relax * float(days), 0.5).astype(np.float32, copy=False)
    T_sst = (T_sst + _sst_frac * (T_eq - T_sst)).astype(np.float32, copy=False)
    
    # --- Evaporation: bulk aerodynamic formula, applied to T_sst ---
    # Ocean evaporation depends on SST (surface saturation) and near-surface air humidity.
    if wind_u is not None and wind_v is not None and humidity is not None:
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        # Saturation humidity at SST (the surface provides moisture to the air)
        T_c_sst = np.clip(T_sst - 273.15, -60.0, 60.0)
        es_sst = 6.112 * np.exp(17.67 * T_c_sst / (T_c_sst + 243.5))
        qsat_sst = np.clip(0.622 * es_sst / (_pp.surface_pressure_pa / 100.0), 1e-6, 0.035)
        deficit = np.maximum(0.0, qsat_sst - humidity)
        C_D = np.where(sea_mask, 1.5e-3, 0.5e-3)
        E = C_D * wind_speed * deficit * 1000.0
        E = np.clip(E, 0.0, 20.0)
        evap_cooling = E * 2.5 * float(days)
        # Enhanced evaporative cooling for hot SSTs
        hot_ocean_excess = np.where(sea_mask & (T_sst > 303.0), T_sst - 303.0, 0.0)
        evap_cooling = evap_cooling + 0.3 * hot_ocean_excess * float(days)
        T_sst = (T_sst - evap_cooling).astype(np.float32, copy=False)
    else:
        base_evap = np.where(sea_mask, 0.01 * (T_sst - 270.0), 0.0)
        hot_evap = np.where((T_sst > 303.0) & sea_mask, 0.3 * (T_sst - 303.0), 0.0)
        evap_cooling = np.maximum(0.0, base_evap + hot_evap)
        T_sst = (T_sst - evap_cooling * float(days)).astype(np.float32, copy=False)
    
    # --- Land surface blend toward seasonal baseline ---
    if T_base_land is not None:
        T_base_land = T_base_land - lapse_rate * alt_km
        land_blend = np.where(land_mask, 0.2, 0.0)
        T_sst = ((1.0 - land_blend) * T_sst + land_blend * T_base_land).astype(np.float32, copy=False)

    # --- Air-sea sensible heat exchange (secondary coupling of T_sst to T_air) ---
    # k_airsea = 0.001/day (~1000-day τ at 50m MLD).
    #
    # Why so weak? With k_airsea=0.004 and T_air=-20°C at 55°N in winter, the
    # steady-state T_sst is:
    #   T_sst_ss = (k_relax*T_eq + k_airsea*T_air) / (k_relax + k_airsea)
    #            = (0.024*271 + 0.004*253) / 0.028 = 269.1K < ice_freeze_temp
    # The atmosphere-ocean coupling OVERRIDES the T_eq floor, pushing T_sst below
    # freezing at 55°N and triggering ice-albedo runaway.
    # With k_airsea=0.001:
    #   T_sst_ss = (0.024*271 + 0.001*253) / 0.025 = 270.3K > freeze_temp ✓
    # The ocean's primary thermal driver is its radiative balance (T_eq), not
    # atmospheric temperature. Sensible heat flux is handled mainly by evaporation.
    k_airsea = _pp.k_airsea
    T_sst = (T_sst + np.where(sea_mask, k_airsea * (T_air - T_sst) * float(days), 0.0)).astype(np.float32, copy=False)

    # --- Feature 5: Deep ocean heat uptake ---
    # The abyssal ocean (~3700m) stores 15–30× more heat than the mixed layer.
    # Exchange rate k_deep ≈ 1/(30yr) delays surface warming to realistic TCR.
    T_deep_out: np.ndarray | None = T_deep_ocean
    if T_deep_ocean is not None and _pp.has_liquid_water_ocean:
        k_deep = float(_pp.deep_ocean_exchange_rate)  # ~9.1e-5 /day
        dT_to_deep = k_deep * (T_sst - T_deep_ocean) * float(days)
        dT_to_deep = np.clip(dT_to_deep, -0.5, 0.5).astype(np.float32, copy=False)
        ocean_f = sea_mask.astype(np.float32)
        T_sst = (T_sst - dT_to_deep * ocean_f).astype(np.float32, copy=False)
        T_deep_out = (T_deep_ocean + dT_to_deep * ocean_f).astype(np.float32, copy=False)

    # --- Feature 7: Meridional eddy heat flux ---
    # Baroclinic eddies and storm tracks transport heat poleward proportional
    # to the meridional temperature gradient.  Parameterised as Laplacian
    # diffusion in the meridional direction only, weighted to 20–70° latitudes.
    _eddy_k = float(_pp.eddy_heat_flux_coeff)
    if _eddy_k > 0.0 and _fb_t.get('eddy_heat_flux', True):
        _abs_lat_1d = np.abs(np.rad2deg(lat_1d))
        _eddy_lat = np.clip(1.0 - ((_abs_lat_1d - 45.0) / 25.0) ** 2, 0.0, 1.0).astype(np.float32, copy=False)
        # Explicit-Euler Laplacian diffusion is only stable for r = eddy_k * dt_sub
        # below ~0.5 (standard forward-difference diffusion CFL bound); beyond that
        # a single big step overshoots and amplifies grid-scale noise instead of
        # smoothing the gradient -- the same large-dt failure mode fixed elsewhere
        # via sub-stepping (atmosphere.py's 8-substep wind integration,
        # _generate_precipitation_substepped). At the default coeff (0.006) this
        # was already stable even at MONTHLY dt=30 (r=0.18), which is why it went
        # unnoticed; test_eddy_heat_flux.py's coeff=0.05 stress-test (used to get
        # a detectable 2-year signal) pushes r to 1.5 at dt=30 and was the actual
        # cause of test_eddy_flux_reduces_gradient's small negative delta -- not a
        # genuine physics conflict with ocean_transport, despite ocean_transport
        # amplifying the resulting grid-scale noise into a measurable signal.
        _eddy_r_limit = 0.4
        _n_eddy_sub = max(1, int(np.ceil(_eddy_k * float(days) / _eddy_r_limit)))
        _dt_eddy_sub = float(days) / _n_eddy_sub
        for _ in range(_n_eddy_sub):
            _T_lap_y = np.zeros_like(T_sst)
            _T_lap_y[1:-1, :] = T_sst[:-2, :] - 2.0 * T_sst[1:-1, :] + T_sst[2:, :]
            _T_lap_y[0, :]     = T_sst[1, :]  - T_sst[0, :]
            _T_lap_y[-1, :]    = T_sst[-2, :] - T_sst[-1, :]
            _T_lap_y = np.clip(_T_lap_y, -20.0, 20.0).astype(np.float32, copy=False)
            T_sst = (T_sst + _eddy_k * _T_lap_y * _eddy_lat[:, None] * _dt_eddy_sub).astype(np.float32, copy=False)

    # --- Ocean Transport ---
    # NOTE (2026-07-03): this block was originally written as a 30-day cache
    # ("ocean decorrelation time"), but the cache key included round(day_of_year)
    # so it never actually hit — the transport recomputed every step for the
    # entire calibrated life of the model. Honoring the 30-day reuse turned out
    # to change climate measurably (a stale ΔT applied for 30 days weakens the
    # seasonal response; the high-obliquity seasonal-range gate fails at 1.07x
    # vs its 1.1x bar) while the measured saving is only ~1 ms/step at
    # production resolution (Numba-free NumPy path). Per-step recompute is
    # therefore the intended, calibrated behavior; the "cache" is kept solely
    # as a state carrier for feedback-flag zeroing and mode bookkeeping.
    OCEAN_UPDATE_INTERVAL_DAYS = 0.0  # recompute every step (see NOTE above)
    _oc = _OCEAN_ADJ_CACHE
    _oc_key = (Hc, Wc, round(float(days), 3))
    _t_now = float(total_days) if total_days is not None else float(day_of_year)
    days_since_ocean = _t_now - float(_oc.get("last_update_day", -9999.0))
    if not _fb_t.get('ocean_transport', True):
        T_ocean_adj = np.zeros((Hc, Wc), dtype=np.float32)
        _oc["adj"] = T_ocean_adj
        _oc["key"] = _oc_key
        _oc["last_update_day"] = _t_now
    elif (
        _oc.get("adj") is None
        or _oc.get("key") != _oc_key
        or days_since_ocean < 0.0  # time went backwards → new run/loaded save
        or days_since_ocean >= OCEAN_UPDATE_INTERVAL_DAYS
    ):
        T_ocean_adj = calculate_ocean_heat_transport(
            T_sst, elev_c, Hc, Wc, day_of_year, days,
            transport_coefficient=float(ocean_transport_coeff),
            exchange_coefficient=float(ocean_exchange_coeff),
            exchange_inertia=float(ocean_exchange_inertia),
            prev_T=T_prev,
            ice_cover=sea_ice,
            T_equilibrium=T_eq,
        )
        T_ocean_adj = np.clip(T_ocean_adj, -10.0, 10.0)

        # Ekman wind-driven advection: shifts surface water 90° from wind (Coriolis).
        # The increment is scaled to ONE `days`-long step (like the transport
        # term above) because the cached T_ocean_adj is re-applied every step
        # until the next 30-day refresh. The old code scaled it to the full
        # 30-day window AND (with the broken cache) re-applied it every day —
        # a ~30x amplification of the intended Ekman heat shift.
        if _pp.has_liquid_water_ocean and _pp.ekman_strength > 0.0:
            u_ek, v_ek = compute_ekman_transport(
                u, v, elev_c,
                ekman_coefficient=0.03 * float(_pp.ekman_strength),
                rotation_direction=float(getattr(_pp, "rotation_direction", 1.0)),
            )
            # Upwind advection of T_sst by Ekman currents over one step.
            # Zonal grid spacing shrinks with cos(lat) on an equirectangular
            # grid; using the equatorial dx everywhere under-shifted zonal
            # Ekman advection at mid/high latitudes.
            dy_m = (np.pi / Hc) * float(_pp.radius_m)
            dx_m = (2.0 * np.pi / Wc) * float(_pp.radius_m) * cos_lat[:, None]
            dt_ek = float(days)
            shift_x = np.clip(u_ek * dt_ek * 86400.0 / dx_m, -0.5, 0.5)
            shift_y = np.clip(v_ek * dt_ek * 86400.0 / dy_m, -0.5, 0.5)
            T_ek = T_sst
            # Simple upwind: dT ≈ -u·∂T/∂x − v·∂T/∂y (finite difference)
            dT_dx = 0.5 * (np.roll(T_ek, -1, axis=1) - np.roll(T_ek, 1, axis=1))  # central diff, periodic x
            dT_dy = np.zeros_like(T_ek)
            dT_dy[1:-1, :] = 0.5 * (T_ek[:-2, :] - T_ek[2:, :])  # northward derivative (row 0 = north pole)
            ekman_adj = np.clip(-(shift_x * dT_dx + shift_y * dT_dy), -1.5, 1.5)
            _ocean_mask, _ = get_masks(elev_c)
            T_ocean_adj = T_ocean_adj + ekman_adj * _ocean_mask.astype(np.float32)

        _oc["adj"] = T_ocean_adj
        _oc["key"] = _oc_key
        _oc["last_update_day"] = _t_now
    else:
        T_ocean_adj = _oc["adj"]
    T_sst = (T_sst + T_ocean_adj).astype(np.float32, copy=False)

    # --- Hadley/Subsidence parameterization (applied to T_sst surface) ---
    lat_deg = np.rad2deg(np.abs(lat_2d))
    subsidence = 0.10 * np.exp(-((lat_deg - 30.0)/10.0)**2) * float(days)
    T_sst = (T_sst + subsidence).astype(np.float32, copy=False)

    # --- Final clamping ---
    # T_sst: ocean surface / land surface (200K–323K)
    T_sst = np.clip(T_sst, 200.0, 323.0)
    # T_air: free air — allow slightly wider range (180K–340K)
    T_air = np.clip(T_air, 180.0, 340.0)

    # Track component contributions
    components = {}
    if track_components:
        components['advection'] = T_air - T_before_advection  # type: ignore[operator]
        components['diffusion'] = T_air - T_before_diffusion  # type: ignore[operator]
        components['radiation'] = k_relax * (T_eq - T_sst) * float(days)
        components['evaporation'] = -evap_cooling
        components['ocean_transport'] = T_ocean_adj
        components['subsidence'] = subsidence
        components['equilibrium_temp'] = T_eq
        components['net_radiation'] = R_net
        components['S_absorbed'] = S_absorbed
        components['L_out'] = L_out
        def _summ(field: np.ndarray) -> dict:
            return {"mean": float(np.mean(field)), "min": float(np.min(field)), "max": float(np.max(field))}
        components["toa"] = {
            "S_absorbed": _summ(S_absorbed),
            "L_out": _summ(L_out),
            "R_net": _summ(R_net),
            "albedo_mean": float(np.mean(albedo_total)),
            "cloud_mean": float(np.mean(cloud_fraction)),
            "epsilon_mean": float(np.mean(epsilon)),
        }

    return T_sst.astype(np.float32), T_air.astype(np.float32), cloud_fraction.astype(np.float32), snow_cover.astype(np.float32), components, T_deep_out


# ============================================================================
# State Serialization Functions
# Enable saving and loading simulation states for experiments
# ============================================================================

def save_state(state: PlanetState, filepath: str | Path) -> None:
    """Save PlanetState to disk using pickle.

    Args:
        state: PlanetState to save
        filepath: Path to save file (will create parent directories if needed)

    Example:
        >>> save_state(current_state, "saves/state_day365.pkl")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = filepath.stat().st_size / 1e6
    print(f"State saved to {filepath} ({file_size_mb:.1f} MB)")


def load_state(filepath: str | Path) -> PlanetState:
    """Load PlanetState from disk.

    Args:
        filepath: Path to saved state file

    Returns:
        Loaded PlanetState

    Example:
        >>> state = load_state("saves/state_day365.pkl")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"State file not found: {filepath}")

    with open(filepath, 'rb') as f:
        state = pickle.load(f)

    print(f"State loaded from {filepath} (day {state.total_days:.1f})")
    return state


def auto_save(state: PlanetState, save_dir: str | Path = "saves",
              every_n_days: float = 365) -> None:
    """Automatically save state at regular intervals.

    Args:
        state: Current PlanetState
        save_dir: Directory to save states (default: "saves")
        every_n_days: Save frequency in simulation days (default: 365)

    Example:
        >>> # In simulation loop
        >>> auto_save(state, every_n_days=100)  # Save every 100 days
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    day_num = int(state.total_days)
    if day_num % int(every_n_days) == 0 and day_num > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"state_day{day_num:06d}_{timestamp}.pkl"
        save_state(state, save_dir / filename)


