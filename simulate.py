"""Time simulation for planet conditions.

Advances atmospheric systems (temperature, wind, precipitation) forward in time
with configurable time scales. Default unit is one day.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple
from pathlib import Path
from datetime import datetime
import pickle
from atmosphere import (
    generate_wind_field, generate_precipitation,
    evolve_wind, _upsample_bilinear_many,
)
from temperature import temperature_kelvin_for_lat, elevation_to_alt_km
from ocean import calculate_ocean_heat_transport, update_sea_ice
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
    from numba import jit, prange
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

# Cache for ocean heat transport adjustment.
# Ocean dynamics are slow (decorrelation time ~30 days), so we recompute
# only once per ocean_update_interval_days and reuse the cached ΔT array.
_OCEAN_ADJ_CACHE: dict = {"adj": None, "last_update_day": -9999.0}


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
    # Baroclinic eddy / thermal-wind proxy. The previous default (3e7) tends to produce
    # unrealistically strong, planet-wide surface jets. Keep conservative by default.
    # Lower default to avoid razor-thin zonal jets at the surface.
    wind_baroclinic_jet_amp: float = 1.0e6,
    wind_baroclinic_mix: float = 2.0,
    # Increased to 3.0 days to prevent oscillations from over-relaxation
    wind_cell_relax_days: float = 3.0,
    ocean_transport_coeff: float = 0.3,
    ocean_exchange_floor: float = 0.65,
    ocean_exchange_span: float = 0.35,
    ocean_exchange_coeff: float = 0.08,
    ocean_exchange_inertia: float = 0.35,
    epsilon_equator: float | None = None,
    epsilon_pole: float | None = None,
    polar_cooling_scale: float = 0.3,  # Reduced from 0.6 to allow more polar warming
    ice_freeze_temp: float = 269.9,  # Require colder water before new sea ice forms
    ice_melt_temp: float = 271.4,    # Preserve some hysteresis without locking in subpolar ice
    ice_freeze_rate: float = 0.045,
    ice_melt_rate: float = 0.19,
    ice_albedo_strength: float = 0.30,  # Reduced from 0.50: weakens ice-albedo runaway; eff full-ice albedo=0.22
    heat_transport_coeff: float = 0.8,
    thermal_diffusion: float = 0.04,
    latent_cooling_coeff: float = 0.015,
    enable_carbon_cycle: bool = True,
    co2_climate_feedback: float = 0.8,
    debug_log: bool = False,
    track_components: bool = False,
    planet_params: PlanetParams | None = None,
) -> tuple[PlanetState, dict]:
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
            koppen_new = classify_koppen(
                monthly_temp, monthly_precip, land_mask_for_biomes,
                elevation=state.elevation
            )
            # Convert Köppen to legacy biome for backward compatibility
            biome_new = koppen_to_legacy_biome(koppen_new)
            biome_last_update = new_total_days
            if debug_log:
                from terrain import LOG
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
    # CO2 Greenhouse Forcing (if carbon cycle enabled)
    # ------------------------------------------------------
    # CRITICAL FIX: CO2 forcing must be applied to BASE TEMPERATURE before temperature evolution,
    # not added to final temperature afterward (which would cause runaway warming).
    # The forcing represents the equilibrium temperature offset that the simulation should relax toward.
    co2_temp_offset = 0.0
    if enable_carbon_cycle:
        co2_forcing = co2_radiative_forcing(state.co2_atmosphere, CO2_PREINDUSTRIAL)
        co2_temp_offset = co2_temperature_response(co2_forcing, co2_climate_feedback)
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
    T_base_land = np.repeat(T_lat_land[:, None], Wc, axis=1).astype(np.float32) + co2_temp_offset
    # Land summer temperature cap: temperature_kelvin_for_lat gives radiative equilibrium,
    # but real land surfaces never reach those peaks because latent heat (ET), soil heat
    # capacity, and turbulent exchange all limit summer temperatures.
    # At 55-65°N the formula returns ~305 K (32°C) in summer — unrealistically hot —
    # pulling the simulation 12-13°C above Earth via the land_blend.
    # Cap: 301 K (~28°C) at the equator, tapering linearly to 286 K (13°C) at 60°+.
    # This matches observed mid-latitude summer maxima for mean daily temperatures.
    _abs_lat_deg_land = np.abs(np.rad2deg(lat))  # (Hc,)
    _land_cap_1d = 301.0 - 15.0 * np.clip(_abs_lat_deg_land / 60.0, 0.0, 1.0)
    # Atmospheric meridional heat transport warms high-latitude land.
    # The Ferrel cell and synoptic eddies carry ~60% of the ocean transport value
    # poleward even over the Antarctic continent.  Without this, Antarctic winter
    # equilibrium falls to ~185 K; the 200 K floor then dominates the annual mean,
    # giving ~203 K (vs Earth 224 K at the South Pole).  Adding 0.60 × 40 K = 24 K
    # at the pole lifts the winter minimum to ~210 K and the annual mean to ~220-226 K.
    # Symmetric for both poles (no AMOC; that term is ocean-only and applied separately).
    _atm_land_transport_1d = (
        0.42 * 34.0 * np.clip((_abs_lat_deg_land - 42.0) / 28.0, 0.0, 1.0) ** 1.5
    )
    T_base_land = T_base_land + _atm_land_transport_1d[:, None].astype(np.float32)
    # Re-apply summer cap: atmospheric transport can only raise winter/polar-night
    # temperatures; it must not push summer land above observed peak means.
    T_base_land = np.minimum(T_base_land, _land_cap_1d[:, None].astype(np.float32))

    # Calculate temperature for lagged day (ocean response with 1.5 month delay)
    lag_days = 50.0 * (float(pp.orbital_period_days) / 365.2422)  # scale thermal lag with year length
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
    _transport_base = 34.0 * np.clip((lat_deg_1d - 42.0) / 28.0, 0.0, 1.0) ** 1.5
    # AMOC bonus: 18K ramp 40-65°N (unchanged) + extra 10K ramp 65-85°N only.
    # 65°N is already well-calibrated (-2°C vs Earth -2°C), so we only add warmth
    # above 65°N where the NH ice runaway causes an 8-10 K cold bias.
    _amoc_bonus = np.where(
        lat > 0,
        16.0 * np.clip((lat_deg_1d - 42.0) / 23.0, 0.0, 1.0)
        + 6.0 * np.clip((lat_deg_1d - 68.0) / 16.0, 0.0, 1.0),
        0.0,
    )  # NH only; at 85°N total = 28K
    _sh_factor = np.where(lat > 0, 1.0, 0.58)   # SH gets weaker baseline transport than NH (no AMOC)
    transport_warming = _transport_base * _sh_factor + _amoc_bonus

    # Seasonal fraction: what fraction of the radiative swing the ocean actually feels
    # Based on ΔT/ΔT_rad ≈ 1/sqrt(1 + (2π τ/P)²) where τ ~ 1-3 years, P = 1 year
    # Equator (τ~0.7yr): ~24%, Mid-lat (τ~1.5yr): ~10%, Polar (τ~3yr): ~5%
    obliq_factor = np.clip(float(pp.obliquity_deg) / 23.44, 0.6, 2.0) ** 0.5
    ocean_seasonal_frac = (0.05 + 0.20 * np.cos(np.deg2rad(lat_deg_1d))) * obliq_factor
    ocean_seasonal_frac = np.clip(ocean_seasonal_frac, 0.03, 0.45)

    # Final ocean base: annual mean + transport warming + small seasonal oscillation
    T_lat_ocean = (T_lat_annual_mean + transport_warming
                   + ocean_seasonal_frac * (T_lat_ocean_lagged - T_lat_annual_mean))

    T_base_ocean = np.repeat(T_lat_ocean[:, None], Wc, axis=1).astype(np.float32) + co2_temp_offset
    # Full-resolution fallback base temperature (used when wind evolves at full resolution
    # before `state.temperature` is initialized).
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
    _transport_base_full = 34.0 * np.clip((lat_deg_full - 42.0) / 28.0, 0.0, 1.0) ** 1.5
    _amoc_bonus_full = np.where(
        lat_full > 0,
        16.0 * np.clip((lat_deg_full - 42.0) / 23.0, 0.0, 1.0)
        + 6.0 * np.clip((lat_deg_full - 68.0) / 16.0, 0.0, 1.0),
        0.0,
    )
    _sh_factor_full = np.where(lat_full > 0, 1.0, 0.58)
    transport_warming_full = _transport_base_full * _sh_factor_full + _amoc_bonus_full
    ocean_seasonal_frac_full = (0.05 + 0.20 * np.cos(np.deg2rad(lat_deg_full))) * obliq_factor
    ocean_seasonal_frac_full = np.clip(ocean_seasonal_frac_full, 0.03, 0.45)
    T_lat_ocean_full = (T_lat_annual_mean_full + transport_warming_full
                        + ocean_seasonal_frac_full * (T_lat_ocean_full_lagged - T_lat_annual_mean_full))
    T_base_ocean_full = np.repeat(T_lat_ocean_full[:, None], W, axis=1).astype(np.float32) + co2_temp_offset
    
    # Blend based on land fraction (will be calculated in _evolve_temperature)
    # Use ocean-lagged temperature as base; _evolve_temperature will handle land/ocean mixing
    T_base = T_base_ocean  # Start with ocean (lagged), land will be corrected in evolution
    
    # Compute coarse elevation grid for wind/temperature evolution
    if state.elevation is not None:
        elev_pad = np.pad(state.elevation.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        elev_c = elev_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    else:
        elev_c = None
    
    # Update temperature with wind advection and land-sea effects
    if state.temperature is not None:
        # Use previous temperature as starting point for advection
        T_prev_pad = np.pad(state.temperature.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        T_prev_coarse = T_prev_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    else:
        T_prev_coarse = T_base.copy()
    if state.ice_cover is not None:
        ice_pad = np.pad(state.ice_cover.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        ice_prev_coarse = ice_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    else:
        ice_prev_coarse = None
    
    # ------------------------------------------------------
    # NEW: Prognostic Wind Evolution (Physics Items 16-33)
    # ------------------------------------------------------
    # If wind is None, initialize it near-rest (small noise) so circulation spins up
    # from pressure gradients (Hadley-like overturning) rather than a synthetic target.
    if state.wind_u is None or state.wind_v is None:
        rng = np.random.default_rng(12345)
        u_full = rng.normal(0.0, 0.15, size=(H, W)).astype(np.float32)
        v_full = rng.normal(0.0, 0.15, size=(H, W)).astype(np.float32)
    else:
        u_full, v_full = state.wind_u, state.wind_v
        
    # Evolve wind at `wind_block_size` resolution (can differ from temperature/precip `block_size`)
    # Then upsample to full resolution for precipitation
    # Cached diagnostic wind for relaxation (once per day/shape/params).
    def _diag_wind_cached(h: int, w: int, temp_field: np.ndarray, elev_field: np.ndarray):
        key = (
            h,
            w,
            int(new_day),  # only vary daily
            float(wind_target_weather_amp),
            float(wind_target_zonal_pressure),
            float(wind_target_terrain_pressure_amp),
            float(wind_target_terrain_flow_amp),
            round(float(pp.solar_constant), 4),
            round(float(pp.obliquity_deg), 4),
            round(float(pp.sidereal_day_hours), 4),
            round(float(pp.radius_m), 1),
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
        )
        cache.update({"key": key, "u": u_diag, "v": v_diag})
        return u_diag, v_diag

    if wind_bs > 1:
        # Downsample wind/temperature/elevation for evolution on the wind grid.
        u_pad = np.pad(u_full.astype(np.float32), ((0, Hcw*wind_bs - H), (0, Wcw*wind_bs - W)), mode="edge")
        v_pad = np.pad(v_full.astype(np.float32), ((0, Hcw*wind_bs - H), (0, Wcw*wind_bs - W)), mode="edge")
        u_coarse_evol = u_pad.reshape(Hcw, wind_bs, Wcw, wind_bs).mean(axis=(1, 3))
        v_coarse_evol = v_pad.reshape(Hcw, wind_bs, Wcw, wind_bs).mean(axis=(1, 3))

        elev_pad_w = np.pad(state.elevation.astype(np.float32), ((0, Hcw*wind_bs - H), (0, Wcw*wind_bs - W)), mode="edge")
        elev_c_w = elev_pad_w.reshape(Hcw, wind_bs, Wcw, wind_bs).mean(axis=(1, 3))

        if state.temperature is not None:
            T_pad_w = np.pad(state.temperature.astype(np.float32), ((0, Hcw*wind_bs - H), (0, Wcw*wind_bs - W)), mode="edge")
            T_for_wind = T_pad_w.reshape(Hcw, wind_bs, Wcw, wind_bs).mean(axis=(1, 3))
        else:
            # When temperature is not yet initialized, use the same lagged-ocean base but on the wind grid.
            lat_w = (0.5 - (np.arange(Hcw, dtype=np.float32) + 0.5) / Hcw) * np.pi
            T_lat_ocean_w = temperature_kelvin_for_lat(
                lat_w,
                day_of_year=int(lagged_day),
                polar_cooling_scale=polar_cooling_scale,
                planet_params=pp,
            )
            T_for_wind = np.repeat(T_lat_ocean_w[:, None], Wcw, axis=1).astype(np.float32)

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
        )

        # Keep winds energized + seasonally varying by weakly relaxing toward a diagnostic wind
        # (generate_wind_field injects synoptic-scale "weather systems" seeded by day_of_year).
        if wind_relax > 0.0:
            u_diag, v_diag = _diag_wind_cached(Hcw, Wcw, T_for_wind, elev_c_w)
            a = float(np.clip(wind_relax, 0.0, 1.0))
            u_coarse_evol = (1.0 - a) * u_coarse_evol + a * u_diag
            v_coarse_evol = (1.0 - a) * v_coarse_evol + a * v_diag
        
        # Upsample back to full resolution using bilinear interpolation
        uv = _upsample_bilinear_many({"u": u_coarse_evol, "v": v_coarse_evol}, H, W, wind_bs)
        u_full, v_full = uv["u"], uv["v"]
    else:
        # Full resolution evolution
        # If wind evolves at higher resolution than the temperature solver, drive it with the
        # coarse temperature field upsampled to full resolution. This avoids injecting
        # grid-scale temperature noise into the wind solver (which can blow up speeds),
        # while still allowing the wind numerics to run on the fine grid.
        T_wind_full = state.temperature if state.temperature is not None else T_base_ocean_full
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
        )
        if wind_relax > 0.0:
            T_for_wind = T_wind_full
            u_diag, v_diag = _diag_wind_cached(H, W, T_for_wind, elev_wind_full)
            a = float(np.clip(wind_relax, 0.0, 1.0))
            u_full = (1.0 - a) * u_full + a * u_diag
            v_full = (1.0 - a) * v_full + a * v_diag

    # Winds to couple into temperature evolution operate on the temperature grid (Hc,Wc).
    if block_size > 1:
        u_pad_t = np.pad(u_full.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        v_pad_t = np.pad(v_full.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        u_coarse = u_pad_t.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
        v_coarse = v_pad_t.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    else:
        u_coarse = u_full
        v_coarse = v_full

    # Apply temperature evolution with advection and radiation
    if state.humidity is not None:
        hum_pad = np.pad(state.humidity.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        humidity_coarse = hum_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    else:
        humidity_coarse = None

    # Downsample precipitation and vegetation biomass for vegetation albedo (Phase 4)
    if state.precipitation is not None:
        P_pad = np.pad(state.precipitation.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        precipitation_coarse = P_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    else:
        precipitation_coarse = None

    if state.vegetation_biomass is not None:
        biomass_pad = np.pad(state.vegetation_biomass.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        biomass_coarse = biomass_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    else:
        biomass_coarse = None

    # Downsample biomes to coarse resolution if available.
    # Use the center element of each block instead of element [0,0] — the center
    # is representative of the block and avoids the top-left corner bias.
    _mid = block_size // 2
    if biome_new is not None:
        biome_pad = np.pad(biome_new.astype(np.int32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        biome_coarse = biome_pad.reshape(Hc, block_size, Wc, block_size)[:, _mid, :, _mid]
    else:
        biome_coarse = None

    # Downsample Köppen classification to coarse resolution if available
    if koppen_new is not None:
        koppen_pad = np.pad(koppen_new.astype(np.int32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        koppen_coarse = koppen_pad.reshape(Hc, block_size, Wc, block_size)[:, _mid, :, _mid]
    else:
        koppen_coarse = None

    # Always track components for diagnostics (minimal overhead)
    T_coarse, cloud_c, snow_c, temp_components = _evolve_temperature(
        T_prev_coarse, T_base, state.elevation, Hc, Wc, block_size, H, W,
        day_of_year=int(new_day), days=days,
        wind_u=u_coarse, wind_v=v_coarse, # Pass evolved wind
        T_base_land=T_base_land,  # Pass land temperature for seasonal lag correction
        ice_cover=ice_prev_coarse,
        heat_transport_coeff=heat_transport_coeff,
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
        track_components=track_components,  # Track components for diagnostics
        precipitation=precipitation_coarse,  # For vegetation albedo (Phase 4)
        vegetation_biomass=biomass_coarse,   # For vegetation albedo (Phase 4)
        biome=biome_coarse,  # Phase 1: Stable biomes for albedo
        koppen_type=koppen_coarse,  # Köppen classification for detailed albedo
        planet_params=pp,
        elev_c=elev_c,  # Pre-computed coarse elevation
    )
    
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
        up = _upsample_bilinear_many({"T": T_coarse, "cloud": cloud_c}, H, W, block_size)
        T_full, cloud_full = up["T"], up["cloud"]
    else:
        T_full = T_coarse
        cloud_full = cloud_c

    # Update wind from temperature gradients (if requested)
    # Already evolved above
    # if update_wind... removed legacy block
    
    if state.elevation is not None:
        P_full, humidity_next, soil_next = generate_precipitation(
            H,
            W,
            state.elevation,
            temperature=T_full,
            wind_u=u_full,
            wind_v=v_full,
            humidity=state.humidity,
            soil_moisture=state.soil_moisture,
            day_of_year=int(new_day),
            dt_days=float(days),
        )
    else:
        P_full = None
        humidity_next = None
        soil_next = None
    # NOTE: latent cooling from precipitation is already applied inside
    # _evolve_temperature (via evaporation) and generate_precipitation.
    # Applying it again here was a double-count and has been removed.
    if T_full is not None:
        T_full = np.clip(T_full, 150.0, 330.0)
    if T_full is not None:
        ice_full, delta_ice = update_sea_ice(
            T_full, state.elevation, state.ice_cover, days,
            freeze_temp=ice_freeze_temp,
            melt_temp=ice_melt_temp,
            freeze_rate=ice_freeze_rate,
            melt_rate=ice_melt_rate,
        )
        # Ice-ocean latent heat feedback: freezing releases heat, melting absorbs heat
        # L_f=334 kJ/kg, rho_ice=917 kg/m³, ~1m effective thickness, ~100m mixed layer
        # gives ~3K per unit ice fraction change
        latent_scale = 3.0  # K per unit ice fraction change
        is_ocean_full, _ = get_masks(state.elevation)
        T_full = T_full + delta_ice * latent_scale * is_ocean_full.astype(np.float32)
    else:
        ice_full = None

    # Debug logging if requested
    if debug_log:
        import logging
        LOG = logging.getLogger("planetsim")
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
        # Create temporary state for carbon cycle computation
        temp_state_for_carbon = PlanetState(
            day_of_year=new_day,
            total_days=new_total_days,
            elevation=state.elevation,
            temperature=T_full,
            wind_u=u_full,
            wind_v=v_full,
            precipitation=P_full,
            co2_atmosphere=state.co2_atmosphere,
            co2_ocean=state.co2_ocean,
            vegetation_biomass=state.vegetation_biomass,
        )

        # Evolve carbon cycle
        co2_atm_new, co2_ocean_new, biomass_new, co2_forcing_result = carbon_cycle_step(
            temp_state_for_carbon, days
        )

        # CO2 greenhouse feedback is now applied to T_base (equilibrium temperature) above,
        # not added to final temperature here. This prevents runaway warming.

        if debug_log:
            LOG.info(f"Carbon cycle: CO2={co2_atm_new:.1f} ppm, forcing={co2_forcing_result:.2f} W/m²")
    else:
        co2_atm_new = state.co2_atmosphere
        co2_ocean_new = state.co2_ocean
        biomass_new = state.vegetation_biomass

    new_state = PlanetState(
        day_of_year=new_day,
        total_days=new_total_days,
        elevation=state.elevation,
        temperature=T_full,
        wind_u=u_full,
        wind_v=v_full,
        precipitation=P_full,
        humidity=humidity_next,
        soil_moisture=soil_next,
        cloud_cover=cloud_full,
        snow_depth=None,
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
    state = PlanetState(
        day_of_year=day_of_year,
        total_days=0.0,
        elevation=elevation,
        temperature=None,
        wind_u=None,
        wind_v=None,
        precipitation=None,
        humidity=None,
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
    wind_u: np.ndarray | None = None,
    wind_v: np.ndarray | None = None,
    heat_transport_coeff: float = 0.8,  # CONSERVATIVE increase from 0.5 (1.6x) for stability
    land_sea_contrast: float = 0.0,
    thermal_diffusion: float = 0.04,    # CONSERVATIVE increase from 0.03 (1.3x) respects CFL
    T_base_land: np.ndarray | None = None,
    ice_cover: np.ndarray | None = None,
    ocean_transport_coeff: float = 0.5,
    ocean_exchange_floor: float = 0.65,
    ocean_exchange_span: float = 0.35,
    ocean_exchange_coeff: float = 0.05,
    ocean_exchange_inertia: float = 0.0,
    epsilon_equator: float = 0.72,
    epsilon_pole: float = 0.50,
    ice_albedo_strength: float = 1.0,
    humidity: np.ndarray | None = None,
    track_components: bool = False,
    precipitation: np.ndarray | None = None,
    vegetation_biomass: np.ndarray | None = None,
    biome: np.ndarray | None = None,  # Phase 1: Stable biome classification
    koppen_type: np.ndarray | None = None,  # Köppen climate classification
    planet_params: PlanetParams | None = None,
    elev_c: np.ndarray | None = None,  # Pre-computed coarse elevation (avoids recomputation)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
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
        elev_pad = np.pad(elevation.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        elev_c = elev_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    # Land/Sea Masks — elev_c is a transient coarse array; skip cache to avoid stale hits
    sea_mask, land_mask = get_masks(elev_c, use_cache=False)
    land_fraction = land_mask.astype(np.float32) # Simplified for now
    
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

    # 3. Thermodynamic Energy Balance
    # dT/dt = Advection + Radiation + Phase Change + Diffusion

    T = T_prev.copy()

    # --- Advection (Wind Transport) ---
    # CFL-correct upwind advection: c = |u| * dt / dx, capped at 0.5 for stability.
    # dx varies with latitude (spherical geometry): dx = 2π R cos(φ) / Wc
    # dy is uniform: dy = π R / Hc
    _pp = planet_params if planet_params is not None else EARTH
    lat_1d = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    cos_lat = np.cos(lat_1d).clip(0.05, 1.0)  # avoid dx→0 at poles
    dx_lat = (2.0 * np.pi * _pp.radius_m * cos_lat / Wc).astype(np.float32)  # (Hc,)
    dy = float(np.pi * _pp.radius_m / Hc)

    dx_2d = dx_lat[:, None]  # broadcast to (Hc, Wc)
    u_cfl = np.clip(np.abs(u) * dt_sec / dx_2d, 0.0, 0.5).astype(np.float32)
    v_cfl = np.clip(np.abs(v) * dt_sec / dy, 0.0, 0.5).astype(np.float32)

    # Only copy when component tracking is active — avoids one full-array allocation per step.
    T_before_advection = T.copy() if track_components else None

    if NUMBA_AVAILABLE:
        # Fast path: Numba-accelerated CFL-correct advection
        T = _advect_temperature_x_numba(T.astype(np.float32), u.astype(np.float32), u_cfl)
        T = _advect_temperature_y_numba(T.astype(np.float32), v.astype(np.float32), v_cfl)
    else:
        # Fallback: NumPy upwind implementation
        T_east = np.roll(T, -1, axis=1)
        T_west = np.roll(T, 1, axis=1)
        T_x = np.where(u >= 0, T_west, T_east)
        T_diff_x = np.clip(T_x - T, -12.0, 12.0)
        T = T + u_cfl * T_diff_x

        T_north = np.roll(T, -1, axis=0)
        T_south = np.roll(T, 1, axis=0)
        T_y = np.where(v >= 0, T_south, T_north)
        T_diff_y = np.clip(T_y - T, -12.0, 12.0)
        T = T + v_cfl * T_diff_y

    # --- Diffusion (Mixing) ---
    T_before_diffusion = T.copy() if track_components else None

    if NUMBA_AVAILABLE:
        # Fast path: Numba-accelerated diffusion
        # Phase 4: Reduced from iterations=3 to iterations=2 for ~10ms speedup
        T = _apply_diffusion_numba(T.astype(np.float32), float(thermal_diffusion),
                                   float(days), iterations=2)
    else:
        # Fallback: original NumPy implementation
        # Phase 4: Reduced from 3 to 2 iterations
        for _ in range(2):
            T_pad = np.pad(T, ((1, 1), (0, 0)), mode="edge")
            c = T_pad[1:-1, :]
            n = T_pad[0:-2, :]
            s = T_pad[2:, :]
            e = np.roll(c, -1, axis=1)
            w = np.roll(c, 1, axis=1)
            T_lap = n + s + e + w - 4.0 * c
            T_lap = np.clip(T_lap, -30.0, 30.0)
            T = T + thermal_diffusion * 1.2 * T_lap * days
        
    # --- Radiative Balance (Physics Item 1, 2, 10, 12) ---
    # Incoming Solar (S_in) - Albedo (A)
    # A depends on: Land/Ocean, Snow/Ice, Cloud
    
    # Approx Latitude
    lat = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    lat_2d = np.repeat(lat[:, None], Wc, axis=1)
    
    # Ice/Snow Cover approximation based on Temp
    # T < 271 K -> Snow/Ice likely
    snow_cover = np.clip((273.15 - T) / 10.0, 0.0, 1.0)
    sea_ice = np.zeros_like(T, dtype=np.float32) if ice_cover is None else np.clip(ice_cover.astype(np.float32), 0.0, 1.0)
    sea_ice = np.where(sea_mask, sea_ice, 0.0)
    if ice_albedo_strength != 1.0:
        sea_ice = np.clip(sea_ice * float(ice_albedo_strength), 0.0, 1.0)
    
    # Cloud Cover approximation (physics-based, no artificial floor)
    # Use humidity when available, plus ascent, orographic lift, and subsidence clearing.
    Tc = np.clip(T - 273.15, -60.0, 60.0)
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    qsat = np.clip(0.622 * es / 1013.25, 1e-6, 0.035).astype(np.float32)
    if humidity is not None:
        q = np.clip(humidity.astype(np.float32), 0.0, qsat)
    else:
        temp_norm = np.clip((T - 255.0) / 45.0, 0.0, 1.0)
        base_q = np.where(sea_mask, 0.012, 0.008).astype(np.float32)
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
    
    # Albedo (with vegetation feedback - Phase 4)
    # Ocean: 0.06, Sea Ice: 0.75, Snow: 0.8, Cloud: 0.5
    # Land albedo now depends on vegetation/biome type

    # Compute biome-based vegetation albedo (if biomass field exists)
    # Phase 1 improvement: Use stable biomes from long-term climate averages (not daily weather)
    # Köppen classification provides more detailed albedo values
    if vegetation_biomass is not None and biome is not None:
        albedo_veg = vegetation_albedo(biome, base_land_albedo=0.2, koppen_type=koppen_type)
        # Use vegetation albedo for land, ocean base albedo for sea
        albedo_sfc = np.where(sea_mask, 0.06 * (1.0 - sea_ice) + 0.65 * sea_ice, albedo_veg)
    else:
        # Fallback: uniform land albedo
        albedo_sfc = np.where(sea_mask, 0.06 * (1.0 - sea_ice) + 0.65 * sea_ice, 0.2)

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
    
    sigma = 5.67e-8
    L_out = epsilon * sigma * (T ** 4)
    
    # Net Radiation
    R_net = S_absorbed - L_out # W/m2
    
    # --- Heat Capacity (Physics Item 7) ---
    # Ocean: High (mixed layer depth ~50m) -> large Cp
    # Land: Low (soil depth ~1m) -> small Cp
    # C_p approx: Water=4200 J/kg/K, Density=1000, Depth=50 -> 2e8 J/K/m2
    # Land: Soil=800 J/kg/K, Density=2000, Depth=0.5 -> 8e5 J/K/m2
    # Ratio ~ 250:1
    
    # Effective Heat Capacity (J / m^2 / K) per day
    # Scaled for stability and speed (days unit)
    # Use relaxation time scale instead of explicit flux to avoid stiff equations
    # dT = R_net / Cp * dt
    
    # Relaxation approach (Newtonian Cooling) to Equilibrium Teq
    # Teq = (S_absorbed / (epsilon * sigma))**0.25
    # dT = k * (Teq - T)
    
    # Calculate equilibrium T based on radiation
    T_eq_rad = (S_absorbed / (epsilon * sigma + 1e-9)) ** 0.25
    T_eq_rad = np.clip(T_eq_rad, 150.0, 350.0) # Safety
    
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

    # --- Ocean Temperature Cap (CRITICAL FIX for ocean overheating) ---
    # Real-world ocean SST maximum is ~32°C (305K) due to strong evaporative cooling.
    # The base temperature calculation (radiative equilibrium) produces 43-48°C for
    # subtropical oceans because it doesn't account for evaporation.
    # Cap ocean equilibrium temperature at 305K to match real-world physics.
    # This represents the fact that ocean evaporation prevents SST from exceeding ~32°C.
    T_eq = np.where(sea_mask, np.minimum(T_eq, 305.0), T_eq)

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
    # Up to 80% MLD reduction at poles in polar summer → ~37m → 19-day time constant
    mld = mld * (1.0 - 0.80 * _polar_ramp * _hemi_summer)
    k_ocean = np.clip(1.0 / (mld * 0.5), 0.005, 0.07)  # 14-200 day time constants

    # Ice insulation: asymmetric effect on ocean-atmosphere coupling
    # Ice insulates against COOLING (prevents heat loss in winter) but does NOT
    # block warming — solar radiation melts ice from above, warm currents erode
    # from below, and leads/polynyas allow heat exchange. This prevents ice from
    # becoming permanently locked in at mid-latitudes.
    cooling_direction = T_eq < T  # True where forcing would cool the ocean
    ice_insulation = np.where(
        cooling_direction,
        1.0 - 0.7 * sea_ice,  # 70% insulation against cooling (keeps ocean warm under ice)
        1.0,                    # No insulation against warming (allows spring melt)
    )
    # Land heat capacity is ~100× less than ocean, but we use a 10-day
    # timescale (k=0.1/day) rather than 1.0/day to avoid single-step
    # overshoots and oscillations.  k=1.0 meant T jumped all the way to T_eq
    # in a single day-step, which produced unrealistic diurnal-scale noise.
    k_relax = np.where(sea_mask, k_ocean * ice_insulation, 0.1)
    
    # Apply Radiative forcing via relaxation
    # Limit relaxation to prevent extreme jumps and reduce oscillations
    T_change = k_relax * (T_eq - T) * days
    T_change = np.clip(T_change, -8.0, 8.0)  # Reduced from ±10K to ±8K per day to reduce oscillations
    T = T + T_change
    
    # --- Phase 2: Wind-Dependent Evaporation (Bulk Aerodynamic Formula) ---
    # E = C_D × wind_speed × (qsat - q)
    # This makes evaporation realistic: stronger with high winds and dry air
    if wind_u is not None and wind_v is not None and humidity is not None:
        # Wind speed
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)

        # Saturation humidity (Clausius-Clapeyron)
        T_c = np.clip(T - 273.15, -60.0, 60.0)
        es = 6.112 * np.exp(17.67 * T_c / (T_c + 243.5))  # Saturation vapor pressure [hPa]
        qsat = np.clip(0.622 * es / 1013.25, 1e-6, 0.035)  # Saturation specific humidity [kg/kg]

        # Humidity deficit (how much more moisture air can hold)
        deficit = np.maximum(0.0, qsat - humidity)

        # Drag coefficient (higher over rough ocean surface)
        C_D = np.where(sea_mask, 1.5e-3, 0.5e-3)  # Ocean vs land

        # Evaporation rate [mm/day equivalent]
        E = C_D * wind_speed * deficit * 1000.0
        E = np.clip(E, 0.0, 20.0)  # Cap at 20 mm/day (realistic maximum)

        # Latent cooling: 2.5 K per mm/day evaporated
        evap_cooling = E * 2.5 * days

        # --- Enhanced evaporative cooling for hot oceans ---
        # Real tropical oceans have intense evaporation that prevents SST > 32°C
        # Add extra evaporative cooling that scales with temperature above 30°C (303K)
        # This represents the nonlinear increase in evaporation at high SST
        # REDUCED: Changed threshold from 28C to 30C and coefficient from 0.5 to 0.3
        # to allow tropical oceans to reach realistic 28-30°C temperatures
        hot_ocean_excess = np.where(sea_mask & (T > 303.0), T - 303.0, 0.0)
        hot_ocean_cooling = 0.3 * hot_ocean_excess * days  # Extra 0.3 K/day per degree above 30°C
        evap_cooling = evap_cooling + hot_ocean_cooling

        T = T - evap_cooling
    else:
        # Fallback to simple temperature-dependent evaporation if wind/humidity not available
        base_evap = np.where(sea_mask, 0.01 * (T - 270.0), 0.0)
        # Enhanced hot ocean evaporation - kicks in at 30°C (303K)
        # Coefficient tuned to prevent >32°C while allowing realistic 28-30°C tropics
        hot_evap = np.where((T > 303.0) & sea_mask, 0.3 * (T - 303.0), 0.0)
        evap_cooling = np.maximum(0.0, base_evap + hot_evap)
        T = T - evap_cooling * days
    
    # --- Surface Physics (Items 5, 8, 9) ---
    # Blend land temperatures toward T_base_land (which has proper seasonal response)
    # Ocean already uses T_base_ocean (lagged) in the equilibrium calculation above
    if T_base_land is not None:
        # IMPORTANT: apply the same lapse-rate correction here; otherwise this blend
        # pulls mountains back toward the latitude-only baseline and cancels orographic cooling.
        T_base_land = T_base_land - lapse_rate * alt_km
        # Land should track T_base_land more closely (it has immediate seasonal response)
        land_blend = np.where(land_mask, 0.2, 0.0)  # 20% pull toward base for land (was 0.3; reduces Arctic land overcooling)
        T = (1.0 - land_blend) * T + land_blend * T_base_land

    # --- Ocean Transport (temporally sub-sampled for performance) ---
    # Ocean circulation has a decorrelation time of ~30 days, so we recompute
    # the heat-transport adjustment only once per OCEAN_UPDATE_INTERVAL_DAYS and
    # reuse the cached ΔT array in between.  The cache key includes the grid
    # shape and day-of-year so it resets automatically on resolution changes or
    # simulation restarts.
    OCEAN_UPDATE_INTERVAL_DAYS = 30.0
    _oc = _OCEAN_ADJ_CACHE
    _oc_key = (Hc, Wc, round(day_of_year))
    _total = float(np.sum(T))  # cheap proxy to detect if grid changed
    days_since_ocean = float(day_of_year) - float(_oc.get("last_update_day", -9999.0))
    if (
        _oc.get("adj") is None
        or _oc.get("key") != _oc_key
        or abs(days_since_ocean) >= OCEAN_UPDATE_INTERVAL_DAYS
    ):
        T_ocean_adj = calculate_ocean_heat_transport(
            T, elev_c, Hc, Wc, day_of_year, days,
            transport_coefficient=float(ocean_transport_coeff),
            exchange_strength_floor=float(ocean_exchange_floor),
            exchange_strength_span=float(ocean_exchange_span),
            exchange_coefficient=float(ocean_exchange_coeff),
            exchange_inertia=float(ocean_exchange_inertia),
            prev_T=T_prev,
            ice_cover=sea_ice,
            T_equilibrium=T_eq,
        )
        T_ocean_adj = np.clip(T_ocean_adj, -10.0, 10.0)
        _oc["adj"] = T_ocean_adj
        _oc["key"] = _oc_key
        _oc["last_update_day"] = float(day_of_year)
    else:
        T_ocean_adj = _oc["adj"]
    T = T + T_ocean_adj

    # --- Hadley/Subsidence (Keep existing simplified param) ---
    # (Ideally this emerges from dynamics, but keeping parameterization for stability)
    # Recalculate Hadley simply - REDUCED from 5K to 0.5K per day (was too strong)
    lat_deg = np.rad2deg(np.abs(lat_2d))
    subsidence = 0.10 * np.exp(-((lat_deg - 30.0)/10.0)**2) * days
    T = T + subsidence

    # --- FINAL TEMPERATURE CLAMPING (Critical for stability) ---
    # Clamp to realistic Earth-like temperature range
    # Absolute minimum: -73°C (200K) - allows for realistic Antarctic winter (Vostok: -89°C)
    # Absolute maximum: +50°C (323K) - prevents extreme hot spots
    # Previous floor of 240K (-33°C) was too high and prevented realistic polar temperatures
    T = np.clip(T, 200.0, 323.0)  # Lowered minimum from 240K to 200K for realistic polar winters

    # Track component contributions if requested
    components = {}
    if track_components:
        # Calculate what each component contributed (in K change)
        # T_before_* are only allocated when track_components=True (see above)
        components['advection'] = T - T_before_advection  # type: ignore[operator]
        components['diffusion'] = T - T_before_diffusion  # type: ignore[operator]

        # For other components, compute the change they would have caused
        # (these are already computed in the main simulation loop)
        components['radiation'] = k_relax * (T_eq - T) * days
        components['evaporation'] = -evap_cooling  # Phase 2: Now wind-dependent
        components['ocean_transport'] = T_ocean_adj
        components['subsidence'] = subsidence
        components['equilibrium_temp'] = T_eq
        components['net_radiation'] = R_net
        def _summ(field: np.ndarray) -> dict:
            return {
                "mean": float(np.mean(field)),
                "min": float(np.min(field)),
                "max": float(np.max(field)),
            }
        components["toa"] = {
            "S_absorbed": _summ(S_absorbed),
            "L_out": _summ(L_out),
            "R_net": _summ(R_net),
            "albedo_mean": float(np.mean(albedo_total)),
            "cloud_mean": float(np.mean(cloud_fraction)),
            "epsilon_mean": float(np.mean(epsilon)),
        }

    return T.astype(np.float32), cloud_fraction.astype(np.float32), snow_cover.astype(np.float32), components


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


