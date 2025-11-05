"""Time simulation for planet conditions.

Advances atmospheric systems (temperature, wind, precipitation) forward in time
with configurable time scales. Default unit is one day.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple
from atmosphere import generate_wind_field, generate_precipitation
from temperature import temperature_kelvin_for_lat


class PlanetState(NamedTuple):
    """Current planet state snapshot."""
    day_of_year: float  # Fractional day (0-365.2422)
    elevation: np.ndarray  # (H, W) terrain elevation [0,1]
    temperature: np.ndarray | None = None  # (H, W) temperature (K)
    wind_u: np.ndarray | None = None  # (H, W) eastward wind (m/s)
    wind_v: np.ndarray | None = None  # (H, W) northward wind (m/s)
    precipitation: np.ndarray | None = None  # (H, W) precipitation (mm/day)
    humidity: np.ndarray | None = None  # (H, W) specific humidity


def simulate_step(
    state: PlanetState,
    days: float = 1.0,
    *,
    block_size: int = 3,
    evap_coeff: float = 1.0,
    uplift_coeff: float = 1.0,
    rain_efficiency: float = 0.7,
    precip_iterations: int = 48,
    update_wind: bool = True,
    update_precip: bool = True,
) -> PlanetState:
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
        evap_coeff: Evaporation coefficient
        uplift_coeff: Orographic uplift coefficient
        rain_efficiency: Rain efficiency
        precip_iterations: Precipitation solver iterations
        update_wind: Whether to recompute wind field
        update_precip: Whether to recompute precipitation

    Returns:
        New state with updated day_of_year and atmospheric fields
    """
    new_day = (state.day_of_year + days) % 365.2422
    H, W = state.elevation.shape
    Hc, Wc = (max(1, (H + block_size - 1) // block_size),
              max(1, (W + block_size - 1) // block_size))

    # Get base insolation temperature (latitude-dependent)
    lat = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    T_lat = temperature_kelvin_for_lat(lat, day_of_year=int(new_day))
    T_base = np.repeat(T_lat[:, None], Wc, axis=1).astype(np.float32)
    
    # Update temperature with wind advection and land-sea effects
    if state.temperature is not None:
        # Use previous temperature as starting point for advection
        T_prev_pad = np.pad(state.temperature.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        T_prev_coarse = T_prev_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    else:
        T_prev_coarse = T_base.copy()
    
    # Apply temperature evolution with advection
    T_coarse = _evolve_temperature(
        T_prev_coarse, T_base, state.elevation, Hc, Wc, block_size, H, W,
        day_of_year=int(new_day), days=days
    )
    
    if block_size > 1:
        # Use bilinear interpolation for smooth upsampling (eliminates blocky artifacts)
        # Create coordinate arrays for interpolation
        y_coarse = np.arange(Hc, dtype=np.float32)
        x_coarse = np.arange(Wc, dtype=np.float32)
        y_fine = np.linspace(0, Hc - 1, H, dtype=np.float32)
        x_fine = np.linspace(0, Wc - 1, W, dtype=np.float32)
        
        # Create meshgrid for fine coordinates
        Y_fine, X_fine = np.meshgrid(y_fine, x_fine, indexing='ij')
        
        # Find integer indices and fractional parts for bilinear interpolation
        y_idx = np.floor(Y_fine).astype(np.int32)
        x_idx = np.floor(X_fine).astype(np.int32)
        y_frac = Y_fine - y_idx
        x_frac = X_fine - x_idx
        
        # Clamp indices to valid range
        y_idx = np.clip(y_idx, 0, Hc - 1)
        x_idx = np.clip(x_idx, 0, Wc - 1)
        y_idx_next = np.clip(y_idx + 1, 0, Hc - 1)
        x_idx_next = np.clip(x_idx + 1, 0, Wc - 1)
        
        # Bilinear interpolation: interpolate in x first, then y
        # Top edge
        T_top_left = T_coarse[y_idx, x_idx]
        T_top_right = T_coarse[y_idx, x_idx_next]
        T_top = T_top_left * (1.0 - x_frac) + T_top_right * x_frac
        
        # Bottom edge
        T_bot_left = T_coarse[y_idx_next, x_idx]
        T_bot_right = T_coarse[y_idx_next, x_idx_next]
        T_bot = T_bot_left * (1.0 - x_frac) + T_bot_right * x_frac
        
        # Interpolate in y
        T_full = (T_top * (1.0 - y_frac) + T_bot * y_frac).astype(np.float32)
    else:
        T_full = T_coarse

    # Update wind from temperature gradients (if requested or needed for precipitation)
    if update_wind or (update_precip and (state.wind_u is None or state.wind_v is None)):
        # Downsample elevation for wind field if available
        elev_coarse = None
        if state.elevation is not None:
            elev_pad = np.pad(state.elevation.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
            elev_coarse = elev_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
        u_coarse, v_coarse = generate_wind_field(
            Hc, Wc, day_of_year=int(new_day), block_size=1, elevation=elev_coarse
        )
        if block_size > 1:
            u_full = np.repeat(np.repeat(u_coarse, block_size, axis=0), block_size, axis=1)[:H, :W]
            v_full = np.repeat(np.repeat(v_coarse, block_size, axis=0), block_size, axis=1)[:H, :W]
        else:
            u_full, v_full = u_coarse, v_coarse
    else:
        u_full, v_full = state.wind_u, state.wind_v

    # Update precipitation from wind, temperature, elevation (if requested)
    if update_precip:
        P_full, q_full = generate_precipitation(
            H, W, state.elevation,
            day_of_year=int(new_day),
            evap_coeff=evap_coeff,
            uplift_coeff=uplift_coeff,
            rain_efficiency=rain_efficiency,
            iterations=precip_iterations,
            block_size=block_size,
        )
    else:
        P_full, q_full = state.precipitation, state.humidity

    return PlanetState(
        day_of_year=new_day,
        elevation=state.elevation,
        temperature=T_full,
        wind_u=u_full,
        wind_v=v_full,
        precipitation=P_full,
        humidity=q_full,
    )


def simulate_multiple_steps(
    initial_state: PlanetState,
    total_days: float,
    step_days: float = 1.0,
    **kwargs,
) -> list[PlanetState]:
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
    current = initial_state
    n_steps = int(np.ceil(total_days / step_days))
    for _ in range(n_steps):
        dt = min(step_days, total_days - (len(states) - 1) * step_days)
        if dt <= 0:
            break
        current = simulate_step(current, days=dt, **kwargs)
        states.append(current)
    return states


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
        elevation=elevation,
        temperature=None,
        wind_u=None,
        wind_v=None,
        precipitation=None,
        humidity=None,
    )
    return simulate_step(state, days=0.0, **kwargs)


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
    heat_transport_coeff: float = 0.5,
    land_sea_contrast: float = 6.0,
    thermal_diffusion: float = 0.03,
) -> np.ndarray:
    """Evolve temperature with wind advection, land-sea effects, and heat transport.
    
    - Land-sea contrast: land heats/cools faster (coastal effects)
    - Meridional heat transport: winds carry heat poleward/equatorward
    - Longitudinal variation: day/night cycle approximated by longitude
    - Thermal diffusion: smooths temperature gradients
    
    All arrays should have shape (Hc, Wc) for correct broadcasting.
    Use regular assignment (T = T + ...) instead of in-place (T += ...) when shapes may differ.
    """
    # Validate expected shapes at start
    assert T_prev.shape == (Hc, Wc), f"T_prev shape {T_prev.shape} != ({Hc}, {Wc})"
    assert T_base.shape == (Hc, Wc), f"T_base shape {T_base.shape} != ({Hc}, {Wc})"
    # Downsample elevation
    elev_pad = np.pad(elevation.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
    elev_c = elev_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    # Use smoothed land-sea mask to reduce sharp transitions
    elev_median = float(np.median(elev_c))
    sea_mask_binary = elev_c <= elev_median
    # Smooth the transition: create fractional mask (0=sea, 1=land) with gradient
    elev_diff = elev_c - elev_median
    elev_range = float(np.max(elev_diff) - np.min(elev_diff))
    if elev_range > 1e-6:
        # Smooth transition over ~10% of elevation range
        transition_width = elev_range * 0.1
        land_fraction = np.clip((elev_diff + transition_width) / (2.0 * transition_width), 0.0, 1.0)
    else:
        land_fraction = sea_mask_binary.astype(np.float32)
    sea_mask = 1.0 - land_fraction
    land_mask = land_fraction
    
    # Get wind field for advection
    elev_for_wind = elev_c if elevation is not None else None
    u, v = generate_wind_field(Hc, Wc, day_of_year=day_of_year, block_size=1, elevation=elev_for_wind)
    
    # Base temperature from insolation
    T = T_prev.copy()
    
    # Time step in days (convert to seconds for advection)
    dt_days = max(0.01, min(days, 1.0))  # Cap at 1 day per step
    dt_sec = dt_days * 86400.0
    
    # Grid spacing (approximate, assumes equirectangular)
    R = 6.371e6  # Earth radius in meters
    dlat = np.pi / Hc  # radians per cell
    dlon = 2.0 * np.pi / Wc  # radians per cell
    dx = R * dlon * np.cos(np.linspace(-np.pi/2, np.pi/2, Hc))[:, None]  # (Hc, 1)
    dy = R * dlat  # constant
    
    # Land-sea temperature contrast (coastal effects)
    # Land heats/cools faster than ocean (higher thermal mass for water)
    # Seasonal: land warmer relative to ocean in summer, cooler in winter
    lat_rad = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    # Seasonal factor: northern hemisphere summer -> warmer land, winter -> cooler land
    season_factor = np.cos(lat_rad) * np.sin(2.0 * np.pi * day_of_year / 365.2422)  # (Hc,)
    # Use fractional mask for smoother transitions
    T_land_offset = land_sea_contrast * (land_fraction - sea_mask) * (1.0 + 0.3 * season_factor[:, None])  # (Hc, Wc)
    assert T_land_offset.shape == (Hc, Wc), f"T_land_offset shape {T_land_offset.shape} != ({Hc}, {Wc})"
    assert T.shape == (Hc, Wc), f"T shape {T.shape} != ({Hc}, {Wc})"
    T = T + T_land_offset * 0.25  # Reduced magnitude for smoother transitions
    
    # Diurnal variation removed - static diurnal cycle created unrealistic persistent hot spots
    # Temperature variation now comes from: land-sea contrast, wind advection, and thermal diffusion
    
    # Meridional heat transport (wind advection)
    # Advect temperature using upwind scheme
    wind_speed = np.sqrt(u * u + v * v) + 1e-6
    wind_ref = 12.0  # m/s reference
    
    # Zonal advection (east-west) - critical for longitudinal variation
    u_scale = np.clip(np.abs(u) / wind_ref, 0.0, 1.0)  # (Hc, Wc)
    T_east = np.roll(T, -1, axis=1)
    T_west = np.roll(T, 1, axis=1)
    T_x = np.where(u >= 0, T_west, T_east)  # (Hc, Wc)
    assert T_x.shape == (Hc, Wc), f"T_x shape {T_x.shape} != ({Hc}, {Wc})"
    # Increase zonal advection strength to create more longitudinal variation
    zonal_coeff = heat_transport_coeff * 1.5  # Stronger zonal transport
    T = T + zonal_coeff * u_scale * (T_x - T) * dt_days
    
    # Meridional advection (north-south)
    v_scale = np.clip(np.abs(v) / wind_ref, 0.0, 1.0)  # (Hc, Wc)
    T_north = np.roll(T, -1, axis=0)
    T_south = np.roll(T, 1, axis=0)
    T_y = np.where(v >= 0, T_south, T_north)  # (Hc, Wc)
    assert T_y.shape == (Hc, Wc), f"T_y shape {T_y.shape} != ({Hc}, {Wc})"
    T = T + heat_transport_coeff * v_scale * (T_y - T) * dt_days
    
    # Thermal diffusion (smooth temperature gradients)
    # Apply multiple diffusion passes for better smoothing
    for _ in range(2):  # Two passes for smoother gradients
        T_pad = np.pad(T, 1, mode="edge")
        T_lap = T_pad[0:-2, 1:-1] + T_pad[2:, 1:-1] + T_pad[1:-1, 0:-2] + T_pad[1:-1, 2:] - 4.0 * T  # (Hc, Wc)
        assert T_lap.shape == (Hc, Wc), f"T_lap shape {T_lap.shape} != ({Hc}, {Wc})"
        T = T + thermal_diffusion * T_lap * dt_days
    
    # Relax toward base temperature (insolation equilibrium) - but only for large deviations
    # Land relaxes faster (smaller thermal mass) - use fractional mask for smooth transition
    # Only relax extreme temperatures, preserving longitudinal variation
    relax_rate = (0.03 + 0.02 * land_fraction).astype(np.float32)  # (Hc, Wc) - reduced rates
    # Only relax if temperature deviates significantly from base (preserve longitudinal variation)
    T_deviation = T - T_base  # Deviation from base
    T_deviation_mag = np.abs(T_deviation)
    # Only relax if deviation > 5K (allows longitudinal variation to persist)
    relax_mask = T_deviation_mag > 5.0
    relax_rate = relax_rate * relax_mask.astype(np.float32)
    assert relax_rate.shape == (Hc, Wc), f"relax_rate shape {relax_rate.shape} != ({Hc}, {Wc})"
    assert T.shape == (Hc, Wc), f"T shape {T.shape} != ({Hc}, {Wc})"
    assert T_base.shape == (Hc, Wc), f"T_base shape {T_base.shape} != ({Hc}, {Wc})"
    # Use regular assignment - only relax extreme deviations
    T = T - relax_rate * T_deviation * dt_days
    
    # Clamp to reasonable range
    T = np.clip(T, 180.0, 350.0)
    
    return T.astype(np.float32)

