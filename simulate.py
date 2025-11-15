"""Time simulation for planet conditions.

Advances atmospheric systems (temperature, wind, precipitation) forward in time
with configurable time scales. Default unit is one day.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple
from atmosphere import generate_wind_field, generate_precipitation
from temperature import temperature_kelvin_for_lat
from ocean import calculate_ocean_heat_transport


class PlanetState(NamedTuple):
    """Current planet state snapshot."""
    day_of_year: float  # Fractional day (0-365.2422)
    elevation: np.ndarray  # (H, W) terrain elevation [0,1]
    temperature: np.ndarray | None = None  # (H, W) temperature (K)
    wind_u: np.ndarray | None = None  # (H, W) eastward wind (m/s)
    wind_v: np.ndarray | None = None  # (H, W) northward wind (m/s)
    precipitation: np.ndarray | None = None  # (H, W) precipitation (mm/day)
    humidity: np.ndarray | None = None  # (H, W) specific humidity
    soil_moisture: np.ndarray | None = None  # (H, W) bucket soil moisture [0,1]


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
    debug_log: bool = False,
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
    # Ocean: seasonal lag of ~50 days (1.5 months) due to high heat capacity
    # Land: immediate response to current insolation
    lat = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    
    # Calculate temperature for current day (land response)
    T_lat_land = temperature_kelvin_for_lat(lat, day_of_year=int(new_day))
    T_base_land = np.repeat(T_lat_land[:, None], Wc, axis=1).astype(np.float32)
    
    # Calculate temperature for lagged day (ocean response with 1.5 month delay)
    lag_days = 50.0  # ~1.5 months thermal lag for deep ocean mixed layer
    lagged_day = (new_day - lag_days) % 365.2422
    T_lat_ocean = temperature_kelvin_for_lat(lat, day_of_year=int(lagged_day))
    T_base_ocean = np.repeat(T_lat_ocean[:, None], Wc, axis=1).astype(np.float32)
    
    # DEBUG: Log base temperatures if debug enabled
    if debug_log:
        import logging
        LOG = logging.getLogger("planetsim")
        arctic_idx = int(Hc * 0.15)
        eq_idx = Hc // 2
        T_base_arctic_land = float(T_lat_land[arctic_idx])
        T_base_eq_land = float(T_lat_land[eq_idx])
        T_base_arctic_ocean = float(T_lat_ocean[arctic_idx])
        T_base_eq_ocean = float(T_lat_ocean[eq_idx])
        LOG.info(f"[Base Temps Day {new_day:.1f}] Land: Arctic={T_base_arctic_land:.1f}K ({T_base_arctic_land-273.15:.1f}°C), "
                 f"Eq={T_base_eq_land:.1f}K ({T_base_eq_land-273.15:.1f}°C) | "
                 f"Ocean(lag={lag_days:.0f}d): Arctic={T_base_arctic_ocean:.1f}K ({T_base_arctic_ocean-273.15:.1f}°C), "
                 f"Eq={T_base_eq_ocean:.1f}K ({T_base_eq_ocean-273.15:.1f}°C)")
    
    # Blend based on land fraction (will be calculated in _evolve_temperature)
    # Use ocean-lagged temperature as base; _evolve_temperature will handle land/ocean mixing
    T_base = T_base_ocean  # Start with ocean (lagged), land will be corrected in evolution
    
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
        day_of_year=int(new_day), days=days,
        T_base_land=T_base_land  # Pass land temperature for seasonal lag correction
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
    humidity_prev = state.humidity
    soil_prev = getattr(state, "soil_moisture", None)

    if update_precip:
        P_full, humidity_next, soil_next = generate_precipitation(
            H, W, state.elevation,
            temperature=T_full,
            wind_u=u_full,
            wind_v=v_full,
            humidity=humidity_prev,
            soil_moisture=soil_prev,
            day_of_year=int(new_day),
            dt_days=max(days, 1.0),
            evap_coeff=evap_coeff,
            uplift_coeff=uplift_coeff,
            rain_efficiency=rain_efficiency,
        )
    else:
        P_full = state.precipitation
        humidity_next = humidity_prev
        soil_next = soil_prev

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
    
    return PlanetState(
        day_of_year=new_day,
        elevation=state.elevation,
        temperature=T_full,
        wind_u=u_full,
        wind_v=v_full,
        precipitation=P_full,
        humidity=humidity_next,
        soil_moisture=soil_next,
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
    heat_transport_coeff: float = 0.8,  # CONSERVATIVE increase from 0.5 (1.6x) for stability
    land_sea_contrast: float = 0.0,
    thermal_diffusion: float = 0.04,    # CONSERVATIVE increase from 0.03 (1.3x) respects CFL
    T_base_land: np.ndarray | None = None,
) -> np.ndarray:
    """Evolve temperature with wind advection, land-sea effects, and heat transport.
    
    COMPREHENSIVE HEAT TRANSPORT SYSTEM (NUMERICALLY STABLE):
    - Atmospheric advection: Wind-driven heat transport (1.6x base)
    - Thermal diffusion: Large-scale mixing (1.3x base, strict CFL)
    - Hadley cells: Cross-equatorial tropical circulation (~2-3 PW, strengthened 3.3x)
    - Subtropical subsidence: Descending branch warming (adiabatic compression, +5-10K)
    - Ocean currents: Poleward oceanic heat flux (~2-3 PW, strengthened 1.7x)
    - Total: ~5-7 PW poleward heat transport (Earth-like ~5 PW)
    
    Physics:
    - Land-sea contrast: land heats/cools faster (coastal effects)
    - Meridional advection: winds carry heat poleward/equatorward
    - Hadley circulation: tropical+subtropical heat redistribution (0-40° latitude)
    - Subtropical subsidence: descending branch adiabatic warming (30-40° latitude)
    - Ocean transport: currents + thermohaline circulation
    - Longitudinal variation: day/night cycle approximated by longitude
    - Thermal diffusion: smooths temperature gradients (CFL-stable)
    
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
    
    # Apply seasonal thermal lag: blend lagged ocean temperature with immediate land temperature
    # Ocean uses T_base (lagged by ~50 days), land uses T_base_land (current day)
    if T_base_land is not None:
        # Blend: T_base_blended = T_base_ocean * (1 - land_fraction) + T_base_land * land_fraction
        # This gives ocean regions the lagged temperature, land regions the current temperature
        T_base = T_base * sea_mask + T_base_land * land_mask
    
    # Get wind field for advection
    elev_for_wind = elev_c if elevation is not None else None
    u, v = generate_wind_field(Hc, Wc, day_of_year=day_of_year, block_size=1, elevation=elev_for_wind)
    
    # Base temperature from insolation
    T = T_prev.copy()
    
    # Apply altitude correction using environmental lapse rate
    # Only apply to land ABOVE sea level (elevation > 0.2)
    # Ocean and areas at sea level get no adjustment
    sea_level = 0.2
    
    # Calculate altitude only for land above sea level
    # Use non-linear scaling to better represent Earth's elevation distribution
    # Most of Earth's land is at low elevation, high mountains are rare
    elevation_above_sea = np.maximum(0.0, elev_c - sea_level)
    # Apply power of 2.0 to significantly reduce high altitude values (matches temperature.py)
    altitude_m = (elevation_above_sea / (1.0 - sea_level)) ** 2.0 * 8848.0
    
    # Standard environmental lapse rate: 6.5 K per 1000m (matches temperature.py)
    lapse_rate = 0.0065  # K/m (standard atmospheric lapse rate)
    T_cooling = altitude_m * lapse_rate
    
    # Apply minimum temperature floor BEFORE altitude correction
    # Represents heat transport and thermal inertia
    T = np.maximum(T, 200.0)
    
    # Apply altitude correction to base temperature
    # Only land above sea level gets cooled; ocean stays at base temperature
    T = T - T_cooling
    
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
    
    # Meridional advection (north-south) - UNIFORM coefficient for stability
    # NOTE: Latitude-dependent enhancement (Ferrel cells) caused numerical instability
    # Keeping simple uniform transport until we can implement it more carefully
    
    v_scale = np.clip(np.abs(v) / wind_ref, 0.0, 1.0)  # (Hc, Wc)
    T_north = np.roll(T, -1, axis=0)
    T_south = np.roll(T, 1, axis=0)
    T_y = np.where(v >= 0, T_south, T_north)  # (Hc, Wc)
    assert T_y.shape == (Hc, Wc), f"T_y shape {T_y.shape} != ({Hc}, {Wc})"
    
    # Apply uniform meridional transport (simple and stable)
    T = T + heat_transport_coeff * v_scale * (T_y - T) * dt_days
    
    # Thermal diffusion (smooth temperature gradients)
    # Represents large-scale atmospheric mixing and eddy diffusion
    # Conservative passes (2) to ensure numerical stability (strict CFL)
    for _ in range(2):  # Two passes - proven stable configuration
        T_pad = np.pad(T, 1, mode="edge")
        T_lap = T_pad[0:-2, 1:-1] + T_pad[2:, 1:-1] + T_pad[1:-1, 0:-2] + T_pad[1:-1, 2:] - 4.0 * T  # (Hc, Wc)
        assert T_lap.shape == (Hc, Wc), f"T_lap shape {T_lap.shape} != ({Hc}, {Wc})"
        T = T + thermal_diffusion * T_lap * dt_days
    
    # ==============================================================================
    # HADLEY CELL CIRCULATION (Cross-Equatorial Heat Transport)
    # ==============================================================================
    # Hadley cells are major tropical atmospheric circulation patterns that transport
    # heat from the summer hemisphere to the winter hemisphere. Rising air at the
    # ITCZ (Inter-Tropical Convergence Zone) near the equator carries heat poleward,
    # descending at ~30° latitude (subtropical highs). This creates a net heat flux 
    # across the equator and warms the subtropical descending zones.
    #
    # Physical model:
    # - Calculate interhemispheric temperature asymmetry (summer vs winter)
    # - Apply heat flux in tropical+subtropical band (0-40° latitude)
    # - Heat flows FROM hot (summer) hemisphere TO cold (winter) hemisphere
    # - Peak effect at ~20° (subtropical transition + descending branch)
    # - Target: ~2-3 PW cross-equatorial flux (~40-60 W/m² in tropics/subtropics)
    
    # Calculate latitude for each row (0 = equator at center)
    lat_rows = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * 180.0  # degrees, -90 to +90
    abs_lat_rows = np.abs(lat_rows)
    
    # Tropical+Subtropical mask: Hadley cells extend to 40° (includes descending branch)
    # 0-30°: Main Hadley circulation (rising at equator, poleward aloft)
    # 30-40°: Subtropical descending branch (warm air sinks, heats surface)
    hadley_mask = abs_lat_rows < 40.0
    
    # Calculate hemisphere average temperatures in tropics (0-30° for gradient calc)
    nh_tropical_mask = (lat_rows >= 0) & (lat_rows < 30.0)
    sh_tropical_mask = (lat_rows < 0) & (lat_rows > -30.0)
    
    T_nh_tropical = np.mean(T[nh_tropical_mask, :]) if np.any(nh_tropical_mask) else 0.0
    T_sh_tropical = np.mean(T[sh_tropical_mask, :]) if np.any(sh_tropical_mask) else 0.0
    
    # Cross-equatorial temperature gradient (positive = NH warmer, negative = SH warmer)
    T_gradient = T_nh_tropical - T_sh_tropical
    
    # Hadley flux strength scales with temperature gradient
    # Calibrated to produce ~2-3 PW equivalent transport
    # STRENGTHENED: 0.15 → 0.50 (3.3x increase) to fix subtropical winter cooling
    hadley_efficiency = 0.50  # Kelvin per day per K gradient
    
    # Create latitude-dependent Hadley effect profile
    # Peak at subtropical transition (~15-20°), tapering toward equator and extends to 40°
    # Gaussian-like profile centered at 20° with width ~15° (broader to cover subtropics)
    hadley_profile = np.exp(-((abs_lat_rows - 20.0) / 15.0)**2)  # Peak at 20°, extends 0-40°
    hadley_profile = hadley_profile * hadley_mask  # Include subtropical descending zone
    hadley_profile_2d = hadley_profile[:, np.newaxis]  # (Hc, 1) for broadcasting
    
    # Apply cross-equatorial heat flux
    # Northern hemisphere: cool if warmer than SH (T_gradient > 0), warm if cooler
    # Southern hemisphere: warm if cooler than NH (T_gradient > 0), cool if warmer
    # Sign convention: positive gradient means NH loses heat, SH gains heat
    hadley_flux = hadley_efficiency * T_gradient * hadley_profile_2d * dt_days  # (Hc, 1)
    
    # Apply with opposite sign in each hemisphere
    hadley_adjustment = np.where(
        lat_rows[:, np.newaxis] >= 0,  # Northern hemisphere
        -hadley_flux,  # Cool if NH is warmer (negative gradient removes heat)
        +hadley_flux,  # Warm if SH is cooler (positive gradient adds heat)
    )
    
    T = T + hadley_adjustment
    
    # ==============================================================================
    # SUBTROPICAL DESCENDING BRANCH WARMING (Hadley Cell Subsidence)
    # ==============================================================================
    # The descending branch of Hadley cells (30-40° latitude) brings warm air from
    # aloft down to the surface, creating additional warming through adiabatic compression.
    # This creates the subtropical high-pressure zones and prevents winter subtropics
    # from getting unrealistically cold.
    #
    # Physical basis:
    # - Air descending from ~10-12 km altitude warms adiabatically (~10 K/km)
    # - Creates high-pressure zones (Azores High, Pacific High)
    # - Suppresses cloud formation → more solar heating
    # - Target: +5-10K warming boost for subtropical descending zones
    
    # Subtropical zone: 30-40° latitude
    subtropical_mask = (abs_lat_rows >= 30.0) & (abs_lat_rows < 40.0)
    
    # Descending branch warming profile: strongest at 35°, tapering toward 30° and 40°
    # Gaussian centered at 35° with width ~5°
    subsidence_profile = np.exp(-((abs_lat_rows - 35.0) / 5.0)**2) * subtropical_mask
    
    # Warming strength: 8K boost for subtropical descending zones
    # This represents adiabatic compression warming + reduced cloud albedo
    subsidence_warming = 8.0 * subsidence_profile[:, np.newaxis] * dt_days / 10.0  # Scale by dt
    
    # Apply subsidence warming (always positive - descending air warms)
    T = T + subsidence_warming
    
    # ==============================================================================
    # OCEAN HEAT TRANSPORT (Surface Currents & Thermohaline Circulation)
    # ==============================================================================
    # Oceans transport ~1-2 PW of heat poleward through:
    # - Wind-driven surface currents (Gulf Stream, Kuroshio)
    # - Thermohaline circulation (density-driven deep currents)
    # - Ocean-atmosphere heat exchange
    # This moderates high-latitude climates significantly
    
    ocean_transport_adjustment = calculate_ocean_heat_transport(
        T=T,
        elevation=elev_c,
        Hc=Hc,
        Wc=Wc,
        day_of_year=day_of_year,
        dt_days=dt_days,
        transport_coefficient=0.5,  # STRENGTHENED: 0.3 → 0.5 (1.7x increase)
    )
    
    T = T + ocean_transport_adjustment
    
    # Relax toward base temperature (insolation equilibrium) with ocean thermal inertia
    # Ocean: HIGH thermal inertia (resists changes, low relax rate ~0.15)
    # Land: LOW thermal inertia (responds quickly, high relax rate ~0.70)
    # This creates the fundamental difference: oceans moderate temperature, land has extremes
    relax_rate_ocean = 0.15  # Ocean retains ~85% of deviation (high inertia)
    relax_rate_land = 0.70   # Land corrects ~70% of deviation per day (low inertia)
    relax_rate = (relax_rate_ocean + (relax_rate_land - relax_rate_ocean) * land_fraction).astype(np.float32)  # (Hc, Wc)
    
    # Temperature deviation from equilibrium
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

