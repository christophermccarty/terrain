"""Time simulation for planet conditions.

Advances atmospheric systems (temperature, wind, precipitation) forward in time
with configurable time scales. Default unit is one day.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple
from atmosphere import generate_wind_field
from temperature import temperature_kelvin_for_lat, elevation_to_alt_km
from ocean import calculate_ocean_heat_transport

# Cache for diagnostic/relaxation wind to avoid recomputing every step.
_RELAX_CACHE = {"key": None, "u": None, "v": None}


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


def simulate_step(
    state: PlanetState,
    days: float = 1.0,
    *,
    block_size: int = 3,
    wind_block_size: int | None = None,
    update_wind: bool = True,
    wind_relax: float = 0.0,
    wind_target_weather_amp: float = 0.0,
    wind_target_zonal_pressure: float = 1.0,
    wind_target_terrain_pressure_amp: float = 0.25,
    wind_target_terrain_flow_amp: float = 0.25,
    wind_pgf_temp_scale: float = 450.0,
    wind_pgf_terrain_scale: float = 900.0,
    wind_drag_base: float = 2.0e-7,
    wind_drag_elev_scale: float = 6.0e-7,
    wind_damping: float = 0.25,
    # Baroclinic eddy / thermal-wind proxy. The previous default (3e7) tends to produce
    # unrealistically strong, planet-wide surface jets. Keep conservative by default.
    wind_baroclinic_jet_amp: float = 0.0,
    wind_baroclinic_mix: float = 2.0,
    debug_log: bool = False,
    track_components: bool = False,
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
    new_day = (state.day_of_year + days) % 365.2422
    new_total_days = float(state.total_days) + float(days)
    H, W = state.elevation.shape
    Hc, Wc = (max(1, (H + block_size - 1) // block_size),
              max(1, (W + block_size - 1) // block_size))
    wind_bs = max(1, int(block_size if wind_block_size is None else wind_block_size))
    Hcw, Wcw = (max(1, (H + wind_bs - 1) // wind_bs),
                max(1, (W + wind_bs - 1) // wind_bs))

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
    # Full-resolution fallback base temperature (used when wind evolves at full resolution
    # before `state.temperature` is initialized).
    lat_full = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
    T_lat_ocean_full = temperature_kelvin_for_lat(lat_full, day_of_year=int(lagged_day))
    T_base_ocean_full = np.repeat(T_lat_ocean_full[:, None], W, axis=1).astype(np.float32)
    
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
    from atmosphere import evolve_wind
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
            T_lat_ocean_w = temperature_kelvin_for_lat(lat_w, day_of_year=int(lagged_day))
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
            baroclinic_jet_amp=float(wind_baroclinic_jet_amp),
            baroclinic_mix=float(wind_baroclinic_mix),
        )

        # Keep winds energized + seasonally varying by weakly relaxing toward a diagnostic wind
        # (generate_wind_field injects synoptic-scale "weather systems" seeded by day_of_year).
        if wind_relax > 0.0:
            u_diag, v_diag = _diag_wind_cached(Hcw, Wcw, T_for_wind, elev_c_w)
            a = float(np.clip(wind_relax, 0.0, 1.0))
            u_coarse_evol = (1.0 - a) * u_coarse_evol + a * u_diag
            v_coarse_evol = (1.0 - a) * v_coarse_evol + a * v_diag
        
        # Upsample back to full resolution using bilinear interpolation
        from atmosphere import _upsample_bilinear
        u_full = _upsample_bilinear(u_coarse_evol, H, W, wind_bs)
        v_full = _upsample_bilinear(v_coarse_evol, H, W, wind_bs)
    else:
        # Full resolution evolution
        # If wind evolves at higher resolution than the temperature solver, drive it with the
        # coarse temperature field upsampled to full resolution. This avoids injecting
        # grid-scale temperature noise into the wind solver (which can blow up speeds),
        # while still allowing the wind numerics to run on the fine grid.
        T_wind_full = state.temperature if state.temperature is not None else T_base_ocean_full
        elev_wind_full = state.elevation
        if wind_bs < block_size and block_size > 1:
            from atmosphere import _upsample_bilinear
            if state.temperature is not None:
                T_wind_full = _upsample_bilinear(T_prev_coarse, H, W, block_size)
            else:
                T_wind_full = _upsample_bilinear(T_base, H, W, block_size)
            if elev_c is not None:
                elev_wind_full = _upsample_bilinear(elev_c, H, W, block_size)

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
            baroclinic_jet_amp=float(wind_baroclinic_jet_amp),
            baroclinic_mix=float(wind_baroclinic_mix),
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
    # Always track components for diagnostics (minimal overhead)
    T_coarse, cloud_c, snow_c, temp_components = _evolve_temperature(
        T_prev_coarse, T_base, state.elevation, Hc, Wc, block_size, H, W,
        day_of_year=int(new_day), days=days,
        wind_u=u_coarse, wind_v=v_coarse, # Pass evolved wind
        T_base_land=T_base_land,  # Pass land temperature for seasonal lag correction
        track_components=track_components  # Track components for diagnostics
    )
    
    # Upsample components to full resolution if needed
    if block_size > 1 and temp_components:
        from atmosphere import _upsample_bilinear
        temp_components_full = {}
        for name, field in temp_components.items():
            if isinstance(field, np.ndarray) and field.shape == (Hc, Wc):
                temp_components_full[name] = _upsample_bilinear(field, H, W, block_size)
            else:
                # Scalar or already full resolution
                temp_components_full[name] = field
        temp_components = temp_components_full
    
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
        
        # Bilinear interpolation helper
        def interp(f):
            top = f[y_idx, x_idx] * (1.0 - x_frac) + f[y_idx, x_idx_next] * x_frac
            bot = f[y_idx_next, x_idx] * (1.0 - x_frac) + f[y_idx_next, x_idx_next] * x_frac
            return top * (1.0 - y_frac) + bot * y_frac

        T_full = interp(T_coarse).astype(np.float32)
        # Interpolate diagnostics
        cloud_full = interp(cloud_c).astype(np.float32)
        snow_full = interp(snow_c).astype(np.float32)
    else:
        T_full = T_coarse
        cloud_full = cloud_c
        snow_full = snow_c

    # Update wind from temperature gradients (if requested)
    # Already evolved above
    # if update_wind... removed legacy block
    
    # Precipitation simulation removed (for performance / simplicity).
    P_full = None
    humidity_next = None
    soil_next = None

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
        snow_depth=snow_full,
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
    track_components: bool = False,
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
    # Downsample elevation
    elev_pad = np.pad(elevation.astype(np.float32), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
    elev_c = elev_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    elev_median = float(np.median(elev_c))
    
    # Land/Sea Masks
    sea_mask = elev_c <= elev_median
    land_mask = ~sea_mask
    land_fraction = land_mask.astype(np.float32) # Simplified for now
    
    # 2. Wind Field (Prognostic or Diagnostic)
    if wind_u is None or wind_v is None:
        u, v = generate_wind_field(Hc, Wc, day_of_year=day_of_year, block_size=1, elevation=elev_c)
    else:
        u, v = wind_u, wind_v
        
    dt = days * 86400.0
    
    # 3. Thermodynamic Energy Balance
    # dT/dt = Advection + Radiation + Phase Change + Diffusion
    
    T = T_prev.copy()
    
    # --- Advection (Wind Transport) ---
    # Zonal - with stability check to prevent extreme gradients
    # FURTHER REDUCED to prevent regional hot spots and reduce oscillations
    u_ref = 15.0
    u_scale = np.clip(np.abs(u) / u_ref, 0, 1.0)
    T_east = np.roll(T, -1, axis=1)
    T_west = np.roll(T, 1, axis=1)
    T_x = np.where(u >= 0, T_west, T_east)
    # Limit advection to prevent extreme temperature jumps
    T_diff_x = np.clip(T_x - T, -12.0, 12.0)  # Reduced from ±15K to ±12K
    # Reduce advection strength further to prevent hot spots
    T = T + 0.4 * heat_transport_coeff * u_scale * T_diff_x * days  # Reduced from 0.5 to 0.4
    
    # Meridional - with stability check
    v_scale = np.clip(np.abs(v) / u_ref, 0, 1.0)
    T_north = np.roll(T, -1, axis=0)
    T_south = np.roll(T, 1, axis=0)
    T_y = np.where(v >= 0, T_south, T_north)
    # Limit advection to prevent extreme temperature jumps
    T_diff_y = np.clip(T_y - T, -12.0, 12.0)  # Reduced from ±15K to ±12K
    # Reduce advection strength further to prevent hot spots
    T = T + 0.4 * heat_transport_coeff * v_scale * T_diff_y * days  # Reduced from 0.5 to 0.4
    
    # --- Diffusion (Mixing) ---
    # Increased diffusion to reduce regional hot spots and improve stability
    # More iterations and higher coefficient to smooth out asymmetries
    for _ in range(3):  # Increased from 2 to 3 iterations
        # Periodic longitude (axis=1), clamped poles (axis=0)
        T_pad = np.pad(T, ((1, 1), (0, 0)), mode="edge")
        c = T_pad[1:-1, :]
        n = T_pad[0:-2, :]
        s = T_pad[2:, :]
        e = np.roll(c, -1, axis=1)
        w = np.roll(c, 1, axis=1)
        T_lap = n + s + e + w - 4.0 * c
        # Clamp laplacian to prevent extreme smoothing
        T_lap = np.clip(T_lap, -30.0, 30.0)  # Max ±30K smoothing per iteration
        T = T + thermal_diffusion * 1.2 * T_lap * days  # Increased effective diffusion by 20%
        
    # --- Radiative Balance (Physics Item 1, 2, 10, 12) ---
    # Incoming Solar (S_in) - Albedo (A)
    # A depends on: Land/Ocean, Snow/Ice, Cloud
    
    # Approx Latitude
    lat = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    lat_2d = np.repeat(lat[:, None], Wc, axis=1)
    
    # Ice/Snow Cover approximation based on Temp
    # T < 271 K -> Snow/Ice likely
    snow_cover = np.clip((273.15 - T) / 10.0, 0.0, 1.0)
    
    # Cloud Cover approximation (RH proxy)
    # Warmer air holds more water, but assume constant RH availability for now
    # Cloud fraction increases with uplift/convergence (simplified)
    cloud_fraction = 0.4 + 0.2 * np.sin(lat_2d * 3.0) # Base climatology
    
    # Albedo
    # Ocean: 0.06, Land: 0.2, Snow: 0.8, Cloud: 0.5
    albedo_sfc = np.where(sea_mask, 0.06, 0.2)
    albedo_sfc = albedo_sfc * (1 - snow_cover) + 0.8 * snow_cover
    albedo_total = albedo_sfc * (1 - cloud_fraction) + 0.5 * cloud_fraction
    
    # Insolation Q (Daily mean) - use proper astronomical calculation
    # Use Earth's obliquity (23.44°) not 0.4 radians
    obliq = np.deg2rad(23.44)
    gamma = 2.0 * np.pi * (day_of_year - 80.0) / 365.2422
    decl = np.arcsin(np.sin(obliq) * np.sin(gamma))
    
    # Clamp to avoid domain errors in polar regions
    lat_safe = np.clip(lat_2d, -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
    cos_h = np.clip(-np.tan(lat_safe) * np.tan(decl), -1.0, 1.0)
    h = np.arccos(cos_h) # hour angle radians (0 to pi)
    h = np.where(cos_h <= -1.0, np.pi, h)  # 24h daylight
    h = np.where(cos_h >= 1.0, 0.0, h)     # polar night
    
    Q = 1361.0 * (1.0/np.pi) * (h * np.sin(lat_safe)*np.sin(decl) + np.cos(lat_safe)*np.cos(decl)*np.sin(h))
    Q = np.maximum(0.0, Q)
    
    S_absorbed = Q * (1.0 - albedo_total)
    
    # Outgoing Longwave (L_out) = sigma * T^4 * epsilon
    # Greenhouse effect reduces OLR - use latitude-dependent epsilon (like temperature.py)
    abs_lat_deg = np.rad2deg(np.abs(lat_2d))
    epsilon_equator = 0.78  # Increased from 0.75 to match temperature.py and warm global mean
    epsilon_pole = 0.55     # Increased from 0.50 to reduce polar extremes
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
    # Use weighted blend: 95% base (which has proper physics) + 5% radiation-only
    # Increased from 90/10 to 95/5 to better preserve correct base temperatures and warm global mean
    T_eq = 0.95 * T_base + 0.05 * T_eq_rad

    # --- Orographic cooling (lapse rate) ---
    # Previously missing: high terrain never cooled as a function of altitude.
    # Apply to equilibrium temperature so radiation relaxes toward a colder state aloft.
    lapse_rate = 6.5  # K/km
    alt_km = elevation_to_alt_km(elev_c)
    T_eq = T_eq - lapse_rate * alt_km
    
    # Relaxation rate k (1/days)
    # Ocean: Slow (0.05/day = 20 day time constant) for better thermal inertia
    # Land: Fast (1.0/day = 1 day time constant)
    # REDUCED ocean rate from 0.1 to 0.05 to increase thermal inertia and reduce oscillations
    k_relax = np.where(sea_mask, 0.05, 1.0)
    
    # Apply Radiative forcing via relaxation
    # Limit relaxation to prevent extreme jumps and reduce oscillations
    T_change = k_relax * (T_eq - T) * days
    T_change = np.clip(T_change, -8.0, 8.0)  # Reduced from ±10K to ±8K per day to reduce oscillations
    T = T + T_change
    
    # --- Latent & Sensible Heat (Physics Item 3, 11) ---
    # Enhanced: Stronger cooling for very hot regions to prevent extreme temperatures
    # Evap heat loss ~ proportional to T (Clausius-Clapeyron)
    # Cooling = L * E
    # Base cooling reduced, but enhanced for hot regions (>30°C = 303K)
    base_evap = np.where(sea_mask, 0.01 * (T - 270.0), 0.0)
    hot_evap = np.where((T > 303.0) & sea_mask, 0.03 * (T - 303.0), 0.0)  # Extra cooling for hot regions
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
        land_blend = np.where(land_mask, 0.3, 0.0)  # 30% pull toward base for land
        T = (1.0 - land_blend) * T + land_blend * T_base_land

    # --- Ocean Transport (Keep existing) ---
    T_ocean_adj = calculate_ocean_heat_transport(
        T, elev_c, Hc, Wc, day_of_year, days, transport_coefficient=0.5
    )
    # Clamp ocean transport to prevent extreme adjustments
    T_ocean_adj = np.clip(T_ocean_adj, -10.0, 10.0)  # Max ±10K per day
    T = T + T_ocean_adj

    # --- Hadley/Subsidence (Keep existing simplified param) ---
    # (Ideally this emerges from dynamics, but keeping parameterization for stability)
    # Recalculate Hadley simply - REDUCED from 5K to 0.5K per day (was too strong)
    lat_deg = np.rad2deg(np.abs(lat_2d))
    subsidence = 0.5 * np.exp(-((lat_deg - 30.0)/10.0)**2) * days  # Reduced 10x
    T = T + subsidence

    # --- FINAL TEMPERATURE CLAMPING (Critical for stability) ---
    # Clamp to realistic Earth-like temperature range
    # Absolute minimum: -33°C (240K) - typical polar winter (matches T_min in temperature.py)
    # Absolute maximum: +50°C (323K) - reduced from 60°C to prevent extreme hot spots
    T = np.clip(T, 240.0, 323.0)  # Reduced maximum from 333K to 323K to prevent extreme temperatures

    # Track component contributions if requested
    components = {}
    if track_components:
        # Calculate what each component contributed (in K change)
        T_after_advection = T_prev.copy()
        T_after_advection = T_after_advection + heat_transport_coeff * u_scale * T_diff_x * days
        T_after_advection = T_after_advection + heat_transport_coeff * v_scale * T_diff_y * days
        components['advection'] = T_after_advection - T_prev
        
        T_after_diffusion = T_after_advection.copy()
        for _ in range(2):
            T_pad = np.pad(T_after_diffusion, ((1, 1), (0, 0)), mode="edge")
            c = T_pad[1:-1, :]
            n = T_pad[0:-2, :]
            s = T_pad[2:, :]
            e = np.roll(c, -1, axis=1)
            w = np.roll(c, 1, axis=1)
            T_lap = n + s + e + w - 4.0 * c
            T_lap = np.clip(T_lap, -30.0, 30.0)
            T_after_diffusion = T_after_diffusion + thermal_diffusion * T_lap * days
        components['diffusion'] = T_after_diffusion - T_after_advection
        
        T_after_radiation = T_after_diffusion.copy()
        T_after_radiation = T_after_radiation + k_relax * (T_eq - T_after_radiation) * days
        components['radiation'] = T_after_radiation - T_after_diffusion
        
        T_after_evap = T_after_radiation.copy()
        T_after_evap = T_after_evap - evap_cooling * days
        components['evaporation'] = T_after_evap - T_after_radiation
        
        components['ocean_transport'] = T_ocean_adj
        components['subsidence'] = subsidence
        components['equilibrium_temp'] = T_eq
        components['net_radiation'] = R_net

    return T.astype(np.float32), cloud_fraction.astype(np.float32), snow_cover.astype(np.float32), components


