"""Ocean circulation and thermal transport modeling.

This module handles oceanic heat transport through:
1. Wind-driven surface currents (e.g., Gulf Stream, Kuroshio)
2. Thermohaline circulation (density-driven deep ocean currents)
3. Ocean-atmosphere heat exchange

Ocean currents on Earth transport ~1-2 PW of heat poleward, comparable to
atmospheric transport. This is critical for moderating high-latitude climates.
"""

import numpy as np


def _ocean_mask_from_elevation(elevation: np.ndarray, *, assume_loaded_if_zeros_frac: float = 0.05) -> np.ndarray:
    """Return boolean ocean mask from normalized elevation.

    - Loaded DEMs in this project encode ocean as exactly 0.0.
    - Procedural terrain does not, so we fall back to a median sea level split
      (matching the rest of the simulation).
    """
    elev = np.asarray(elevation, dtype=np.float32)
    zeros_frac = float(np.mean(elev == 0.0)) if elev.size else 0.0
    if zeros_frac > assume_loaded_if_zeros_frac:
        return elev == 0.0
    elev_median = float(np.median(elev))
    return elev <= elev_median


def update_sea_ice(
    T: np.ndarray,
    elevation: np.ndarray,
    prev_ice: np.ndarray | None,
    dt_days: float,
    *,
    freeze_temp: float = 269.5,
    melt_temp: float = 273.35,
    freeze_rate: float = 0.06,
    melt_rate: float = 0.16,
) -> np.ndarray:
    """Update sea ice fraction based on persistent cold ocean temperatures."""
    is_ocean = _ocean_mask_from_elevation(elevation)
    ice = np.zeros_like(T, dtype=np.float32) if prev_ice is None else np.asarray(prev_ice, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    freeze = (T <= freeze_temp) & is_ocean
    melt = (T >= melt_temp) & is_ocean
    if np.any(freeze):
        ice = np.where(freeze, np.minimum(1.0, ice + freeze_rate * dt_days), ice)
    if np.any(melt):
        ice = np.where(melt, np.maximum(0.0, ice - melt_rate * dt_days), ice)
    return np.where(is_ocean, ice, 0.0)


def calculate_ocean_heat_transport(
    T: np.ndarray,
    elevation: np.ndarray,
    Hc: int,
    Wc: int,
    day_of_year: int,
    dt_days: float,
    transport_coefficient: float = 0.3,
    exchange_strength_floor: float = 0.65,
    exchange_strength_span: float = 0.35,
    exchange_coefficient: float = 0.05,
    exchange_inertia: float = 0.0,
    prev_T: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate ocean heat transport and return temperature adjustment.
    
    Implements simplified ocean circulation:
    - Poleward surface currents carry warm water from tropics
    - Equatorward deep currents return cold water
    - Western boundary current intensification (Gulf Stream effect)
    - Ocean-atmosphere heat exchange
    
    Args:
        T: Current temperature field (Hc, Wc) in Kelvin
        elevation: Elevation/bathymetry field (Hc, Wc), normalized 0-1
        Hc: Height of coarse grid
        Wc: Width of coarse grid
        day_of_year: Current day (1-365) for seasonal effects
        dt_days: Time step in days
        transport_coefficient: Strength of ocean transport (0.0-1.0)
        
    Returns:
        Temperature adjustment array (Hc, Wc) in Kelvin
        
    Physical basis:
    - Ocean transports ~1-2 PW poleward (similar to atmosphere)
    - Western boundary currents (Gulf Stream, Kuroshio) carry warm water north
    - Eastern boundary currents (California, Canary) carry cold water south
    - Heat released in high latitudes warms atmosphere
    """
    # Identify ocean vs land from elevation.
    # NOTE: use a stable mask for loaded DEMs (ocean=0) to avoid latitudinal ringing/stripes.
    is_ocean = _ocean_mask_from_elevation(elevation)
    
    # Calculate latitude for each row
    lat_rows = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * 180.0  # degrees, -90 to +90
    abs_lat_rows = np.abs(lat_rows)
    lat_rows_2d = lat_rows[:, np.newaxis]  # (Hc, 1) for broadcasting
    
    # Initialize temperature adjustment
    T_adjustment = np.zeros_like(T)
    
    # ============================================================================
    # POLEWARD OCEAN HEAT TRANSPORT (Surface Currents)
    # ============================================================================
    # Warm tropical water flows poleward in surface currents, loses heat to
    # atmosphere in mid-high latitudes. This warms high-latitude air.
    
    # Calculate zonal (east-west) average temperature in ocean
    # Avoid np.nanmean (which can warn on all-NaN rows) by doing explicit sum/count.
    ocean_count = np.sum(is_ocean, axis=1).astype(np.float32)  # (Hc,)
    ocean_sum = np.sum(T * is_ocean.astype(np.float32), axis=1)  # (Hc,)
    global_mean = float(np.mean(T))
    with np.errstate(invalid='ignore', divide='ignore'):
        T_ocean_zonal = ocean_sum / np.maximum(ocean_count, 1.0)
    # Replace "no ocean in this latitude row" with a neutral fallback to prevent sharp jumps
    T_ocean_zonal = np.where(ocean_count > 0.0, T_ocean_zonal, global_mean).astype(np.float32)
    
    # Light latitudinal smoothing before taking derivatives (reduces banding/ringing).
    # This is intentionally minimal (3-pt filter, 2 passes) to keep the climatology similar.
    for _ in range(2):
        z = np.pad(T_ocean_zonal, 1, mode="edge")
        T_ocean_zonal = (0.25 * z[:-2] + 0.50 * z[1:-1] + 0.25 * z[2:]).astype(np.float32)
    
    # Calculate poleward heat flux
    # Peak at mid-latitudes (30-60°) where major currents (Gulf Stream) exist
    # Gaussian profile centered at 45° with width ~20°
    current_strength = np.exp(-((abs_lat_rows - 45.0) / 20.0)**2)  # (Hc,)

    # Temperature gradient drives heat transport
    # Use finite difference to approximate meridional gradient
    # CRITICAL FIX: Gradient should be (T_north - T_south) so that heat flows
    # from warm equator TO cold poles (positive gradient = poleward flux)
    # Row indexing: row 0 = 90°N, row Hc-1 = 90°S (increasing row = southward)
    T_ocean_gradient = np.zeros_like(T_ocean_zonal)
    T_ocean_gradient[1:-1] = (T_ocean_zonal[:-2] - T_ocean_zonal[2:]) / 2.0  # FIXED: (north - south)
    T_ocean_gradient[0] = T_ocean_zonal[0] - T_ocean_zonal[1]  # FIXED: North pole gradient
    T_ocean_gradient[-1] = T_ocean_zonal[-2] - T_ocean_zonal[-1]  # FIXED: South pole gradient
    
    # Poleward flux: move heat from warm (low lat) to cold (high lat)
    # Positive gradient (warm south) = poleward heat flux
    poleward_flux = transport_coefficient * current_strength * T_ocean_gradient * dt_days  # (Hc,)
    
    # Apply heat loss at source (tropics) and gain at sink (high latitudes)
    # This is a simplified divergence: ∂/∂y(flux)
    flux_divergence = np.zeros_like(poleward_flux)
    flux_divergence[1:-1] = (poleward_flux[2:] - poleward_flux[:-2]) / 2.0
    flux_divergence[0] = poleward_flux[1] - poleward_flux[0]
    flux_divergence[-1] = poleward_flux[-1] - poleward_flux[-2]
    
    # Clamp flux divergence to prevent extreme adjustments
    flux_divergence = np.clip(flux_divergence, -5.0, 5.0)  # Max ±5K per day
    
    # Apply to ocean cells only
    T_adjustment = -flux_divergence[:, np.newaxis] * is_ocean.astype(np.float32)
    
    # ============================================================================
    # WESTERN BOUNDARY CURRENT INTENSIFICATION
    # ============================================================================
    # Real ocean: currents are stronger on western sides (Gulf Stream, Kuroshio)
    # due to Coriolis effect and continental boundaries
    # This creates stronger heat transport on western coasts
    
    # Create east-west gradient: western side = higher transport
    x_positions = np.arange(Wc, dtype=np.float32) / Wc  # 0 (west) to 1 (east)
    western_enhancement = 1.0 + 0.5 * (1.0 - x_positions)  # 1.5 at west, 1.0 at east
    
    # Apply western boundary enhancement
    T_adjustment = T_adjustment * western_enhancement[np.newaxis, :]
    
    # ============================================================================
    # OCEAN-ATMOSPHERE HEAT EXCHANGE
    # ============================================================================
    # High-latitude oceans release heat to cold atmosphere
    # Low-latitude oceans absorb heat from warm atmosphere
    # This creates a net poleward heat transport mechanism
    
    # Calculate temperature difference between ocean and atmosphere
    # (In reality, T represents near-surface air temp, which couples to ocean)
    # High latitudes: ocean warmer than air → heat release
    # Low latitudes: ocean cooler than air → heat absorption
    
    # Equatorial reference temperature (warm baseline)
    # Apply inertia by blending toward the previous ocean temperature.
    if prev_T is not None and exchange_inertia > 0.0:
        a = float(np.clip(exchange_inertia, 0.0, 1.0))
        T_eff = (1.0 - a) * T + a * np.asarray(prev_T, dtype=np.float32)
    else:
        T_eff = T
    T_equator = np.mean(T_eff[abs_lat_rows < 10.0, :])
    
    # Heat exchange scales with latitude (stronger at high latitudes)
    # Reduce polar coupling to avoid excessive heat loss from ocean surface.
    exchange_strength = (abs_lat_rows / 90.0) ** 1.5  # 0 at equator, 1 at poles
    exchange_strength = float(exchange_strength_floor) + float(exchange_strength_span) * exchange_strength
    exchange_coefficient = float(exchange_coefficient)  # K per day (weak coupling)
    
    # Ocean releases/absorbs heat based on temperature excess
    # CRITICAL FIX: This term was adding excessive heat to poles (+3K/day), causing
    # Antarctic to warm by +40°C in winter. Disabled temporarily - proper ocean
    # heat transport is handled by the flux divergence term above.
    # TODO: Reformulate this to properly model ocean-atmosphere coupling without
    # creating unrealistic poleward heat transport.
    T_excess = T_eff - T_equator
    heat_exchange = np.zeros_like(T)  # DISABLED: was causing +3K/day warming at poles
    # Old (broken) formula: heat_exchange = -exchange_coefficient * exchange_strength * T_excess * dt_days
    heat_exchange = heat_exchange * is_ocean.astype(np.float32)
    
    # Add heat exchange to adjustment
    T_adjustment = T_adjustment + heat_exchange
    
    # Final clamp on total ocean adjustment
    T_adjustment = np.clip(T_adjustment, -10.0, 10.0)  # Max ±10K total per day
    
    return T_adjustment


def compute_ekman_transport(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    elevation: np.ndarray,
    ekman_coefficient: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute wind-driven Ekman transport (Phase 3: Ocean-Wind Coupling).

    Ekman transport is the net transport of water due to wind stress, deflected
    by Coriolis force. In the Northern Hemisphere, transport is 90° to the right
    of the wind; in the Southern Hemisphere, 90° to the left.

    Args:
        wind_u: (H,W) Eastward wind speed [m/s]
        wind_v: (H,W) Northward wind speed [m/s]
        elevation: (H,W) Elevation map (to determine ocean mask)
        ekman_coefficient: Wind-to-current scaling factor (default 0.03 = 3% of wind)

    Returns:
        (u_ekman, v_ekman): Ekman transport velocities [m/s]

    Physics:
    - Surface current ~ 3% of wind speed (observed ratio)
    - Deflected 45° by Coriolis (simplified from 90° for surface layer)
    - Right in NH, left in SH
    """
    H, W = wind_u.shape

    # Ocean mask (where currents exist)
    is_ocean = elevation <= 0.02

    # Latitude grid for Coriolis deflection
    lat = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi  # radians, -π/2 to +π/2
    lat_2d = np.repeat(lat[:, None], W, axis=1)

    # Coriolis deflection angle (45° to the right in NH, left in SH)
    # Simplified from full 90° Ekman spiral to represent surface layer average
    deflection = np.sign(lat_2d) * (np.pi / 4.0)  # ±45°

    # Wind speed and direction
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
    wind_angle = np.arctan2(wind_v, wind_u)  # Angle from east

    # Ekman current: scaled wind speed, deflected by Coriolis
    current_speed = ekman_coefficient * wind_speed
    current_angle = wind_angle + deflection

    # Convert back to u, v components
    u_ekman = current_speed * np.cos(current_angle)
    v_ekman = current_speed * np.sin(current_angle)

    # Mask to ocean only (no currents over land)
    u_ekman = np.where(is_ocean, u_ekman, 0.0)
    v_ekman = np.where(is_ocean, v_ekman, 0.0)

    return u_ekman.astype(np.float32), v_ekman.astype(np.float32)


def get_major_ocean_currents(
    Hc: int,
    Wc: int,
    day_of_year: int,
    wind_u: np.ndarray | None = None,
    wind_v: np.ndarray | None = None,
    elevation: np.ndarray | None = None,
    ekman_weight: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ocean current velocity field with wind coupling (Phase 3).

    Returns u (east-west) and v (north-south) velocity components for ocean currents.
    Combines climatological gyre patterns with wind-driven Ekman transport.

    Args:
        Hc: Height of grid
        Wc: Width of grid
        day_of_year: Day of year (for seasonal variations)
        wind_u: Optional (H,W) wind field for Ekman transport
        wind_v: Optional (H,W) wind field for Ekman transport
        elevation: Optional (H,W) elevation for ocean mask
        ekman_weight: Weight for Ekman transport (0.6 = 60% wind-driven, 40% climatology)

    Returns:
        (u, v): Velocity components in m/s, both shape (Hc, Wc)

    Major ocean currents modeled:
    - Subtropical gyres (clockwise NH, counterclockwise SH) [climatology]
    - Ekman transport (wind-driven, Coriolis-deflected) [dynamic]
    - Equatorial currents (eastward) [climatology]
    - Western boundary currents (Gulf Stream, Kuroshio) [climatology]
    - Antarctic Circumpolar Current [climatology]
    """
    # Calculate latitude and longitude grids
    lat_rows = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * 180.0
    lon_cols = ((np.arange(Wc, dtype=np.float32) + 0.5) / Wc) * 360.0 - 180.0
    
    lat_grid, lon_grid = np.meshgrid(lat_rows, lon_cols, indexing='ij')
    
    # Initialize velocity fields
    u = np.zeros((Hc, Wc), dtype=np.float32)  # East-west
    v = np.zeros((Hc, Wc), dtype=np.float32)  # North-south
    
    # Subtropical gyres (20-50° latitude)
    # Clockwise in NH, counterclockwise in SH
    gyre_lat = np.clip((np.abs(lat_grid) - 20.0) / 30.0, 0.0, 1.0)
    gyre_strength = np.sin(gyre_lat * np.pi)  # Peak at mid-latitude
    
    # Zonal (east-west) component: westward at low edge, eastward at high edge
    u_gyre = gyre_strength * np.sin((lat_grid - 35.0 * np.sign(lat_grid)) * np.pi / 30.0)
    
    # Meridional (north-south) component: poleward on west, equatorward on east
    v_gyre = gyre_strength * np.cos((lon_grid + 180.0) * np.pi / 180.0) * 0.5
    v_gyre = v_gyre * np.sign(lat_grid)  # Match hemisphere
    
    u += u_gyre * 0.2  # Scale to ~0.2 m/s
    v += v_gyre * 0.1  # Scale to ~0.1 m/s
    
    # Western boundary current intensification (Gulf Stream, Kuroshio)
    # Strong poleward flow on western boundaries
    western_boundary = np.exp(-((lon_grid + 90.0) / 30.0)**2)  # Peak at ~90°W
    boundary_current = western_boundary * gyre_strength * 1.0  # Strong (1 m/s)
    v += boundary_current * np.sign(lat_grid)
    
    # Equatorial currents (eastward near equator)
    equatorial_mask = np.exp(-(lat_grid / 10.0)**2)  # Gaussian at equator
    u += equatorial_mask * 0.3  # Eastward flow

    # Antarctic Circumpolar Current (eastward flow around 50-60°S)
    acc_mask = np.exp(-((lat_grid + 55.0) / 10.0)**2)  # Centered at 55°S
    u += acc_mask * 0.5  # Strong eastward flow

    # Phase 3: Add wind-driven Ekman transport if wind fields provided
    if wind_u is not None and wind_v is not None and elevation is not None:
        u_ekman, v_ekman = compute_ekman_transport(wind_u, wind_v, elevation)

        # Blend: ekman_weight% wind-driven, (1-ekman_weight)% climatological
        # Default 60% wind, 40% climatology keeps some stability
        u = (1.0 - ekman_weight) * u + ekman_weight * u_ekman
        v = (1.0 - ekman_weight) * v + ekman_weight * v_ekman

    return u, v


def generate_ocean_currents(
    elevation: np.ndarray,
    *,
    wind_u: np.ndarray | None = None,
    wind_v: np.ndarray | None = None,
    day_of_year: int = 0,
    time_days: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dynamic surface ocean current field for visualization.

    Combines a climatological gyre pattern with wind-driven surface drift and
    a light, deterministic mesoscale variability term. Land boundaries block
    and redirect flow for realistic coastal current patterns.
    """
    H, W = elevation.shape
    is_ocean = _ocean_mask_from_elevation(elevation)
    is_land = ~is_ocean

    # Phase 3: Pass wind fields to enable Ekman transport coupling
    base_u, base_v = get_major_ocean_currents(
        H, W, day_of_year=int(day_of_year),
        wind_u=wind_u, wind_v=wind_v, elevation=elevation
    )

    # Seasonal modulation (small amplitude).
    seasonal = 0.9 + 0.1 * np.sin(2.0 * np.pi * (float(day_of_year) / 365.2422))
    u = base_u * seasonal
    v = base_v * seasonal

    # Wind-driven surface drift (Ekman-like, modest rotation).
    if wind_u is not None and wind_v is not None:
        lat_rows = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0
        lat_grid = lat_rows[:, None].astype(np.float32)
        sign = np.sign(lat_grid)
        angle = np.deg2rad(20.0) * sign
        ca = np.cos(angle)
        sa = np.sin(angle)
        u_rot = wind_u * ca - wind_v * sa
        v_rot = wind_u * sa + wind_v * ca
        wind_scale = 0.03
        u = u + wind_scale * u_rot
        v = v + wind_scale * v_rot

    # Deterministic mesoscale variability for visual dynamics.
    t = float(time_days if time_days is not None else day_of_year)
    lon = ((np.arange(W, dtype=np.float32) + 0.5) / W) * 360.0 - 180.0
    lat = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    phase = 0.6 * np.sin(np.deg2rad(lon_grid * 3.0) + t * 0.08) * np.cos(np.deg2rad(lat_grid * 2.0) - t * 0.06)
    u = u + 0.04 * phase
    v = v + 0.02 * phase

    # Apply land boundary conditions: block flow into land, enhance coastal currents.
    # Meridional (north-south) flow blocked at land boundaries.
    v_blocked = v.copy()
    v_blocked[is_land] = 0.0
    # Reduce meridional flow near land (coastal friction).
    land_north = np.roll(is_land, -1, axis=0)
    land_south = np.roll(is_land, 1, axis=0)
    coastal_mask = (land_north | land_south) & is_ocean
    v_blocked = np.where(coastal_mask, v_blocked * 0.7, v_blocked)

    # Zonal (east-west) flow blocked at land boundaries.
    u_blocked = u.copy()
    u_blocked[is_land] = 0.0
    # Reduce zonal flow near land.
    land_east = np.roll(is_land, -1, axis=1)
    land_west = np.roll(is_land, 1, axis=1)
    coastal_mask_z = (land_east | land_west) & is_ocean
    u_blocked = np.where(coastal_mask_z, u_blocked * 0.7, u_blocked)

    # Western boundary current enhancement at actual western coastlines.
    # Detect western ocean boundaries (ocean cells with land to the west).
    western_coast = is_ocean & land_west
    if np.any(western_coast):
        lat_rows = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0
        lat_grid = lat_rows[:, None].astype(np.float32)
        abs_lat = np.abs(lat_grid)
        midlat_mask = (abs_lat >= 20.0) & (abs_lat <= 60.0)
        enhancement = western_coast & midlat_mask
        v_blocked = np.where(enhancement, v_blocked * 1.4, v_blocked)

    # Final mask: zero on land.
    u_final = u_blocked * is_ocean.astype(np.float32)
    v_final = v_blocked * is_ocean.astype(np.float32)
    return u_final.astype(np.float32), v_final.astype(np.float32)


def apply_ocean_transport(
    T: np.ndarray,
    elevation: np.ndarray,
    Hc: int,
    Wc: int,
    day_of_year: int,
    dt_days: float,
    transport_coefficient: float = 0.3,
) -> np.ndarray:
    """Apply ocean heat transport to temperature field.
    
    Main interface function for ocean module. Calculates and applies
    ocean heat transport effects to the temperature field.
    
    Args:
        T: Temperature field (Hc, Wc) in Kelvin
        elevation: Elevation field (Hc, Wc), normalized 0-1
        Hc: Grid height
        Wc: Grid width
        day_of_year: Current day (1-365)
        dt_days: Time step in days
        transport_coefficient: Ocean transport strength (0.0-1.0)
        
    Returns:
        Updated temperature field (Hc, Wc) in Kelvin
    """
    # Calculate ocean heat transport adjustment
    T_adjustment = calculate_ocean_heat_transport(
        T, elevation, Hc, Wc, day_of_year, dt_days, transport_coefficient
    )
    
    # Apply adjustment
    T_new = T + T_adjustment
    
    return T_new

