"""Ocean circulation and thermal transport modeling.

This module handles oceanic heat transport through:
1. Wind-driven surface currents (e.g., Gulf Stream, Kuroshio)
2. Thermohaline circulation (density-driven deep ocean currents)
3. Ocean-atmosphere heat exchange

Ocean currents on Earth transport ~1-2 PW of heat poleward, comparable to
atmospheric transport. This is critical for moderating high-latitude climates.
"""

import numpy as np


def calculate_ocean_heat_transport(
    T: np.ndarray,
    elevation: np.ndarray,
    Hc: int,
    Wc: int,
    day_of_year: int,
    dt_days: float,
    transport_coefficient: float = 0.3,
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
    # Identify ocean vs land from elevation
    # Using median as sea level (similar to simulate.py)
    elev_median = float(np.median(elevation))
    is_ocean = elevation <= elev_median
    
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
    T_ocean = np.where(is_ocean, T, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):  # Suppress NaN/divide warnings
        T_ocean_zonal = np.nanmean(T_ocean, axis=1)  # (Hc,) - zonal average
    
    # Replace NaN with global mean (for rows with no ocean)
    # Use full temperature field mean as fallback
    global_mean = np.mean(T)
    T_ocean_zonal = np.where(np.isnan(T_ocean_zonal), global_mean, T_ocean_zonal)
    
    # Calculate poleward heat flux
    # Peak at mid-latitudes (30-60°) where major currents (Gulf Stream) exist
    # Gaussian profile centered at 45° with width ~20°
    current_strength = np.exp(-((abs_lat_rows - 45.0) / 20.0)**2)  # (Hc,)
    
    # Temperature gradient drives heat transport
    # Use finite difference to approximate meridional gradient
    T_ocean_gradient = np.zeros_like(T_ocean_zonal)
    T_ocean_gradient[1:-1] = (T_ocean_zonal[2:] - T_ocean_zonal[:-2]) / 2.0
    T_ocean_gradient[0] = T_ocean_zonal[1] - T_ocean_zonal[0]
    T_ocean_gradient[-1] = T_ocean_zonal[-1] - T_ocean_zonal[-2]
    
    # Poleward flux: move heat from warm (low lat) to cold (high lat)
    # Positive gradient (warm south) = poleward heat flux
    poleward_flux = transport_coefficient * current_strength * T_ocean_gradient * dt_days  # (Hc,)
    
    # Apply heat loss at source (tropics) and gain at sink (high latitudes)
    # This is a simplified divergence: ∂/∂y(flux)
    flux_divergence = np.zeros_like(poleward_flux)
    flux_divergence[1:-1] = (poleward_flux[2:] - poleward_flux[:-2]) / 2.0
    flux_divergence[0] = poleward_flux[1] - poleward_flux[0]
    flux_divergence[-1] = poleward_flux[-1] - poleward_flux[-2]
    
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
    T_equator = np.mean(T[abs_lat_rows < 10.0, :])
    
    # Heat exchange scales with latitude (stronger at high latitudes)
    exchange_strength = (abs_lat_rows / 90.0) ** 1.5  # 0 at equator, 1 at poles
    exchange_coefficient = 0.05  # K per day (weak coupling)
    
    # Ocean releases/absorbs heat based on temperature excess
    T_excess = T - T_equator
    heat_exchange = -exchange_coefficient * exchange_strength[:, np.newaxis] * T_excess * dt_days
    heat_exchange = heat_exchange * is_ocean.astype(np.float32)
    
    # Add heat exchange to adjustment
    T_adjustment = T_adjustment + heat_exchange
    
    return T_adjustment


def get_major_ocean_currents(
    Hc: int,
    Wc: int,
    day_of_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate simplified ocean current velocity field.
    
    Returns u (east-west) and v (north-south) velocity components for ocean currents.
    This is a highly simplified model of major ocean gyres.
    
    Args:
        Hc: Height of grid
        Wc: Width of grid
        day_of_year: Day of year (for seasonal variations)
        
    Returns:
        (u, v): Velocity components in m/s, both shape (Hc, Wc)
        
    Major ocean currents modeled:
    - Subtropical gyres (clockwise NH, counterclockwise SH)
    - Equatorial currents (eastward)
    - Western boundary currents (Gulf Stream, Kuroshio)
    - Antarctic Circumpolar Current
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
    
    return u, v


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

