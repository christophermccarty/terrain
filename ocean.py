"""Ocean circulation and thermal transport modeling.

This module handles oceanic heat transport through:
1. Wind-driven surface currents (e.g., Gulf Stream, Kuroshio)
2. Thermohaline circulation (density-driven deep ocean currents)
3. Ocean-atmosphere heat exchange

Ocean currents on Earth transport ~1-2 PW of heat poleward, comparable to
atmospheric transport. This is critical for moderating high-latitude climates.
"""

import numpy as np
from masks import get_masks


def _lat_deg(n: int) -> np.ndarray:
    """Row-centred latitudes [°] for a grid with n rows (north→south, float32)."""
    return ((0.5 - (np.arange(n, dtype=np.float32) + 0.5) / n) * 180.0)


def _lat_rad(n: int) -> np.ndarray:
    """Row-centred latitudes [rad] for a grid with n rows (north→south, float32)."""
    return ((0.5 - (np.arange(n, dtype=np.float32) + 0.5) / n) * np.pi)


_K_G_ICE = 3e-5   # Stefan growth constant [m² / (day × cold_excess_unit)]
_H_ICE_REF = 0.2  # denominator offset to prevent div/0 and limit initial growth [m]
_K_M_ICE = 0.003  # melt rate [m / (day × K above melt_temp)]
_H_ICE_MAX = 5.0  # maximum realistic sea ice thickness [m]


def update_sea_ice(
    T: np.ndarray,
    elevation: np.ndarray,
    prev_ice: np.ndarray | None,
    dt_days: float,
    prev_thickness: np.ndarray | None = None,
    *,
    freeze_temp: float = 269.5,
    melt_temp: float = 273.35,
    freeze_rate: float = 0.06,
    melt_rate: float = 0.16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update sea ice fraction and thickness for one time step.

    Returns:
        (ice_new, delta_ice, thickness_new):
            ice_new      — updated ice fraction [0–1]
            delta_ice    — change in ice fraction (>0 freezing, <0 melting)
            thickness_new — updated ice thickness [m]; 0 where no ice
    """
    is_ocean, _ = get_masks(elevation)
    ice = np.zeros_like(T, dtype=np.float32) if prev_ice is None else np.asarray(prev_ice, dtype=np.float32)
    ice_prev = ice.copy()
    T = np.asarray(T, dtype=np.float32)
    H = T.shape[0]
    lat_deg = _lat_deg(H)
    abs_lat = np.abs(lat_deg)[:, None]
    polar_factor = np.clip((abs_lat - 55.0) / 25.0, 0.0, 1.0).astype(np.float32, copy=False)
    margin_factor = (1.0 - polar_factor).astype(np.float32, copy=False)

    cold_excess = np.clip((freeze_temp - T) / 2.5, 0.0, 1.5).astype(np.float32, copy=False)
    warm_excess = np.clip((T - melt_temp) / 2.5, 0.0, 1.5).astype(np.float32, copy=False)
    open_water = (1.0 - np.clip(ice, 0.0, 1.0)).astype(np.float32, copy=False)

    # Growth is fastest over newly exposed cold water and increases toward the poles.
    growth = freeze_rate * dt_days * cold_excess * open_water * (0.22 + 0.78 * polar_factor)
    # Melt accelerates once temperatures rise well above the persistence threshold.
    melt = melt_rate * dt_days * warm_excess * (0.75 + 0.25 * ice + 0.35 * margin_factor)

    ice = np.where(is_ocean, np.clip(ice + growth - melt, 0.0, 1.0), 0.0)
    ice_new = np.where(is_ocean, ice, 0.0)
    delta_ice = ice_new - np.where(is_ocean, ice_prev, 0.0)

    # --- Thickness evolution (Stefan's law, parameterised) ---
    # Growth: dh/dt ∝ cold_excess / (h + H_REF) — slows as ice thickens.
    # Melt:   dh/dt ∝ warm_K — proportional to temperature above melt threshold.
    h = (np.zeros_like(T, dtype=np.float32)
         if prev_thickness is None else np.asarray(prev_thickness, dtype=np.float32))
    growth_h = (_K_G_ICE * cold_excess * ice_new * float(dt_days)
                / (h + _H_ICE_REF)).astype(np.float32, copy=False)
    warm_K = np.clip(T - melt_temp, 0.0, None).astype(np.float32, copy=False)
    melt_h = (_K_M_ICE * warm_K * float(dt_days)).astype(np.float32, copy=False)
    h_new = np.clip(h + growth_h - melt_h, 0.0, _H_ICE_MAX)
    # Thickness is zero wherever there is no ice
    thickness_new = np.where(ice_new > 0.0, h_new, 0.0).astype(np.float32, copy=False)

    return ice_new, delta_ice, thickness_new


def calculate_ocean_heat_transport(
    T: np.ndarray,
    elevation: np.ndarray,
    Hc: int,
    Wc: int,
    day_of_year: int,
    dt_days: float,
    transport_coefficient: float = 0.3,
    exchange_coefficient: float = 0.03,
    exchange_inertia: float = 0.0,
    prev_T: np.ndarray | None = None,
    ice_cover: np.ndarray | None = None,
    T_equilibrium: np.ndarray | None = None,
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
    is_ocean, _ = get_masks(elevation)
    
    # Calculate latitude for each row
    lat_rows = _lat_deg(Hc)
    abs_lat_rows = np.abs(lat_rows)
    
    # Initialize temperature adjustment
    T_adjustment = np.zeros_like(T)
    
    # ============================================================================
    # POLEWARD OCEAN HEAT TRANSPORT (Surface Currents)
    # ============================================================================
    # Warm tropical water flows poleward in surface currents, loses heat to
    # atmosphere in mid-high latitudes. This warms high-latitude air.
    
    # Calculate zonal (east-west) average temperature in ocean
    # Avoid np.nanmean (which can warn on all-NaN rows) by doing explicit sum/count.
    ocean_count = np.sum(is_ocean, axis=1).astype(np.float32, copy=False)  # (Hc,)
    ocean_sum = np.sum(T * is_ocean.astype(np.float32), axis=1)  # (Hc,)
    # Area-weighted global mean as neutral fallback for rows that have no ocean.
    # On an equirectangular grid each row represents a different area ∝ cos(lat),
    # so we must weight by cos(lat) to avoid over-representing polar rows.
    cos_lat_1d = np.cos(np.deg2rad(lat_rows))                   # (Hc,)
    lat_weight = cos_lat_1d[:, np.newaxis]                       # (Hc, 1) broadcast
    total_area = float(np.sum(lat_weight))
    global_mean = float(np.sum(T * lat_weight) / max(total_area, 1.0))
    with np.errstate(invalid='ignore', divide='ignore'):
        T_ocean_zonal = ocean_sum / np.maximum(ocean_count, 1.0)
    # Replace "no ocean in this latitude row" with a neutral fallback to prevent sharp jumps
    T_ocean_zonal = np.where(ocean_count > 0.0, T_ocean_zonal, global_mean).astype(np.float32, copy=False)
    
    # Light latitudinal smoothing before taking derivatives (reduces banding/ringing).
    # Single 3-point pass is sufficient; a second pass adds little beyond the first.
    for _ in range(1):
        z = np.pad(T_ocean_zonal, 1, mode="edge")
        T_ocean_zonal = (0.25 * z[:-2] + 0.50 * z[1:-1] + 0.25 * z[2:]).astype(np.float32, copy=False)
    
    # Calculate poleward heat flux.
    # NH: Gulf Stream / AMOC deposit heat at ~45-65°N → Gaussian centred at 45°N.
    # SH: Southern Ocean (ACC) is primarily *zonal*; it does NOT create strong
    #     meridional heat convergence at 45°S.  Using the same |lat| Gaussian
    #     symmetrically over-warms SH mid-latitudes by ~10 K.  Use a flat
    #     reduced profile (35 % of peak) for SH so heat is deposited at the
    #     high-latitude end via polar_damp rather than at 45°S.
    nh_strength = np.exp(-((lat_rows - 45.0) / 20.0) ** 2)          # positive at NH latitudes
    nh_subpolar_bonus = 0.22 * np.clip((lat_rows - 52.0) / 18.0, 0.0, 1.0)
    sh_strength = 0.35 * np.ones_like(lat_rows)                      # flat ~ACC-like
    current_strength = np.where(lat_rows >= 0, nh_strength + nh_subpolar_bonus, sh_strength)  # (Hc,)

    # Polar damping: real ocean currents weaken at the highest latitudes,
    # but the Antarctic Circumpolar Current remains strong through 65-70°.
    # Extended full-transport zone to 65° (was 60°); minimum raised to 20% (was 10%).
    # Previously: clip((75-lat)/15, 0.1, 1.0) → 3.5% effective transport at 70°S under ice.
    # Now: clip((80-lat)/15, 0.2, 1.0) → at 70°S: 0.67 × 0.7 (ice 30%) = 47% — enough to warm.
    polar_damp = np.clip((80.0 - abs_lat_rows) / 15.0, 0.2, 1.0)  # 1.0 at <65°, 0.2 at >80°
    current_strength = current_strength * polar_damp

    # Ice partially blocks surface currents, but the deep thermohaline circulation and
    # Antarctic Circumpolar Current continue under ice. Reduced from 50% to 30% blocking.
    # At 50%: combined with polar_damp this left ~3.5% transport at 70°S — too weak for
    # seasonal melt. At 30%: ~47% transport reaches 70°S, allowing summer ice loss.
    if ice_cover is not None:
        ice_arr = np.clip(np.asarray(ice_cover, dtype=np.float32), 0.0, 1.0)
        ice_ocean = ice_arr * is_ocean.astype(np.float32)
        ice_zonal = np.sum(ice_ocean, axis=1) / np.maximum(ocean_count, 1.0)
        ice_block = np.where(lat_rows >= 0, 0.18, 0.26).astype(np.float32, copy=False)
        current_strength = current_strength * (1.0 - ice_block * ice_zonal)

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
    
    # Western boundary currents form on the western side of each OCEAN BASIN
    # (ocean cells with land immediately to their west), not the western edge
    # of the map. The previous formulation (`1.5 at map column 0 tapering to
    # 1.0 at column W-1`) boosted transport along the dateline mid-Pacific and
    # gave the actual Gulf Stream/Kuroshio positions no enhancement at all,
    # while injecting a spurious net-heat zonal gradient across every basin.
    # Same topology rule as get_major_ocean_currents' WBC mask; the boost
    # decays over the two cells downstream (east) of the boundary.
    land_west = np.roll(~is_ocean, 1, axis=1)
    midlat_band = (np.abs(lat_rows) >= 15.0) & (np.abs(lat_rows) <= 65.0)
    wbc_core = (is_ocean & land_west & midlat_band[:, np.newaxis]).astype(np.float32)
    western_enhancement = (
        1.0
        + 0.5 * wbc_core
        + 0.35 * np.roll(wbc_core, 1, axis=1)   # 1 cell east of the boundary
        + 0.2 * np.roll(wbc_core, 2, axis=1)    # 2 cells east
    )
    T_adjustment = T_adjustment * np.clip(western_enhancement, 1.0, 1.5)
    
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
    
    # Ocean-atmosphere exchange: restore T toward radiative equilibrium.
    # The old approach used T_equator as the restoring target, which added an
    # unconditional poleward flux and caused +3 K/day warming at poles.
    # Now we accept T_equilibrium from simulate.py (the per-cell radiative
    # equilibrium).  Rate 0.03 K/day → ~30-day ocean skin-layer timescale.
    # If T_equilibrium is not provided the exchange remains zero (safe default).
    if T_equilibrium is not None:
        if prev_T is not None and exchange_inertia > 0.0:
            a = float(np.clip(exchange_inertia, 0.0, 1.0))
            T_eff = (1.0 - a) * T + a * np.asarray(prev_T, dtype=np.float32)
        else:
            T_eff = T
        T_ref = np.asarray(T_equilibrium, dtype=np.float32)
        # K/day restoring rate toward radiative equilibrium (~30-day skin layer
        # at the 0.03 default). Now actually wired to the tunable parameter —
        # it was previously hardcoded to 0.03 while callers/optimizer configs
        # believed they were tuning it.
        exchange_rate = float(exchange_coefficient)
        heat_exchange = -exchange_rate * (T_eff - T_ref) * dt_days
        # Latitude factor: slightly stronger coupling at high latitudes where
        # ocean-atmosphere temperature contrast drives the largest heat flux.
        lat_factor = 0.5 + 0.5 * (abs_lat_rows / 90.0)
        heat_exchange = heat_exchange * lat_factor[:, np.newaxis]
        heat_exchange = np.clip(heat_exchange, -2.0, 2.0)   # ≤ 2 K/step
        heat_exchange = heat_exchange * is_ocean.astype(np.float32)
        T_adjustment = T_adjustment + heat_exchange
    
    # Final clamp on total ocean adjustment
    T_adjustment = np.clip(T_adjustment, -10.0, 10.0)  # Max ±10K total per day
    
    return T_adjustment


def compute_ekman_transport(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    elevation: np.ndarray,
    ekman_coefficient: float = 0.03,
    rotation_direction: float = 1.0,
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
    is_ocean, _ = get_masks(elevation)

    # Latitude grid for Coriolis deflection
    lat = _lat_rad(H)
    lat_2d = np.repeat(lat[:, None], W, axis=1)

    # Coriolis deflection angle (45° to the right in NH, left in SH for a
    # prograde rotator; mirrored for retrograde planets, whose Coriolis
    # deflection flips sign — matches planet_params.coriolis_parameter).
    # Simplified from full 90° Ekman spiral to represent surface layer average
    deflection = np.sign(lat_2d) * np.sign(float(rotation_direction) or 1.0) * (np.pi / 4.0)

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


def compute_gyre_currents(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    elevation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a 2D barotropic gyre circulation from wind-stress curl (Jul 2026).

    Unlike `compute_ekman_transport` (a local, per-cell deflection of the wind
    itself), this solves for a real streamfunction from the *curl* of wind
    stress over the whole ocean basin, giving currents genuine east-west
    (gyre) structure -- western boundary currents, subpolar gyres -- that the
    existing 1D zonal-mean transport (`calculate_ocean_heat_transport`) and
    Ekman deflection can't produce on their own. Reuses
    `atmosphere._streamfunction_from_vorticity`, the same generic `∇²ψ=ω`
    spectral Poisson solver already used to divergence-clean the diagnosed
    wind field in `generate_wind_field` -- no new solver needed, just a new
    vorticity source (wind-stress curl instead of wind vorticity) and the
    standard barotropic relation `u=-∂ψ/∂y, v=∂ψ/∂x` instead of atmosphere.py's
    own sign/scale convention, which this function matches exactly for
    consistency (see its usage in `atmosphere.generate_wind_field`).

    Args:
        wind_u: (H,W) Eastward wind speed [m/s]
        wind_v: (H,W) Northward wind speed [m/s]
        elevation: (H,W) Elevation map (to determine ocean mask)

    Returns:
        (u_gyre, v_gyre): barotropic gyre current velocities [m/s], ocean-only

    Physics:
    - Wind stress ∝ wind_speed² (quadratic drag), matching Wanninkhof-style
      wind-stress conventions used elsewhere in this codebase
      (carbon_cycle.ocean_co2_flux).
    - Vorticity ω = ∂τ_y/∂x - ∂τ_x/∂y (curl of wind stress).
    - Streamfunction ψ solves ∇²ψ = ω on the periodic-in-x, DFT-in-y coarse
      grid (an approximation that tolerates no true meridional boundary
      condition, same caveat as its atmosphere.py usage).
    - No natural physical amplitude from this idealized solve -- clipped to
      ±0.5 m/s the same way `compute_ekman_transport`'s informal bound works,
      rather than a derived value.
    """
    is_ocean, _ = get_masks(elevation)
    ocean_f = is_ocean.astype(np.float32)

    from atmosphere import _streamfunction_from_vorticity, _ddx_periodic

    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
    tau_x = (wind_u * wind_speed * ocean_f).astype(np.float32)
    tau_y = (wind_v * wind_speed * ocean_f).astype(np.float32)

    dtau_y_dx = _ddx_periodic(tau_y)
    dtau_x_dy = np.gradient(tau_x, axis=0)
    omega = (dtau_y_dx - dtau_x_dy).astype(np.float32)

    psi = _streamfunction_from_vorticity(omega)
    H, W = wind_u.shape
    u_gyre = -np.gradient(psi, axis=0) * (H / np.pi)
    v_gyre = -_ddx_periodic(psi) * (W / (2.0 * np.pi))

    u_gyre = (np.clip(u_gyre, -0.5, 0.5) * ocean_f).astype(np.float32)
    v_gyre = (np.clip(v_gyre, -0.5, 0.5) * ocean_f).astype(np.float32)
    return u_gyre, v_gyre


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
    lat_rows = _lat_deg(Hc)
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
    
    # Western boundary current intensification (Gulf Stream, Kuroshio).
    # When elevation is available, derive WBC locations from topology: any
    # mid-latitude ocean cell with land immediately to its west is a WBC site.
    # This works for any planet or continent configuration.
    # Falls back to a hardcoded Gaussian at 90°W (Earth Gulf-Stream only) when
    # no elevation map is provided.
    if elevation is not None:
        is_ocean_e, _ = get_masks(elevation)
        land_west_e = np.roll(~is_ocean_e, 1, axis=1)   # land one column to the west
        midlat = (np.abs(lat_grid) >= 20.0) & (np.abs(lat_grid) <= 65.0)
        wbc_mask = is_ocean_e & land_west_e & midlat
        v += wbc_mask.astype(np.float32) * np.sign(lat_grid) * 1.0
    else:
        western_boundary = np.exp(-((lon_grid + 90.0) / 30.0)**2)  # Earth-specific fallback
        v += western_boundary * gyre_strength * np.sign(lat_grid)
    
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
    is_ocean, _ = get_masks(elevation)
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
        lat_rows = _lat_deg(H)
        lat_grid = lat_rows[:, None].astype(np.float32, copy=False)
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
    lat = _lat_deg(H)
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
        lat_rows = _lat_deg(H)
        lat_grid = lat_rows[:, None].astype(np.float32, copy=False)
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


def evolve_salinity(
    salinity: np.ndarray,
    T_sst: np.ndarray,
    elevation: np.ndarray,
    precipitation: np.ndarray | None,
    ice_delta: np.ndarray,
    dt_days: float = 1.0,
    pp=None,
) -> np.ndarray:
    """Evolve ocean surface salinity [PSU] by one time step.

    Tendencies (PSU/day):
      E–P balance  : evaporation concentrates, rain dilutes
      Brine rejection: freezing ice expels salt; melting injects freshwater
      Restoring     : slow drift back toward reference (deep mixing proxy), τ=2yr
    """
    from planet_params import EARTH
    if pp is None:
        pp = EARTH

    sea_mask, _ = get_masks(elevation)
    ocean_f = sea_mask.astype(np.float32)

    sal = salinity.copy()

    # Evaporation: raises salinity where T > 270 K (open water, not frozen)
    T_above_freeze = np.clip(T_sst - 270.0, 0.0, None).astype(np.float32, copy=False)
    evap_rate = 0.002 * T_above_freeze  # PSU/day per degree above freeze

    # Precipitation: dilutes salinity (P in mm/day → scale 0.001 to PSU/day units)
    if precipitation is not None:
        precip_dilution = precipitation.astype(np.float32, copy=False) * 0.001
    else:
        precip_dilution = np.zeros_like(sal)

    ep_tendency = (evap_rate - precip_dilution) * float(dt_days)

    # Brine rejection: freezing (ice_delta > 0) → salt expelled to ocean
    # Melting (ice_delta < 0) → freshwater → salinity decreases
    # Scale: 0.5 PSU per unit of ice_delta per day (crude but plausible)
    brine_tendency = np.clip(ice_delta, -0.5, 0.5) * 0.5 * float(dt_days)

    # Restoring toward reference with τ = 2yr = 730 days
    tau_restore = 730.0
    restore_tendency = -(sal - float(pp.salinity_reference_psu)) / tau_restore * float(dt_days)

    sal_new = sal + (ep_tendency + brine_tendency + restore_tendency) * ocean_f
    sal_new = np.clip(sal_new, 0.0, 45.0).astype(np.float32, copy=False)
    # Land cells remain 0
    sal_new = np.where(ocean_f > 0.5, sal_new, 0.0).astype(np.float32, copy=False)

    return sal_new
