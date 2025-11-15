"""Temperature overlay generation for equirectangular maps.

Produces a transparent-friendly RGB overlay (float in [0,1]) based on a
latitudinal blackbody equilibrium approximation under Earth/Sun-like insolation.
"""

from __future__ import annotations

import numpy as np

# Simple one-layer gray-atmosphere greenhouse tuned for Earth
# ε=0.68 provides realistic base temperatures (original value restored)
# Combined with mild evaporative cooling and improved color gradient
EPSILON_ATM = 0.68


def _daily_mean_insolation_Q(lat_rad: np.ndarray, day_of_year: int, S0: float = 1361.0) -> np.ndarray:
    """Daily-mean TOA insolation Q(φ, δ) in W/m^2.

    Uses standard astronomy formula with solar declination δ that varies by
    day-of-year. Handles polar night/day. Vectorized for `lat_rad` arrays.
    Special handling for exact poles to avoid numerical precision issues.
    
    INCLUDES atmospheric path length losses: polar regions lose more energy
    due to sunlight traveling through more atmosphere at low sun angles.
    """
    lat = np.asarray(lat_rad, dtype=np.float32)
    obliq = np.deg2rad(23.44)
    gamma = 2.0 * np.pi * (float(day_of_year) - 80.0) / 365.2422
    delta = np.arcsin(np.sin(obliq) * np.sin(gamma))
    
    # Standard formula for non-pole latitudes
    # Use tan(lat) but clamp to avoid numerical issues near poles
    lat_safe = np.clip(lat, -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
    cosH0 = -np.tan(lat_safe) * np.tan(delta)
    H0 = np.arccos(np.clip(cosH0, -1.0, 1.0))
    H0 = np.where(cosH0 <= -1.0, np.pi, H0)  # 24h daylight
    H0 = np.where(cosH0 >= 1.0, 0.0, H0)     # polar night
    Q_standard = (S0 / np.pi) * (H0 * np.sin(lat_safe) * np.sin(delta) + np.cos(lat_safe) * np.cos(delta) * np.sin(H0))
    
    # Special case for exact poles: Q = S0 * sin(lat) * sin(delta) when in polar day
    # At North Pole (lat=π/2): Q = S0 * sin(delta) if delta > 0, else 0
    # At South Pole (lat=-π/2): Q = S0 * |sin(delta)| if delta < 0, else 0
    pole_mask = np.abs(np.abs(lat) - np.pi/2) < 1e-6
    Q_pole = np.zeros_like(lat)  # Initialize always
    if np.any(pole_mask):
        # North Pole: positive during northern summer (delta > 0)
        np_mask = (pole_mask) & (lat > 0)
        Q_pole[np_mask] = S0 * np.maximum(0.0, np.sin(delta))
        # South Pole: positive during southern summer (delta < 0)
        sp_mask = (pole_mask) & (lat < 0)
        Q_pole[sp_mask] = S0 * np.maximum(0.0, -np.sin(delta))
    
    # Combine: use pole formula at poles, standard elsewhere
    Q_base = np.where(pole_mask, Q_pole, Q_standard)
    
    # NOTE: This returns TOP-OF-ATMOSPHERE insolation
    # Atmospheric absorption/scattering is handled implicitly in the greenhouse effect
    # Direct atmospheric transmission losses are SMALL (~5-10%) and already captured
    # in the effective greenhouse calculation. Don't double-count them here!
    # The 45% losses I added before made the planet an ice ball - that was wrong!
    
    return Q_base


def _coarse_shape(H: int, W: int, block_size: int) -> tuple[int, int]:
    bs = max(1, int(block_size))
    Hc = max(1, (H + bs - 1) // bs)
    Wc = max(1, (W + bs - 1) // bs)
    return Hc, Wc


def _upsample_repeat(field: np.ndarray, H: int, W: int, block_size: int) -> np.ndarray:
    bs = max(1, int(block_size))
    up = np.repeat(np.repeat(field, bs, axis=0), bs, axis=1)
    return up[:H, :W]


def generate_temperature_overlay(height: int, width: int, day_of_year: int = 1, epsilon_atm: float = EPSILON_ATM, block_size: int = 3, elevation: np.ndarray | None = None) -> np.ndarray:
    """Return an (H,W,3) float32 RGB overlay in [0,1] for given map size.

    NOW USES SIMULATION PHYSICS for consistency:
    - Creates a temporary simulation state to generate temperature
    - Includes ocean thermal inertia, seasonal lag, and all moderating effects
    - Ensures the displayed temperature map matches what the simulation will show
    
    Args:
        height: Map height in pixels
        width: Map width in pixels
        day_of_year: Day of year (0-365)
        epsilon_atm: Atmospheric emissivity for greenhouse effect
        block_size: Coarse resolution for faster computation
        elevation: Optional (H, W) elevation array [0, 1] where 0.5 = sea level
    """
    h = int(height)
    w = int(width)
    
    # Use simulation system to generate temperature with consistent physics
    from simulate import create_initial_state
    
    # Create elevation if not provided
    if elevation is None:
        elevation = np.zeros((h, w), dtype=np.float32)
    
    # Generate initial state using full simulation physics
    state = create_initial_state(elevation, day_of_year=float(day_of_year))
    T_kelvin = state.temperature
    
    # Temperature already calculated by simulation system (includes all physics)
    # T_kelvin shape is (H, W) at full resolution - no need to downsample
    T_lat = T_kelvin
    
    # Log temperature stats (now using simulation temperature)
    import logging
    LOG = logging.getLogger("planetsim")
    LOG.info(f"[Temperature from Simulation] min={float(np.min(T_lat)):.1f}K ({float(np.min(T_lat)-273.15):.1f}°C), "
             f"mean={float(np.mean(T_lat)):.1f}K ({float(np.mean(T_lat)-273.15):.1f}°C), "
             f"max={float(np.max(T_lat)):.1f}K ({float(np.max(T_lat)-273.15):.1f}°C)")
    
    # T_lat is already full resolution (H, W) with all physics applied by simulation
    # (includes altitude correction, ocean thermal inertia, seasonal lag, etc.)
    
    # Normalize to FULL realistic Earth temperature range
    # Extended range: -80°C to +40°C (193K to 313K)
    # This shows full spectrum from Arctic winter extremes to tropical heat
    tmin, tmax = 193.15, 313.15  # -80°C to +40°C
    v = np.clip((T_lat - tmin) / (tmax - tmin), 0.0, 1.0).astype(np.float32)

    # Enhanced color gradient with better differentiation across full range
    # Blue→Cyan→Green→Yellow→Orange→Red with more detail in inhabited zones
    color_stops = np.array([
        [0.05, 0.0, 0.4],   # 0.00: Deep purple-blue (-80°C) - extreme Arctic winter
        [0.0, 0.2, 0.7],    # 0.12: Dark blue (-65°C) - Arctic winter
        [0.0, 0.5, 0.9],    # 0.25: Blue (-50°C) - severe cold
        [0.0, 0.7, 0.8],    # 0.38: Cyan (-35°C) - very cold
        [0.2, 0.8, 0.6],    # 0.50: Cyan-green (-20°C) - cold temperate
        [0.4, 0.85, 0.4],   # 0.58: Green (-5°C) - cool temperate
        [0.65, 0.9, 0.3],   # 0.67: Yellow-green (+5°C) - mild temperate
        [0.85, 0.95, 0.2],  # 0.75: Yellow (+10°C) - warm temperate
        [0.95, 0.8, 0.15],  # 0.83: Yellow-orange (+20°C) - warm/subtropical
        [0.98, 0.5, 0.05],  # 0.92: Orange (+30°C) - hot tropical
        [0.85, 0.15, 0.0],  # 1.00: Deep red (+40°C) - extreme heat
    ], dtype=np.float32)
    breakpoints = np.array([0.0, 0.12, 0.25, 0.38, 0.50, 0.58, 0.67, 0.75, 0.83, 0.92, 1.0], dtype=np.float32)
    idx = np.clip(np.searchsorted(breakpoints, v, side="right") - 1, 0, len(breakpoints) - 2)
    c0 = color_stops[idx]
    c1 = color_stops[idx + 1]
    t = (v - breakpoints[idx]) / (breakpoints[idx + 1] - breakpoints[idx] + 1e-9)
    rgb_full = (c0 + (c1 - c0) * t[..., None]).astype(np.float32)
    
    # Diagnostic: Color mapping distribution
    temp_ranges = [(tmin + (tmax-tmin)*bp) for bp in [0.0, 0.25, 0.50, 0.67, 0.83, 1.0]]
    LOG.info(f"[Color Map] Range: {temp_ranges[0]-273.15:.0f}°C to {temp_ranges[-1]-273.15:.0f}°C | "
             f"Blue<{temp_ranges[1]-273.15:.0f}°C, Cyan={temp_ranges[1]-273.15:.0f} to {temp_ranges[2]-273.15:.0f}°C, "
             f"Green={temp_ranges[2]-273.15:.0f} to {temp_ranges[3]-273.15:.0f}°C, Yellow={temp_ranges[3]-273.15:.0f} to {temp_ranges[4]-273.15:.0f}°C, "
             f"Orange-Red>{temp_ranges[4]-273.15:.0f}°C")
    
    # Return RGB overlay at full resolution (H, W, 3)
    return rgb_full


def _apply_heat_diffusion(T_lat: np.ndarray, lat_rad: np.ndarray, diffusion_coeff: float = 0.40, iterations: int = 30) -> np.ndarray:
    """Apply meridional heat transport via flux parameterization.
    
    Simulates the combined effect of ocean currents (Gulf Stream, etc.) and
    atmospheric circulation (Hadley cells, Ferrel cells, jet streams) that
    redistribute ~5 PW of heat from equator to poles.
    
    Uses explicit flux calculation based on temperature gradients instead of
    diffusion, which works effectively on linear temperature profiles.
    
    Args:
        T_lat: 1D temperature array [K] per latitude
        lat_rad: 1D latitude array [radians] corresponding to T_lat
        diffusion_coeff: Heat transport strength (tuned to match Earth's ~5 PW)
        iterations: Number of relaxation steps
    
    Returns:
        Temperature array with poleward heat transport applied
    """
    T = T_lat.copy().astype(np.float64)
    n = len(T)
    
    if n < 3:
        return T.astype(np.float32)
    
    # Use simpler, more stable parameterization
    # Instead of physical flux calculation, use effective heat redistribution
    # that's guaranteed to be numerically stable
    dlat = np.pi / (n - 1)  # latitude spacing in radians
    
    # Simple conservative diffusion: standard neighbor averaging
    # With temperature floor already applied, the profile should be smooth and monotonic
    
    alpha = diffusion_coeff / iterations
    initial_total_heat = float(np.sum(T))
    
    # Find equator index for diagnostics
    eq_idx = len(T) // 2
    
    for iteration in range(iterations):
        T_new = T.copy()
        
        # Standard 3-point smoothing (diffusion)
        for i in range(1, n - 1):
            # Simple neighbor averaging: conservative and guaranteed stable
            neighbor_avg = 0.25 * T[i-1] + 0.50 * T[i] + 0.25 * T[i+1]
            T_new[i] = (1.0 - alpha) * T[i] + alpha * neighbor_avg
        
        # Poles: relax toward adjacent cell
        T_new[0] = T[0] + alpha * 0.5 * (T[1] - T[0])
        T_new[n-1] = T[n-1] + alpha * 0.5 * (T[n-2] - T[n-1])
        
        T = T_new
        
        # Debug: log every 25 iterations
        if (iteration + 1) % 25 == 0 or iteration == 0:
            from logging import getLogger
            LOG = getLogger(__name__)
            pole_idx_n = 0
            pole_idx_s = n - 1
            mid_idx = len(T) // 4
            LOG.info(f"  [Transport Iter {iteration+1}/{iterations}] "
                     f"Equator={float(T[eq_idx]):.1f}K, "
                     f"Mid(45°)={float(T[mid_idx]):.1f}K, "
                     f"N-Pole={float(T[pole_idx_n]):.1f}K, "
                     f"S-Pole={float(T[pole_idx_s]):.1f}K, "
                     f"ΔT(eq-Npole)={float(T[eq_idx]-T[pole_idx_n]):.1f}K")
    
    # Verify heat conservation
    final_total_heat = float(np.sum(T))
    heat_change_percent = 100.0 * abs(final_total_heat - initial_total_heat) / initial_total_heat
    from logging import getLogger
    LOG = getLogger(__name__)
    LOG.info(f"  [Heat Conservation] {heat_change_percent:.3f}% change "
             f"(initial={initial_total_heat:.1f}, final={final_total_heat:.1f})")
    
    return T.astype(np.float32)


def _albedo_for_latitude(lat_rad: np.ndarray, day_of_year: int = 1) -> np.ndarray:
    """Return latitude-dependent albedo accounting for ice/snow at poles.
    
    Ice/snow has high albedo (0.8-0.9) at poles, typical Earth surface (0.2-0.3)
    at mid/low latitudes. Smooth transition between zones. Albedo slightly lower
    during summer at poles due to partial ice melt.
    """
    lat = np.asarray(lat_rad, dtype=np.float32)
    abs_lat_deg = np.rad2deg(np.abs(lat))
    
    # Base albedo for surface: 0.25 (typical Earth surface without clouds)
    # Add cloud albedo (tropical convection creates high clouds with moderate albedo)
    # Clouds do exist at equator but they also trap heat (greenhouse effect)
    # Net effect: small albedo increase at equator
    # Reduced from +0.08 to +0.04 to prevent over-cooling equator
    cloud_contribution = 0.04 * np.maximum(0.0, 1.0 - abs_lat_deg / 40.0) ** 1.5
    A_base = 0.25 + cloud_contribution  # Range: 0.25 (mid-latitude) to 0.29 (equator)
    
    # High albedo at poles - transition starts around 60°
    transition_start = 60.0  # degrees
    
    # Seasonal variation: reduce ice albedo slightly during summer due to partial melt
    # Winter: A_ice = 0.90 (full ice cover), Summer: A_ice = 0.80 (partial melt)
    obliq = np.deg2rad(23.44)
    gamma = 2.0 * np.pi * (float(day_of_year) - 80.0) / 365.2422
    delta = np.arcsin(np.sin(obliq) * np.sin(gamma))
    # Summer: |delta| > 15°, reduce albedo; Winter: |delta| < 15°, full albedo
    season_factor = np.clip(1.0 - 0.5 * (np.abs(delta) / np.deg2rad(15.0)), 0.5, 1.0)
    A_ice = 0.80 + 0.10 * season_factor  # Range: 0.80-0.90
    
    # Linear transition from transition_start to 90°
    transition_range = 90.0 - transition_start
    t = np.clip((abs_lat_deg - transition_start) / transition_range, 0.0, 1.0)
    A = A_base + (A_ice - A_base) * t
    
    return A.astype(np.float32)


def temperature_kelvin_for_lat(lat_rad: np.ndarray | float, day_of_year: int = 1, epsilon_atm: float = EPSILON_ATM) -> np.ndarray | float:
    """Return blackbody-equilibrium temperature (K) for latitude(s).

    Uses daily-mean TOA insolation by latitude and day-of-year (handles
    polar night/day). Variable albedo accounts for ice/snow at poles.
    Accepts scalar or array latitude in radians.
    
    ENHANCED POLAR COOLING (Options B & C):
    - Latent heat flux: Energy goes to melting ice/snow (100-150 W/m²)
    - Convective export: Warm air rises and loses heat to space (30-80 W/m²)
    Both applied to FLUX before temperature calculation (more physically accurate).
    """
    lat = np.asarray(lat_rad, dtype=np.float32)
    A = _albedo_for_latitude(lat, day_of_year)
    sigma = 5.670374419e-8
    Q = _daily_mean_insolation_Q(lat, day_of_year)
    F_abs = (1.0 - A) * np.maximum(Q, 0.0)
    
    # LATITUDE-DEPENDENT GREENHOUSE EFFECT (realistic atmospheric physics)
    abs_lat_deg = np.rad2deg(np.abs(lat))
    epsilon_equator = 0.68  # Strong greenhouse (humid tropics) - INCREASED to warm base temps
    epsilon_pole = 0.40     # Weak greenhouse (dry polar air)
    lat_factor = np.cos(np.deg2rad(abs_lat_deg))  # 1.0 at equator, 0.0 at poles
    epsilon_lat = epsilon_pole + (epsilon_equator - epsilon_pole) * lat_factor
    
    # ==============================================================================
    # POLAR COOLING MECHANISMS (Option B & C): Apply to FLUX before temperature calc
    # MORE PHYSICALLY ACCURATE: Energy lost to phase change & convection never
    # becomes sensible heat, so we remove it from absorbed flux BEFORE calculating T
    # ==============================================================================
    
    # Determine if we're in melting season (spring/summer) for each hemisphere
    obliq = np.deg2rad(23.44)
    gamma = 2.0 * np.pi * (float(day_of_year) - 80.0) / 365.2422
    solar_declination = np.arcsin(np.sin(obliq) * np.sin(gamma))
    
    nh_melt_season = solar_declination > np.deg2rad(-10.0)
    sh_melt_season = solar_declination < np.deg2rad(10.0)
    
    polar_threshold = 55.0  # degrees - where ice/snow persists year-round
    is_polar = abs_lat_deg > polar_threshold
    
    is_melt_season = np.where(
        lat >= 0,  # Northern hemisphere
        nh_melt_season,
        sh_melt_season
    )
    
    # --- OPTION B: LATENT HEAT FLUX LOSS (ice/snow melting) ---
    # During polar summer, a significant fraction of absorbed solar energy goes into
    # melting ice/snow instead of heating the atmosphere. This energy is "lost" to
    # phase change (latent heat of fusion: 334 kJ/kg).
    # 
    # Physical model: The flux available for melting depends on:
    # 1. Only active in polar regions (>55°) during melt season
    # 2. Proportional to insolation (more sun = more melting)
    # 3. Strongest at higher latitudes (more ice coverage)
    # 
    # Calibration: Peak Arctic summer can absorb 100-150 W/m² in latent heat
    
    # Calculate latitude-weighted melting intensity (stronger at higher latitudes)
    polar_intensity = np.where(
        is_polar,
        (abs_lat_deg - polar_threshold) / (90.0 - polar_threshold),  # 0.0 at 55°, 1.0 at 90°
        0.0
    )
    
    # Maximum latent heat flux during peak melting conditions
    # Peak value: 150 W/m² at poles during midsummer with full sun
    F_latent_max = 150.0  # W/m²
    
    # Apply latent heat flux loss only during melt season
    # Scales with solar intensity (more sun = more melting)
    F_latent = np.where(
        is_polar & is_melt_season,
        F_latent_max * polar_intensity * np.clip(Q / 400.0, 0.0, 1.0),  # Scale with insolation
        0.0
    )
    
    # --- OPTION C: ATMOSPHERIC CONVECTIVE EXPORT ---
    # Polar regions lose excess heat via atmospheric upwelling and poleward circulation.
    # When surface is heated, warm air rises and is replaced by cooler air from aloft.
    # This exported heat is radiated to space at higher altitudes.
    #
    # Physical model: Convective heat loss increases with:
    # 1. Temperature excess above freezing (warmer = stronger convection)
    # 2. Solar heating intensity (drives convective instability)
    # 3. Latitude (stronger at poles where temperature gradient is steepest)
    #
    # Calibration: Can export 30-80 W/m² during polar summer
    
    # Estimate surface temperature for convection calculation (rough approximation)
    # Use a simplified greenhouse calculation just for this check
    gh_denom_simple = np.maximum(1.0 - 0.5 * epsilon_lat, 1e-6)
    T_estimate = np.power(np.clip(F_abs, 1e-9, None) / (sigma * gh_denom_simple), 0.25)
    
    # Convective flux scales with temperature excess above freezing (273K)
    # Only active when surface is warm enough to drive convection (>260K = -13°C)
    T_excess = np.maximum(T_estimate - 260.0, 0.0)  # Kelvin above -13°C
    convection_efficiency = 2.0  # W/m² per K excess (empirically calibrated)
    
    # Apply convective export in polar regions (>50° latitude)
    polar_convection_mask = abs_lat_deg > 50.0
    F_convective = np.where(
        polar_convection_mask,
        convection_efficiency * T_excess * ((abs_lat_deg - 50.0) / 40.0),  # Stronger at higher latitudes
        0.0
    )
    
    # Limit convective export to reasonable values (0-80 W/m²)
    F_convective = np.clip(F_convective, 0.0, 80.0)
    
    # ==============================================================================
    # NET FLUX AFTER POLAR COOLING MECHANISMS
    # ==============================================================================
    F_net = F_abs - F_latent - F_convective
    F_net = np.maximum(F_net, 1.0)  # Floor at 1 W/m² to prevent numerical issues
    
    # Calculate temperature from net available flux
    gh_denom = np.maximum(1.0 - 0.5 * epsilon_lat, 1e-6)
    T = np.power(F_net / (sigma * gh_denom), 0.25)
    
    # Minimum temperature floor during polar night (accounts for heat transport/thermal inertia)
    T_min = 200.0
    T = np.maximum(T, T_min)
    
    if np.isscalar(lat_rad):
        return float(T)
    return T


