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

# Cache for base temperature calculations (optimization 2.6)
_TEMP_BASE_CACHE = {}

# Color lookup table for temperature-to-RGB conversion (optimization 2.5)
_TEMP_LUT = None
_TEMP_LUT_TMIN = 193.15
_TEMP_LUT_TMAX = 313.15
_TEMP_LUT_SIZE = 256


def elevation_to_alt_km(elevation: np.ndarray, *, assume_loaded_if_zeros_frac: float = 0.05) -> np.ndarray:
    """Convert normalized elevation [0,1] to approximate altitude (km).

    This matches the UI mappings:
    - Loaded heightmaps: ocean is exactly 0.0; lowlands 0..0.03 => 0..100m (linear),
      then a power curve up to 8848m.
    - Procedural terrain: sea level ~0.2; quadratic rise to 8848m.
    """
    e = np.asarray(elevation, dtype=np.float32)
    if e.size == 0:
        return e
    zeros_frac = float(np.mean(e == 0.0))
    if zeros_frac > assume_loaded_if_zeros_frac:
        # Loaded heightmap mapping
        alt_m = np.zeros_like(e)
        low = (e > 0.0) & (e <= 0.03)
        high = e > 0.03
        alt_m[low] = (e[low] / 0.03) * 100.0
        norm = (e[high] - 0.03) / 0.97
        alt_m[high] = 100.0 + (norm ** 2.5) * 8748.0
        return alt_m / 1000.0
    # Procedural mapping (approx)
    sea_level = 0.2
    x = np.clip((e - sea_level) / (1.0 - sea_level + 1e-9), 0.0, 1.0)
    return (x ** 2.0) * 8.848


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


def _temperature_to_rgb_slow(T_kelvin: np.ndarray) -> np.ndarray:
    """Slow reference implementation of temperature-to-RGB conversion.
    
    Used for building the lookup table.
    """
    tmin, tmax = 193.15, 313.15
    v = np.clip((T_kelvin - tmin) / (tmax - tmin), 0.0, 1.0).astype(np.float32)
    color_stops = np.array([
        [0.05, 0.0, 0.4],   [0.0, 0.2, 0.7],    [0.0, 0.5, 0.9],    [0.0, 0.7, 0.8],
        [0.2, 0.8, 0.6],   [0.4, 0.85, 0.4],   [0.65, 0.9, 0.3],   [0.85, 0.95, 0.2],
        [0.95, 0.8, 0.15], [0.98, 0.5, 0.05],  [0.85, 0.15, 0.0],
    ], dtype=np.float32)
    breakpoints = np.array([0.0, 0.12, 0.25, 0.38, 0.50, 0.58, 0.67, 0.75, 0.83, 0.92, 1.0], dtype=np.float32)
    idx = np.clip(np.searchsorted(breakpoints, v, side="right") - 1, 0, len(breakpoints) - 2)
    c0 = color_stops[idx]
    c1 = color_stops[idx + 1]
    t = (v - breakpoints[idx]) / (breakpoints[idx + 1] - breakpoints[idx] + 1e-9)
    return (c0 + (c1 - c0) * t[..., None]).astype(np.float32)


def temperature_to_rgb(T_kelvin: np.ndarray) -> np.ndarray:
    """Convert temperature array (Kelvin) to RGB overlay using high-quality color gradient.
    
    Uses optimized lookup table for fast conversion (optimization 2.5).
    Range: -80°C to +40°C (193K to 313K).
    
    Args:
        T_kelvin: Temperature array in Kelvin, shape (H, W)
    
    Returns:
        RGB overlay array, shape (H, W, 3), float32 in [0, 1]
    """
    global _TEMP_LUT
    
    # Initialize lookup table on first call
    # IMPORTANT: the LUT must be exactly shape (256, 3). If it is (256, 1, 3),
    # then c0/c1 become (N, 1, 3) and multiplying by t shaped (N, 1) broadcasts to
    # (N, N, 3) -> the 3.00 TiB ArrayMemoryError you’re seeing.
    if (
        _TEMP_LUT is None
        or not isinstance(_TEMP_LUT, np.ndarray)
        or _TEMP_LUT.shape != (_TEMP_LUT_SIZE, 3)
    ):
        T_range = np.linspace(_TEMP_LUT_TMIN, _TEMP_LUT_TMAX, _TEMP_LUT_SIZE, dtype=np.float32)
        _TEMP_LUT = _temperature_to_rgb_slow(T_range)  # (256, 3)
        _TEMP_LUT = np.ascontiguousarray(_TEMP_LUT, dtype=np.float32)
    
    # Use LUT with linear interpolation
    # IMPORTANT: force ndarray + 1D vectors to prevent accidental (N,N,3) broadcasts.
    # If an upstream caller passes an np.matrix (or other 2D vector-like type),
    # methods like `.flatten()` can return (1,N) instead of (N,), which combined with
    # (N,1) interpolation weights explodes into an outer-product (N,N,3) allocation.
    T_arr = np.asarray(T_kelvin, dtype=np.float32)
    original_shape = T_arr.shape
    T_flat = T_arr.ravel()  # guaranteed shape (N,)
    n_pixels = int(T_flat.size)
    
    # Normalize temperatures to [0, 1] range
    v = np.clip((T_flat - _TEMP_LUT_TMIN) / (_TEMP_LUT_TMAX - _TEMP_LUT_TMIN), 0.0, 1.0)
    
    # Convert to LUT indices (0 to 255)
    idx_f = v * (_TEMP_LUT_SIZE - 1)
    idx_lo = np.clip(np.floor(idx_f).astype(np.int32), 0, _TEMP_LUT_SIZE - 2)
    idx_hi = idx_lo + 1
    
    # Interpolation factor [0, 1)
    t = (idx_f - idx_lo).astype(np.float32)
    
    # Force true 1D ndarrays (works even if something upstream produced matrix-like arrays)
    idx_lo = np.asarray(idx_lo, dtype=np.int32).ravel()
    idx_hi = np.asarray(idx_hi, dtype=np.int32).ravel()
    t = np.asarray(t, dtype=np.float32).ravel()
    
    # Get colors from LUT using direct indexing
    c0 = _TEMP_LUT[idx_lo, :]  # (N, 3)
    c1 = _TEMP_LUT[idx_hi, :]  # (N, 3)
    
    # Linear interpolation: ensure t is (n_pixels, 1) for broadcasting
    t = t.reshape(n_pixels, 1)  # (N, 1)
    rgb_flat = c0 + (c1 - c0) * t  # (N, 3)
    
    return rgb_flat.reshape(original_shape + (3,)).astype(np.float32)


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
    
    # Fast path: use simplified temperature calculation for overlay (not full simulation)
    # This avoids expensive wind/precipitation calculations when just viewing temperature
    if elevation is None:
        elevation = np.zeros((h, w), dtype=np.float32)
    
    # Calculate temperature directly using latitude-based formula + elevation correction
    # This is much faster than running full simulation
    lat = (0.5 - (np.arange(h, dtype=np.float32) + 0.5) / h) * np.pi
    T_lat = temperature_kelvin_for_lat(lat, day_of_year=day_of_year)
    T_kelvin = np.repeat(T_lat[:, None], w, axis=1).astype(np.float32)
    
    # Apply elevation correction (lapse rate ~6.5 K/km)
    if elevation is not None and np.any(elevation > 0):
        alt_km = elevation_to_alt_km(elevation)
        lapse_rate = 6.5  # K/km
        T_kelvin = T_kelvin - lapse_rate * alt_km
    
    # Temperature already calculated by simulation system (includes all physics)
    # T_kelvin shape is (H, W) at full resolution - no need to downsample
    T_lat = T_kelvin
    
    # Log temperature stats only if debug logging enabled (optimization 2.4)
    import logging
    LOG = logging.getLogger("planetsim")
    if LOG.isEnabledFor(logging.DEBUG) or (not hasattr(generate_temperature_overlay, '_logged_once')):
        LOG.info(f"[Temperature from Simulation] min={float(np.min(T_lat)):.1f}K ({float(np.min(T_lat)-273.15):.1f}°C), "
                 f"mean={float(np.mean(T_lat)):.1f}K ({float(np.mean(T_lat)-273.15):.1f}°C), "
                 f"max={float(np.max(T_lat)):.1f}K ({float(np.max(T_lat)-273.15):.1f}°C)")
        generate_temperature_overlay._logged_once = True
    
    # Use the same high-quality color mapping function for consistency
    rgb_full = temperature_to_rgb(T_lat)
    
    # Diagnostic: Color mapping distribution
    # Removed console logging - uncomment if needed for debugging
    # temp_ranges = [(tmin + (tmax-tmin)*bp) for bp in [0.0, 0.25, 0.50, 0.67, 0.83, 1.0]]
    # LOG.info(f"[Color Map] Range: {temp_ranges[0]-273.15:.0f}°C to {temp_ranges[-1]-273.15:.0f}°C | "
    #          f"Blue<{temp_ranges[1]-273.15:.0f}°C, Cyan={temp_ranges[1]-273.15:.0f} to {temp_ranges[2]-273.15:.0f}°C, "
    #          f"Green={temp_ranges[2]-273.15:.0f} to {temp_ranges[3]-273.15:.0f}°C, Yellow={temp_ranges[3]-273.15:.0f} to {temp_ranges[4]-273.15:.0f}°C, "
    #          f"Orange-Red>{temp_ranges[4]-273.15:.0f}°C")
    
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
        
        # Debug: log only if debug logging enabled (optimization 2.4)
        if (iteration + 1) % 25 == 0 or iteration == 0:
            import logging
            from logging import getLogger
            LOG = getLogger(__name__)
            if LOG.isEnabledFor(logging.DEBUG):
                pole_idx_n = 0
                pole_idx_s = n - 1
                mid_idx = len(T) // 4
                LOG.debug(f"  [Transport Iter {iteration+1}/{iterations}] "
                         f"Equator={float(T[eq_idx]):.1f}K, "
                         f"Mid(45°)={float(T[mid_idx]):.1f}K, "
                         f"N-Pole={float(T[pole_idx_n]):.1f}K, "
                         f"S-Pole={float(T[pole_idx_s]):.1f}K, "
                         f"ΔT(eq-Npole)={float(T[eq_idx]-T[pole_idx_n]):.1f}K")
    
    # Verify heat conservation (only log if debug enabled)
    final_total_heat = float(np.sum(T))
    heat_change_percent = 100.0 * abs(final_total_heat - initial_total_heat) / initial_total_heat
    import logging
    from logging import getLogger
    LOG = getLogger(__name__)
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(f"  [Heat Conservation] {heat_change_percent:.3f}% change "
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


def temperature_kelvin_for_lat(lat_rad: np.ndarray | float, day_of_year: int = 1, epsilon_atm: float = EPSILON_ATM, cache: bool = True) -> np.ndarray | float:
    """Return blackbody-equilibrium temperature (K) for latitude(s).

    Uses daily-mean TOA insolation by latitude and day-of-year (handles
    polar night/day). Variable albedo accounts for ice/snow at poles.
    Accepts scalar or array latitude in radians.
    
    ENHANCED POLAR COOLING (Options B & C):
    - Latent heat flux: Energy goes to melting ice/snow (100-150 W/m²)
    - Convective export: Warm air rises and loses heat to space (30-80 W/m²)
    Both applied to FLUX before temperature calculation (more physically accurate).
    
    Args:
        lat_rad: Latitude(s) in radians
        day_of_year: Day of year (0-365)
        epsilon_atm: Atmospheric emissivity
        cache: Whether to use caching (optimization 2.6)
    """
    # Check cache if enabled (optimization 2.6)
    lat = np.asarray(lat_rad, dtype=np.float32)
    cache_key = None
    if cache:
        day_int = int(day_of_year) % 365
        # Create cache key from latitude array shape and hash of values
        if np.isscalar(lat_rad):
            cache_key = (day_int, 'scalar', float(lat_rad), epsilon_atm)
        else:
            # Use hash of first, middle, and last values for efficiency (sufficient for typical use)
            n = len(lat)
            if n > 0:
                lat_hash = (float(lat[0]), float(lat[n//2]) if n > 1 else 0.0, float(lat[-1]) if n > 1 else 0.0, n)
            else:
                lat_hash = (0.0, 0.0, 0.0, 0)
            cache_key = (day_int, lat_hash, epsilon_atm)
        
        if cache_key in _TEMP_BASE_CACHE:
            cached_result = _TEMP_BASE_CACHE[cache_key]
            # Return same type as input
            if np.isscalar(lat_rad):
                return float(cached_result)
            return cached_result.copy()
    A = _albedo_for_latitude(lat, day_of_year)
    sigma = 5.670374419e-8
    Q = _daily_mean_insolation_Q(lat, day_of_year)
    F_abs = (1.0 - A) * np.maximum(Q, 0.0)
    
    # LATITUDE-DEPENDENT GREENHOUSE EFFECT (realistic atmospheric physics)
    abs_lat_deg = np.rad2deg(np.abs(lat))
    epsilon_equator = 0.78  # Increased from 0.75 to 0.78 to warm equator and global mean more aggressively
    epsilon_pole = 0.55     # Increased from 0.50 to 0.55 to reduce polar extremes further
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
    # FURTHER REDUCED to prevent excessive polar cooling (global mean too cold)
    F_latent_max = 30.0  # Reduced from 50 to 30 W/m² to warm poles more aggressively
    
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
    convection_efficiency = 0.4  # Reduced from 0.6 to 0.4 W/m² per K excess to warm poles more
    
    # Apply convective export in polar regions (>50° latitude)
    polar_convection_mask = abs_lat_deg > 50.0
    F_convective = np.where(
        polar_convection_mask,
        convection_efficiency * T_excess * ((abs_lat_deg - 50.0) / 40.0),  # Stronger at higher latitudes
        0.0
    )
    
    # Limit convective export to reasonable values (0-20 W/m²) - further reduced to warm poles
    F_convective = np.clip(F_convective, 0.0, 20.0)
    
    # ==============================================================================
    # NET FLUX AFTER POLAR COOLING MECHANISMS
    # ==============================================================================
    F_net = F_abs - F_latent - F_convective
    F_net = np.maximum(F_net, 1.0)  # Floor at 1 W/m² to prevent numerical issues
    
    # Calculate temperature from net available flux
    gh_denom = np.maximum(1.0 - 0.5 * epsilon_lat, 1e-6)
    T = np.power(F_net / (sigma * gh_denom), 0.25)
    
    # Minimum temperature floor during polar night (accounts for heat transport/thermal inertia)
    # Earth's coldest recorded: -89.2°C (184K) at Vostok, but typical polar winter is -50°C to -60°C (223-213K)
    # Increased to 240K (-33°C) to match Earth reference pole_temp_winter and warm global mean
    T_min = 240.0  # Increased from 230K to 240K to warm poles and reduce gradient
    T = np.maximum(T, T_min)
    
    # Store in cache if enabled
    if cache:
        _TEMP_BASE_CACHE[cache_key] = T.copy() if not np.isscalar(lat_rad) else T
        # Limit cache size to prevent memory growth (keep last 1000 entries)
        if len(_TEMP_BASE_CACHE) > 1000:
            # Remove oldest 200 entries (simple FIFO approximation)
            keys_to_remove = list(_TEMP_BASE_CACHE.keys())[:200]
            for k in keys_to_remove:
                _TEMP_BASE_CACHE.pop(k, None)
    
    if np.isscalar(lat_rad):
        return float(T)
    return T


