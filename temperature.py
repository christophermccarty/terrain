"""Temperature overlay generation for equirectangular maps.

Produces a transparent-friendly RGB overlay (float in [0,1]) based on a
latitudinal blackbody equilibrium approximation under Earth/Sun-like insolation.
"""

from __future__ import annotations

import numpy as np

# Simple one-layer gray-atmosphere greenhouse tuned for Earth
# ε≈0.77 gives global-mean ~288 K from ~255 K effective temperature
EPSILON_ATM = 0.77


def _daily_mean_insolation_Q(lat_rad: np.ndarray, day_of_year: int, S0: float = 1361.0) -> np.ndarray:
    """Daily-mean TOA insolation Q(φ, δ) in W/m^2.

    Uses standard astronomy formula with solar declination δ that varies by
    day-of-year. Handles polar night/day. Vectorized for `lat_rad` arrays.
    Special handling for exact poles to avoid numerical precision issues.
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
    return np.where(pole_mask, Q_pole, Q_standard)


def _coarse_shape(H: int, W: int, block_size: int) -> tuple[int, int]:
    bs = max(1, int(block_size))
    Hc = max(1, (H + bs - 1) // bs)
    Wc = max(1, (W + bs - 1) // bs)
    return Hc, Wc


def _upsample_repeat(field: np.ndarray, H: int, W: int, block_size: int) -> np.ndarray:
    bs = max(1, int(block_size))
    up = np.repeat(np.repeat(field, bs, axis=0), bs, axis=1)
    return up[:H, :W]


def generate_temperature_overlay(height: int, width: int, day_of_year: int = 80, epsilon_atm: float = EPSILON_ATM, block_size: int = 3) -> np.ndarray:
    """Return an (H,W,3) float32 RGB overlay in [0,1] for given map size.

    - Insolation: daily-mean TOA Q(φ, δ) with S0=1361 W/m^2.
    - Variable albedo: ice/snow at poles (A~0.85), typical surface at mid/low latitudes (A~0.25).
    - Blackbody equilibrium with Stefan-Boltzmann σ; no transport/atmosphere.
    - Mapped to a blue→cyan→yellow→red gradient.
    """
    h = int(height)
    w = int(width)
    bs = max(1, int(block_size))
    hc, wc = _coarse_shape(h, w, bs)

    # Latitude centers per row in radians: +π/2 (north) → -π/2 (south)
    lat = (0.5 - (np.arange(hc, dtype=np.float32) + 0.5) / hc) * np.pi

    # Blackbody equilibrium from daily-mean absorbed flux
    # Variable albedo: ice/snow at poles (high albedo), typical surface at mid/low latitudes
    A = _albedo_for_latitude(lat, day_of_year)
    sigma = 5.670374419e-8
    Q = _daily_mean_insolation_Q(lat, day_of_year)
    F_abs = (1.0 - A) * np.maximum(Q, 0.0)
    gh_denom = np.maximum(1.0 - 0.5 * float(epsilon_atm), 1e-6)
    T_lat = np.power(np.clip(F_abs, 1e-9, None) / (sigma * gh_denom), 0.25)
    # Minimum temperature floor during polar night (realistic minimum ~200 K)
    T_lat = np.maximum(T_lat, 200.0)

    # Normalize to a broad 150–320 K range for coloring
    tmin, tmax = 150.0, 320.0
    v = np.clip((T_lat - tmin) / (tmax - tmin), 0.0, 1.0).astype(np.float32)
    v2d = np.repeat(v[:, None], wc, axis=1)

    # Piecewise-linear color map: Blue→Cyan→Yellow→Red
    color_stops = np.array(
        [[0.0, 0.1, 0.8], [0.0, 0.8, 1.0], [1.0, 1.0, 0.2], [0.8, 0.1, 0.0]],
        dtype=np.float32,
    )
    breakpoints = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    idx = np.clip(np.searchsorted(breakpoints, v2d, side="right") - 1, 0, len(breakpoints) - 2)
    c0 = color_stops[idx]
    c1 = color_stops[idx + 1]
    t = (v2d - breakpoints[idx]) / (breakpoints[idx + 1] - breakpoints[idx] + 1e-9)
    coarse = (c0 + (c1 - c0) * t[..., None]).astype(np.float32)
    return _upsample_repeat(coarse, h, w, bs)


def _albedo_for_latitude(lat_rad: np.ndarray, day_of_year: int = 80) -> np.ndarray:
    """Return latitude-dependent albedo accounting for ice/snow at poles.
    
    Ice/snow has high albedo (0.7-0.85) at poles, typical Earth surface (0.2-0.3)
    at mid/low latitudes. Smooth transition between zones. Albedo slightly lower
    during summer at poles due to partial ice melt.
    """
    lat = np.asarray(lat_rad, dtype=np.float32)
    abs_lat_deg = np.rad2deg(np.abs(lat))
    
    # Base albedo: 0.25 for mid/low latitudes (typical Earth surface)
    # High albedo at poles - transition starts around 60°
    # Smooth transition: 0.25 at 0° → A_ice at 90°
    A_base = 0.25
    transition_start = 60.0  # degrees
    
    # Seasonal variation: reduce ice albedo slightly during summer due to partial melt
    # Winter: A_ice = 0.85 (full ice cover), Summer: A_ice = 0.75 (partial melt)
    obliq = np.deg2rad(23.44)
    gamma = 2.0 * np.pi * (float(day_of_year) - 80.0) / 365.2422
    delta = np.arcsin(np.sin(obliq) * np.sin(gamma))
    # Summer: |delta| > 15°, reduce albedo; Winter: |delta| < 15°, full albedo
    season_factor = np.clip(1.0 - 0.5 * (np.abs(delta) / np.deg2rad(15.0)), 0.5, 1.0)
    A_ice = 0.75 + 0.10 * season_factor  # Range: 0.75-0.85
    
    # Linear transition from transition_start to 90°
    transition_range = 90.0 - transition_start
    t = np.clip((abs_lat_deg - transition_start) / transition_range, 0.0, 1.0)
    A = A_base + (A_ice - A_base) * t
    
    return A.astype(np.float32)


def temperature_kelvin_for_lat(lat_rad: np.ndarray | float, day_of_year: int = 80, epsilon_atm: float = EPSILON_ATM) -> np.ndarray | float:
    """Return blackbody-equilibrium temperature (K) for latitude(s).

    Uses daily-mean TOA insolation by latitude and day-of-year (handles
    polar night/day). Variable albedo accounts for ice/snow at poles.
    Accepts scalar or array latitude in radians.
    
    During polar night, applies minimum temperature floor to account for
    heat transport and thermal inertia (realistic minimum ~200 K).
    """
    lat = np.asarray(lat_rad, dtype=np.float32)
    A = _albedo_for_latitude(lat, day_of_year)
    sigma = 5.670374419e-8
    Q = _daily_mean_insolation_Q(lat, day_of_year)
    F_abs = (1.0 - A) * np.maximum(Q, 0.0)
    gh_denom = np.maximum(1.0 - 0.5 * float(epsilon_atm), 1e-6)
    T = np.power(np.clip(F_abs, 1e-9, None) / (sigma * gh_denom), 0.25)
    
    # Minimum temperature floor during polar night (accounts for heat transport/thermal inertia)
    # During polar night, Q ≈ 0, but real poles retain heat (Antarctica winter ~200-220 K)
    T_min = 200.0  # Realistic minimum for polar winter
    T = np.maximum(T, T_min)
    
    if np.isscalar(lat_rad):
        return float(T)
    return T


