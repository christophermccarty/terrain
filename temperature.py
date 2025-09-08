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
    """
    lat = np.asarray(lat_rad, dtype=np.float32)
    obliq = np.deg2rad(23.44)
    gamma = 2.0 * np.pi * (float(day_of_year) - 80.0) / 365.2422
    delta = np.arcsin(np.sin(obliq) * np.sin(gamma))
    cosH0 = -np.tan(lat) * np.tan(delta)
    H0 = np.arccos(np.clip(cosH0, -1.0, 1.0))
    H0 = np.where(cosH0 <= -1.0, np.pi, H0)  # 24h daylight
    H0 = np.where(cosH0 >= 1.0, 0.0, H0)     # polar night
    return (S0 / np.pi) * (H0 * np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.sin(H0))


def generate_temperature_overlay(height: int, width: int, day_of_year: int = 80, epsilon_atm: float = EPSILON_ATM) -> np.ndarray:
    """Return an (H,W,3) float32 RGB overlay in [0,1] for given map size.

    - Insolation: daily-mean TOA Q(φ, δ) with S0=1361 W/m^2; albedo A=0.3.
    - Blackbody equilibrium with Stefan-Boltzmann σ; no transport/atmosphere.
    - Mapped to a blue→cyan→yellow→red gradient.
    """
    h = int(height)
    w = int(width)

    # Latitude centers per row in radians: +π/2 (north) → -π/2 (south)
    lat = (0.5 - (np.arange(h, dtype=np.float32) + 0.5) / h) * np.pi

    # Blackbody equilibrium from daily-mean absorbed flux
    A = 0.3
    sigma = 5.670374419e-8
    Q = _daily_mean_insolation_Q(lat, day_of_year)
    F_abs = (1.0 - A) * np.maximum(Q, 0.0)
    gh_denom = np.maximum(1.0 - 0.5 * float(epsilon_atm), 1e-6)
    T_lat = np.power(np.clip(F_abs, 1e-9, None) / (sigma * gh_denom), 0.25)

    # Normalize to a broad 150–320 K range for coloring
    tmin, tmax = 150.0, 320.0
    v = np.clip((T_lat - tmin) / (tmax - tmin), 0.0, 1.0).astype(np.float32)
    v2d = np.repeat(v[:, None], w, axis=1)

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
    return (c0 + (c1 - c0) * t[..., None]).astype(np.float32)


def temperature_kelvin_for_lat(lat_rad: np.ndarray | float, day_of_year: int = 80, epsilon_atm: float = EPSILON_ATM) -> np.ndarray | float:
    """Return blackbody-equilibrium temperature (K) for latitude(s).

    Uses daily-mean TOA insolation by latitude and day-of-year (handles
    polar night/day). Accepts scalar or array latitude in radians.
    """
    lat = np.asarray(lat_rad, dtype=np.float32)
    A = 0.3
    sigma = 5.670374419e-8
    Q = _daily_mean_insolation_Q(lat, day_of_year)
    F_abs = (1.0 - A) * np.maximum(Q, 0.0)
    gh_denom = np.maximum(1.0 - 0.5 * float(epsilon_atm), 1e-6)
    T = np.power(np.clip(F_abs, 1e-9, None) / (sigma * gh_denom), 0.25)
    if np.isscalar(lat_rad):
        return float(T)
    return T


