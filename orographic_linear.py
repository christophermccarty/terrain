"""Smith--Barstad linear-theory orographic precipitation diagnostics.

This module is an independent, NumPy-only implementation of the transfer
function in Smith & Barstad (2004).  It intentionally returns an *anomaly*;
the host model retains ownership of background precipitation and its water
budget.  The first use is therefore offline validation against named
windward/leeward pairs, not a hidden second rain scheme in the main loop.

The formulation is adapted for PlanetSim from the published linear theory and
is independently expressed here.  It is compatible with the MIT-licensed
reference implementation maintained by fastscape-lem.
"""
from __future__ import annotations

import numpy as np


def smith_barstad_precipitation_anomaly(
    elevation_m: np.ndarray,
    *,
    dx_m: float,
    dy_m: float,
    wind_u_m_s: float,
    wind_v_m_s: float,
    latitude_deg: float,
    moist_stability_s: float = 0.006,
    vapor_scale_height_m: float = 2500.0,
    uplift_sensitivity_kg_m3: float = 0.002,
    conversion_time_s: float = 1000.0,
    fallout_time_s: float = 1000.0,
    pad_cells: int | None = None,
) -> np.ndarray:
    """Return linear-theory terrain precipitation anomaly in mm/day.

    ``wind_u_m_s`` and ``wind_v_m_s`` must be a locally representative,
    approximately uniform upstream wind.  The caller is responsible for
    tiling a global domain; a single global FFT is deliberately not presented
    as valid for a spatially varying planet-scale wind field.
    """
    h = np.asarray(elevation_m, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 2:
        raise ValueError("elevation_m must be a two-dimensional grid")
    if dx_m <= 0.0 or dy_m <= 0.0:
        raise ValueError("grid spacings must be positive")
    if min(moist_stability_s, vapor_scale_height_m, uplift_sensitivity_kg_m3,
           conversion_time_s, fallout_time_s) <= 0.0:
        raise ValueError("linear-theory physical parameters must be positive")

    # Zero terrain must remain exactly a zero anomaly; this also avoids a
    # pointless FFT in the most common unit-test fixture.
    if not np.any(h):
        return np.zeros_like(h, dtype=np.float32)

    pad = int(pad_cells) if pad_cells is not None else min(200, max(h.shape) // 2)
    if pad < 0:
        raise ValueError("pad_cells cannot be negative")
    hp = np.pad(h, pad, mode="constant") if pad else h
    ny, nx = hp.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy_m)[:, None]
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx_m)[None, :]
    sigma = kx * float(wind_u_m_s) + ky * float(wind_v_m_s)
    coriolis = 2.0 * 7.2921159e-5 * np.sin(np.deg2rad(float(latitude_deg)))

    numerator = moist_stability_s ** 2 - sigma ** 2
    denominator = sigma ** 2 - coriolis ** 2
    numerator = np.maximum(numerator, 0.0)
    # Preserve the sign around the inertial frequency while avoiding division
    # by zero.  The sign convention is the upward-propagating wave branch.
    denominator = np.where(
        np.abs(denominator) < np.finfo(np.float64).eps,
        np.copysign(np.finfo(np.float64).eps, denominator + 1e-30),
        denominator,
    )
    vertical_wavenumber = np.sign(sigma)
    vertical_wavenumber[vertical_wavenumber == 0.0] = 1.0
    vertical_wavenumber *= np.sqrt(
        np.abs(numerator / denominator * (kx * kx + ky * ky))
    )

    terrain_hat = np.fft.fft2(hp)
    transfer = (
        uplift_sensitivity_kg_m3 * 1j * sigma
        / (
            (1.0 - 1j * vapor_scale_height_m * vertical_wavenumber)
            * (1.0 + 1j * sigma * conversion_time_s)
            * (1.0 + 1j * sigma * fallout_time_s)
        )
    )
    # kg m-2 s-1 is numerically equivalent to mm s-1 of liquid water.
    anomaly = np.real(np.fft.ifft2(transfer * terrain_hat)) * 86400.0
    if pad:
        anomaly = anomaly[pad:-pad, pad:-pad]
    return anomaly.astype(np.float32, copy=False)
