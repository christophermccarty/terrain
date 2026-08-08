"""Conservative primitives for PlanetSim's native pressure-column path.

The legacy atmosphere carries one humidity field plus optional two-layer
adjustments.  This module is deliberately independent of precipitation and
wind generation: it evolves three vapor reservoirs using two pressure-
coordinate interface velocities supplied by the circulation code.  Keeping
the vertical exchange in a small pure kernel makes its water budget testable
before it is coupled to cloud microphysics or radiation.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PressureColumnStep(NamedTuple):
    """One conservative three-level column update."""

    lower_humidity: np.ndarray
    midlevel_humidity: np.ndarray
    upperlevel_humidity: np.ndarray
    midlevel_temperature: np.ndarray
    upperlevel_temperature: np.ndarray
    omega_lower_mid_pa_s: np.ndarray
    omega_mid_upper_pa_s: np.ndarray
    relative_water_residual: float


def _partition_total_humidity(
    total: np.ndarray,
    midlevel: np.ndarray | None,
    upperlevel: np.ndarray | None,
    *,
    midlevel_fraction: float,
    upperlevel_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return non-negative lower/mid/upper reservoirs summing to ``total``."""
    total = np.clip(np.asarray(total, dtype=np.float64), 0.0, None)
    mid = (
        total * float(midlevel_fraction)
        if midlevel is None
        else np.clip(np.asarray(midlevel, dtype=np.float64), 0.0, None)
    )
    mid = np.minimum(mid, total)
    remaining = total - mid
    upper = (
        total * float(upperlevel_fraction)
        if upperlevel is None
        else np.clip(np.asarray(upperlevel, dtype=np.float64), 0.0, None)
    )
    upper = np.minimum(upper, remaining)
    return total - mid - upper, mid, upper


def _exchange_fraction(omega_pa_s: np.ndarray, dt_days: float, scale_pa_s: float, days: float) -> np.ndarray:
    strength = np.clip(np.abs(omega_pa_s) / scale_pa_s, 0.0, 6.0)
    return 1.0 - np.exp(-dt_days * strength / days)


def evolve_three_level_column(
    total_humidity: np.ndarray,
    lower_temperature_k: np.ndarray,
    lower_divergence_s: np.ndarray,
    midlevel_divergence_s: np.ndarray,
    upperlevel_divergence_s: np.ndarray,
    *,
    midlevel_humidity: np.ndarray | None = None,
    upperlevel_humidity: np.ndarray | None = None,
    midlevel_temperature_k: np.ndarray | None = None,
    upperlevel_temperature_k: np.ndarray | None = None,
    dt_days: float,
    lower_mid_pressure_depth_pa: float = 35_000.0,
    mid_upper_pressure_depth_pa: float = 30_000.0,
    vertical_velocity_scale_pa_s: float = 0.05,
    exchange_days: float = 2.0,
    midlevel_fraction: float = 0.25,
    upperlevel_fraction: float = 0.15,
    midlevel_height_m: float = 3500.0,
    upperlevel_height_m: float = 8000.0,
    lapse_rate_k_m: float = 6.5e-3,
    thermal_relaxation_days: float = 10.0,
    use_flux_form_exchange: bool = False,
    enforce_column_mass_closure: bool = False,
    lower_mass_fraction: float = 0.40,
    midlevel_mass_fraction: float = 0.35,
    upperlevel_mass_fraction: float = 0.25,
) -> PressureColumnStep:
    """Evolve a three-level column using pressure-coordinate continuity.

    Positive omega denotes downward pressure motion. At each interface,
    ``omega = 0.5 * dp * (div_below - div_above)`` is a centred discrete form
    of ``d omega / dp = -div(V)``. Negative omega moves vapor upward; positive
    omega moves it downward. Transfers are donor-limited and internal, so the
    combined vapor column is conserved to roundoff.
    """
    if dt_days <= 0.0 or lower_mid_pressure_depth_pa <= 0.0 or mid_upper_pressure_depth_pa <= 0.0:
        raise ValueError("dt_days and pressure depths must be positive")
    if vertical_velocity_scale_pa_s <= 0.0 or exchange_days <= 0.0 or thermal_relaxation_days <= 0.0:
        raise ValueError("vertical velocity scale and time scales must be positive")
    if not 0.0 <= midlevel_fraction <= 1.0 or not 0.0 <= upperlevel_fraction <= 1.0:
        raise ValueError("initial layer fractions must lie in [0, 1]")

    total = np.asarray(total_humidity, dtype=np.float64)
    lower_t = np.asarray(lower_temperature_k, dtype=np.float64)
    lower_div = np.asarray(lower_divergence_s, dtype=np.float64)
    mid_div = np.asarray(midlevel_divergence_s, dtype=np.float64)
    upper_div = np.asarray(upperlevel_divergence_s, dtype=np.float64)
    if not (
        total.shape == lower_t.shape == lower_div.shape == mid_div.shape == upper_div.shape
    ):
        raise ValueError("all column inputs must share one shape")

    lower_q, mid_q, upper_q = _partition_total_humidity(
        total,
        midlevel_humidity,
        upperlevel_humidity,
        midlevel_fraction=midlevel_fraction,
        upperlevel_fraction=upperlevel_fraction,
    )
    before = lower_q + mid_q + upper_q

    if enforce_column_mass_closure:
        weights = np.asarray(
            (lower_mass_fraction, midlevel_mass_fraction, upperlevel_mass_fraction), dtype=np.float64
        )
        if np.any(weights <= 0.0):
            raise ValueError("pressure-layer mass fractions must be positive")
        upper_div = -(weights[0] * lower_div + weights[1] * mid_div) / weights[2]

    omega_lower_mid = 0.5 * lower_mid_pressure_depth_pa * (lower_div - mid_div)
    omega_mid_upper = 0.5 * mid_upper_pressure_depth_pa * (mid_div - upper_div)

    lower_mid_fraction = (
        np.clip(
            np.abs(omega_lower_mid) * dt_days * 86400.0 / lower_mid_pressure_depth_pa,
            0.0, 1.0,
        )
        if use_flux_form_exchange
        else _exchange_fraction(
            omega_lower_mid, dt_days, vertical_velocity_scale_pa_s, exchange_days
        )
    )
    lower_to_mid = np.where(omega_lower_mid < 0.0, lower_q * lower_mid_fraction, 0.0)
    mid_to_lower = np.where(omega_lower_mid > 0.0, mid_q * lower_mid_fraction, 0.0)
    lower_q = lower_q - lower_to_mid + mid_to_lower
    mid_q = mid_q + lower_to_mid - mid_to_lower

    mid_upper_fraction = (
        np.clip(
            np.abs(omega_mid_upper) * dt_days * 86400.0 / mid_upper_pressure_depth_pa,
            0.0, 1.0,
        )
        if use_flux_form_exchange
        else _exchange_fraction(
            omega_mid_upper, dt_days, vertical_velocity_scale_pa_s, exchange_days
        )
    )
    mid_to_upper = np.where(omega_mid_upper < 0.0, mid_q * mid_upper_fraction, 0.0)
    upper_to_mid = np.where(omega_mid_upper > 0.0, upper_q * mid_upper_fraction, 0.0)
    mid_q = mid_q - mid_to_upper + upper_to_mid
    upper_q = upper_q + mid_to_upper - upper_to_mid

    mid_ref = lower_t - lapse_rate_k_m * midlevel_height_m
    upper_ref = lower_t - lapse_rate_k_m * upperlevel_height_m
    relax = 1.0 - np.exp(-dt_days / thermal_relaxation_days)
    mid_t_in = mid_ref if midlevel_temperature_k is None else np.asarray(midlevel_temperature_k, dtype=np.float64)
    upper_t_in = upper_ref if upperlevel_temperature_k is None else np.asarray(upperlevel_temperature_k, dtype=np.float64)
    if mid_t_in.shape != total.shape or upper_t_in.shape != total.shape:
        raise ValueError("layer temperatures must match humidity shape")
    mid_t = np.clip(mid_t_in + relax * (mid_ref - mid_t_in), 150.0, 350.0)
    upper_t = np.clip(upper_t_in + relax * (upper_ref - upper_t_in), 150.0, 350.0)

    after = lower_q + mid_q + upper_q
    scale = max(float(np.sum(np.abs(before), dtype=np.float64)), 1.0)
    residual = float(np.sum(after - before, dtype=np.float64)) / scale
    return PressureColumnStep(
        lower_q.astype(np.float32),
        mid_q.astype(np.float32),
        upper_q.astype(np.float32),
        mid_t.astype(np.float32),
        upper_t.astype(np.float32),
        omega_lower_mid.astype(np.float32),
        omega_mid_upper.astype(np.float32),
        residual,
    )
