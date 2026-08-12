"""Read-only observables for the normal surface/upper-wind atmosphere.

The default model carries surface temperature, near-surface air temperature,
precipitation, and a prognostic upper-wind layer.  It does not carry an
upper-layer temperature or humidity reservoir.  These diagnostics make that
boundary explicit before attempting to diagnose a thermally direct overturning
strength from incomplete state.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from balanced_dynamics import thermally_direct_overturning


def _tropical_mean(field: np.ndarray, latitude_deg: np.ndarray, edge_deg: float) -> float:
    tropical = np.abs(latitude_deg) <= float(edge_deg)
    weights = np.cos(np.radians(latitude_deg[tropical]))[:, None]
    values = np.asarray(field, dtype=np.float64)[tropical]
    return float(np.sum(values * weights) / np.sum(np.broadcast_to(weights, values.shape)))


def diagnose_two_layer_overturning_state(
    surface_temperature_k: np.ndarray,
    air_temperature_k: np.ndarray,
    precipitation_mm_day: np.ndarray,
    lower_meridional_wind_m_s: np.ndarray | None,
    upper_meridional_wind_m_s: np.ndarray | None,
    *,
    hadley_edge_deg: float,
    upper_temperature_k: np.ndarray | None = None,
    lower_mass_fraction: float = 0.40,
) -> dict[str, Any]:
    """Summarize whether the ordinary 1.5-layer state can diagnose overturning.

    This function intentionally does *not* infer a branch speed from surface
    temperature alone.  Latent heating can be measured from precipitation, but
    a moist-static-energy overturning needs an upper-layer temperature/stability
    state and a radiative tendency.  Returning availability explicitly prevents
    an arbitrary fixed-speed experiment from being mislabeled as diagnosed
    dynamics.
    """
    surface = np.asarray(surface_temperature_k, dtype=np.float64)
    air = np.asarray(air_temperature_k, dtype=np.float64)
    precip = np.asarray(precipitation_mm_day, dtype=np.float64)
    if surface.ndim != 2 or surface.shape[1] != 2 * surface.shape[0]:
        raise ValueError("two-layer diagnostics require a 2:1 global grid")
    if air.shape != surface.shape or precip.shape != surface.shape:
        raise ValueError("surface, air, and precipitation fields must share one shape")
    if not 0.0 < lower_mass_fraction < 1.0:
        raise ValueError("lower mass fraction must lie between zero and one")
    h, _ = surface.shape
    latitude = 90.0 - (np.arange(h, dtype=np.float64) + 0.5) * 180.0 / h
    upper_mass_fraction = 1.0 - float(lower_mass_fraction)
    thermal_equator = thermally_direct_overturning(
        air, hadley_edge_deg=float(hadley_edge_deg), lower_branch_speed_m_s=0.0
    ).thermal_equator_deg
    result: dict[str, Any] = {
        "thermal_equator_deg": float(thermal_equator),
        "tropical_surface_minus_air_k": _tropical_mean(surface - air, latitude, hadley_edge_deg),
        # 1 mm/day is 1 kg m-2 day-1; L_v P converts it to W m-2.
        "tropical_latent_heating_w_m2": _tropical_mean(
            np.maximum(precip, 0.0) * 2.5e6 / 86400.0, latitude, hadley_edge_deg
        ),
        "upper_temperature_available": False,
        "diagnosed_mse_strength_available": False,
        "diagnosed_mse_strength_reason": (
            "normal 1.5-layer state has upper wind but no upper temperature/humidity "
            "or radiative-tendency state"
        ),
    }
    if upper_temperature_k is not None:
        upper = np.asarray(upper_temperature_k, dtype=np.float64)
        if upper.shape != surface.shape:
            raise ValueError("upper temperature must match surface shape")
        result["upper_temperature_available"] = True
        result["tropical_air_minus_upper_k"] = _tropical_mean(air - upper, latitude, hadley_edge_deg)

    if lower_meridional_wind_m_s is None or upper_meridional_wind_m_s is None:
        result["two_layer_mass_flux_available"] = False
        return result
    lower_v = np.asarray(lower_meridional_wind_m_s, dtype=np.float64)
    upper_v = np.asarray(upper_meridional_wind_m_s, dtype=np.float64)
    if lower_v.shape != surface.shape or upper_v.shape != surface.shape:
        raise ValueError("two-layer winds must match surface shape")
    residual = float(lower_mass_fraction) * lower_v + upper_mass_fraction * upper_v
    result.update(
        {
            "two_layer_mass_flux_available": True,
            "tropical_lower_v_m_s": _tropical_mean(lower_v, latitude, hadley_edge_deg),
            "tropical_upper_v_m_s": _tropical_mean(upper_v, latitude, hadley_edge_deg),
            "two_layer_mass_flux_residual_mean_m_s": _tropical_mean(residual, latitude, hadley_edge_deg),
            "two_layer_mass_flux_residual_rms_m_s": float(np.sqrt(np.mean(residual**2))),
        }
    )
    return result
