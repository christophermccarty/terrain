"""Low-order force-restore and Penman--Monteith land surface closure."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class LandSurfaceStep(NamedTuple):
    temperature: np.ndarray
    deep_temperature: np.ndarray
    latent_heat_w_m2: np.ndarray
    sensible_heat_w_m2: np.ndarray


def force_restore_penman_monteith(
    surface_temperature_k: np.ndarray,
    air_temperature_k: np.ndarray,
    deep_temperature_k: np.ndarray | None,
    soil_moisture: np.ndarray,
    wind_speed_m_s: np.ndarray,
    net_radiation_w_m2: np.ndarray,
    land_mask: np.ndarray,
    *,
    dt_days: float,
    surface_heat_capacity_j_m2_k: float,
    deep_heat_capacity_j_m2_k: float,
    restore_days: float,
    surface_resistance_min_s_m: float,
    surface_resistance_dry_s_m: float,
) -> LandSurfaceStep:
    """Advance a two-reservoir land surface one explicit, bounded timestep.

    The available energy is partitioned by the FAO-56 Penman--Monteith form.
    Soil moisture sets surface resistance, so wet land preferentially cools
    through latent heat while dry land partitions more energy into sensible
    heat.  The surface-to-deep exchange is the restore term.
    """
    if dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    if min(surface_heat_capacity_j_m2_k, deep_heat_capacity_j_m2_k,
           restore_days, surface_resistance_min_s_m, surface_resistance_dry_s_m) <= 0.0:
        raise ValueError("force-restore parameters must be positive")
    ts = np.asarray(surface_temperature_k, dtype=np.float64)
    ta = np.asarray(air_temperature_k, dtype=np.float64)
    soil = np.clip(np.asarray(soil_moisture, dtype=np.float64), 0.0, 1.0)
    wind = np.maximum(np.asarray(wind_speed_m_s, dtype=np.float64), 0.2)
    rn = np.asarray(net_radiation_w_m2, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    if not (ts.shape == ta.shape == soil.shape == wind.shape == rn.shape == land.shape):
        raise ValueError("land surface inputs must share a shape")
    # A newly enabled prognostic reservoir has no history.  Initializing it to
    # the surface avoids manufacturing a restore flux merely because the old
    # diagnostic atmosphere and the new land path disagree on their first
    # timestep; it then acquires physical memory through the restore equation.
    td = ts.copy() if deep_temperature_k is None else np.asarray(deep_temperature_k, dtype=np.float64).copy()
    if td.shape != ts.shape:
        raise ValueError("deep_temperature_k must match surface temperature")

    # Saturation-vapour slope (Pa K-1) and a standard psychrometric constant.
    tc = np.clip(ts - 273.15, -70.0, 70.0)
    es_pa = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    delta = 4098.0 * es_pa / (tc + 243.5) ** 2
    gamma = 66.0
    rho_air, cp_air = 1.2, 1004.0
    ra = 1.0 / (1.3e-3 * wind)
    rs = surface_resistance_min_s_m + (1.0 - soil) ** 2 * surface_resistance_dry_s_m
    # A humidity proxy is intentionally not supplied here; the resolved
    # air--surface temperature difference provides a stable vapour-deficit
    # closure while the main moisture model remains authoritative for water.
    vpd_pa = np.maximum(0.0, 120.0 * (ts - ta))
    latent = (delta * rn + rho_air * cp_air * vpd_pa / ra) / (
        delta + gamma * (1.0 + rs / ra)
    )
    latent = np.clip(latent, 0.0, np.maximum(rn + 350.0, 0.0))
    sensible = rho_air * cp_air * (ts - ta) / ra
    restore_flux = surface_heat_capacity_j_m2_k * (ts - td) / (restore_days * 86400.0)
    tendency = (rn - latent - sensible - restore_flux) * dt_days * 86400.0 / surface_heat_capacity_j_m2_k
    ts_next = ts + np.clip(tendency, -5.0, 5.0)
    td_next = td + restore_flux * dt_days * 86400.0 / deep_heat_capacity_j_m2_k
    ts_next = np.where(land, ts_next, ts)
    td_next = np.where(land, td_next, td)
    return LandSurfaceStep(
        ts_next.astype(np.float32), td_next.astype(np.float32),
        latent.astype(np.float32), sensible.astype(np.float32),
    )
