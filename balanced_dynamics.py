"""Standalone pressure-level balanced circulation primitives.

The native model's existing wind solvers remain prognostic.  This module adds
an optional statistical-dynamical balance target derived from hydrostatic
geopotential, with an explicitly bounded ageostrophic cross-isobar component.
It deliberately contains no external-dycore dependency.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class BalancedWind(NamedTuple):
    u_geostrophic: np.ndarray
    v_geostrophic: np.ndarray
    u_ageostrophic: np.ndarray
    v_ageostrophic: np.ndarray

    @property
    def u(self) -> np.ndarray:
        return self.u_geostrophic + self.u_ageostrophic

    @property
    def v(self) -> np.ndarray:
        return self.v_geostrophic + self.v_ageostrophic


class ThermallyDirectOverturning(NamedTuple):
    """Mass-conserving three-level meridional overturning anomaly."""

    lower_v: np.ndarray
    middle_v: np.ndarray
    upper_v: np.ndarray
    thermal_equator_deg: float


def pressure_level_geopotential(
    elevation_m: np.ndarray,
    temperature_k: np.ndarray,
    *,
    gravity_m_s2: float,
    gas_constant_dry: float,
    surface_pressure_pa: float,
    level_pressure_pa: float,
    zonal_mean_temperature: bool = True,
) -> np.ndarray:
    """Hydrostatic geopotential at a pressure surface [m2 s-2]."""
    elevation = np.asarray(elevation_m, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    if elevation.shape != temperature.shape or elevation.ndim != 2:
        raise ValueError("elevation and temperature must share one two-dimensional shape")
    if gravity_m_s2 <= 0.0 or gas_constant_dry <= 0.0:
        raise ValueError("gravity and gas constant must be positive")
    if not 0.0 < level_pressure_pa < surface_pressure_pa:
        raise ValueError("level pressure must lie between zero and surface pressure")
    thermal = np.mean(temperature, axis=1, keepdims=True) if zonal_mean_temperature else temperature
    return (
        float(gravity_m_s2) * elevation
        + float(gas_constant_dry)
        * thermal
        * np.log(float(surface_pressure_pa) / float(level_pressure_pa))
    )


def balanced_pressure_wind(
    geopotential_m2_s2: np.ndarray,
    *,
    radius_m: float,
    sidereal_day_hours: float,
    hadley_edge_deg: float,
    ageostrophic_timescale_hours: float = 0.0,
    max_speed_m_s: float = 80.0,
) -> BalancedWind:
    """Diagnose metric-aware geostrophic and cross-isobar wind components.

    The equatorial taper explicitly removes the geostrophic approximation from
    the direct Hadley circulation rather than using a singular artificial
    Coriolis floor.  The ageostrophic term follows the geopotential gradient
    down-slope on a specified adjustment timescale and is bounded to half the
    geostrophic speed at each cell.
    """
    phi = np.asarray(geopotential_m2_s2, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] != 2 * phi.shape[0]:
        raise ValueError("geopotential must use a two-dimensional 2:1 global grid")
    if radius_m <= 0.0 or sidereal_day_hours <= 0.0 or hadley_edge_deg <= 0.0:
        raise ValueError("radius, sidereal day, and Hadley edge must be positive")
    if ageostrophic_timescale_hours < 0.0 or max_speed_m_s <= 0.0:
        raise ValueError("ageostrophic timescale must be non-negative and speed cap positive")
    h, w = phi.shape
    lat = np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h)
    cos = np.maximum(np.cos(lat), 1.0e-3)
    dy = float(radius_m) * np.pi / h
    dx = float(radius_m) * (2.0 * np.pi / w) * cos[:, None]
    dphi_dy = -np.gradient(phi, dy, axis=0, edge_order=1)
    dphi_dx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dx)
    omega = 2.0 * np.pi / (float(sidereal_day_hours) * 3600.0)
    f = 2.0 * omega * np.sin(lat)
    edge_rad = np.radians(float(hadley_edge_deg))
    edge_f = max(2.0 * omega * np.sin(edge_rad), 1e-8)
    f_safe = np.copysign(np.maximum(np.abs(f), edge_f), f)[:, None]
    hadley_taper = 1.0 - np.exp(-((np.abs(lat) / edge_rad) ** 4))
    u_geo = (-dphi_dy / f_safe) * hadley_taper[:, None]
    v_geo = (dphi_dx / f_safe) * hadley_taper[:, None]
    if ageostrophic_timescale_hours == 0.0:
        u_ageo = np.zeros_like(u_geo)
        v_ageo = np.zeros_like(v_geo)
    else:
        tau_s = float(ageostrophic_timescale_hours) * 3600.0
        u_ageo = -tau_s * dphi_dx * hadley_taper[:, None]
        v_ageo = -tau_s * dphi_dy * hadley_taper[:, None]
        ageo_cap = 0.5 * np.maximum(np.hypot(u_geo, v_geo), 1.0)
        ageo_speed = np.hypot(u_ageo, v_ageo)
        ageo_scale = np.minimum(1.0, ageo_cap / np.maximum(ageo_speed, 1e-12))
        u_ageo *= ageo_scale
        v_ageo *= ageo_scale
    speed = np.hypot(u_geo + u_ageo, v_geo + v_ageo)
    scale = np.minimum(1.0, float(max_speed_m_s) / np.maximum(speed, 1e-12))
    return BalancedWind(
        (u_geo * scale).astype(np.float32),
        (v_geo * scale).astype(np.float32),
        (u_ageo * scale).astype(np.float32),
        (v_ageo * scale).astype(np.float32),
    )


def thermally_direct_overturning(
    temperature_k: np.ndarray,
    *,
    hadley_edge_deg: float,
    lower_branch_speed_m_s: float,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> ThermallyDirectOverturning:
    """Diagnose a compact, thermally centred three-level Hadley anomaly.

    The lower branch converges on the thermally diagnosed ITCZ and the middle
    and upper branches return poleward.  Their mass-weighted sum is exactly
    zero at every latitude, so this is a vertical overturning circulation, not
    a spurious column mass source.  The amplitude is deliberately supplied by
    the caller; this primitive specifies structure and conservation rather
    than embedding an Earth-specific circulation strength.
    """
    temperature = np.asarray(temperature_k, dtype=np.float64)
    if temperature.ndim != 2 or temperature.shape[1] != 2 * temperature.shape[0]:
        raise ValueError("temperature must use a two-dimensional 2:1 global grid")
    if hadley_edge_deg <= 0.0 or lower_branch_speed_m_s < 0.0:
        raise ValueError("Hadley edge must be positive and branch speed non-negative")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(np.sum(fractions), 1.0):
        raise ValueError("layer mass fractions must be three positive values summing to one")
    h, w = temperature.shape
    latitude = 90.0 - (np.arange(h, dtype=np.float64) + 0.5) * 180.0 / h
    zonal_temperature = np.mean(temperature, axis=1)
    tropical = np.abs(latitude) <= 35.0
    excess = np.maximum(zonal_temperature - np.mean(zonal_temperature[tropical]), 0.0)
    if float(np.sum(excess[tropical])) <= 1.0e-12:
        thermal_equator = 0.0
    else:
        thermal_equator = float(
            np.sum(latitude[tropical] * excess[tropical]) / np.sum(excess[tropical])
        )
    thermal_equator = float(np.clip(thermal_equator, -15.0, 15.0))
    relative = latitude - thermal_equator
    x = np.abs(relative) / float(hadley_edge_deg)
    branch = np.where(x < 1.0, np.sin(np.pi * x), 0.0)
    lower = -np.sign(relative) * float(lower_branch_speed_m_s) * branch
    lower[np.abs(relative) < 1.0e-12] = 0.0
    return_speed = -fractions[0] * lower / (fractions[1] + fractions[2])
    return ThermallyDirectOverturning(
        np.broadcast_to(lower[:, None], (h, w)).astype(np.float32),
        np.broadcast_to(return_speed[:, None], (h, w)).astype(np.float32),
        np.broadcast_to(return_speed[:, None], (h, w)).astype(np.float32),
        thermal_equator,
    )


class MoistStaticEnergyOverturning(NamedTuple):
    """Diagnosed Hadley lower-branch speed from a two-term diabatic-heating budget."""

    speed_m_s: float
    latent_heating_k_s: float
    radiative_heating_k_s: float
    total_heating_k_s: float
    omega_pa_s: float


def moist_static_energy_overturning_speed(
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    *,
    radiative_equilibrium_temperature_k: np.ndarray,
    radius_m: float,
    hadley_edge_deg: float,
    layer_pressure_depth_pa: float,
    midlevel_height_m: float,
    latent_relaxation_days: float,
    radiative_relaxation_days: float,
    max_speed_m_s: float,
    precipitation_mm_day: np.ndarray | None = None,
    surface_pressure_pa: float = 101_325.0,
    gravity_m_s2: float = 9.81,
    cp_dry_j_kg_k: float = 1004.0,
    latent_heat_j_kg: float = 2.5e6,
) -> MoistStaticEnergyOverturning:
    """Infer a bounded Hadley lower-branch speed from a full diabatic-heating budget.

    Unlike ``diabatic_overturning_speed``, which infers heating solely from the
    persistent midlevel temperature anomaly, this sums two independently
    diagnosed heating terms before converting to pressure velocity:

    * Latent heating from actual resolved condensation, using the supplied
      precipitation field (``L * P / (cp * column mass)``).  When no
      precipitation field is supplied this falls back to the midlevel-anomaly
      memory ``diabatic_overturning_speed`` uses, so the closure stays usable
      before the column-water path is enabled.
    * A resolved radiative/thermal heating rate: the supplied equilibrium
      temperature (the model's own seasonal radiative-plus-transport target,
      not a fixed constant) relaxed toward at ``radiative_relaxation_days``.

    Both terms are zonal-mean, tropical-band averages, so this remains a
    scalar closure that supplies unresolved overturning strength; the
    mass-conserving three-level structure is still provided by
    ``thermally_direct_overturning``, whose lower/middle/upper branches cancel
    exactly regardless of how this speed was diagnosed.
    """
    lower = np.asarray(lower_temperature_k, dtype=np.float64)
    middle = np.asarray(midlevel_temperature_k, dtype=np.float64)
    radiative_eq = np.asarray(radiative_equilibrium_temperature_k, dtype=np.float64)
    if lower.shape != middle.shape or lower.ndim != 2 or lower.shape[1] != 2 * lower.shape[0]:
        raise ValueError("lower and middle temperatures must share a 2:1 grid")
    if radiative_eq.shape != lower.shape:
        raise ValueError("radiative equilibrium temperature must match lower temperature shape")
    scales = (
        radius_m, hadley_edge_deg, layer_pressure_depth_pa, midlevel_height_m,
        latent_relaxation_days, radiative_relaxation_days, max_speed_m_s,
        surface_pressure_pa, gravity_m_s2, cp_dry_j_kg_k, latent_heat_j_kg,
    )
    if min(scales) <= 0.0:
        raise ValueError("moist-static-energy overturning scales must be positive")

    h = lower.shape[0]
    latitude = 90.0 - (np.arange(h, dtype=np.float64) + 0.5) * 180.0 / h
    tropical = np.abs(latitude) <= float(hadley_edge_deg)

    if precipitation_mm_day is not None:
        precip = np.asarray(precipitation_mm_day, dtype=np.float64)
        if precip.shape != lower.shape:
            raise ValueError("precipitation must match lower temperature shape")
        column_mass_kg_m2 = float(surface_pressure_pa) / float(gravity_m_s2)
        condensation_kg_m2_s = np.maximum(precip, 0.0) / 86400.0
        latent_heating_field = (
            float(latent_heat_j_kg) * condensation_kg_m2_s
            / (float(cp_dry_j_kg_k) * column_mass_kg_m2)
        )
    else:
        reference_middle = lower - 6.5e-3 * float(midlevel_height_m)
        anomaly_k = np.maximum(middle - reference_middle, 0.0)
        latent_heating_field = anomaly_k / (float(latent_relaxation_days) * 86400.0)

    radiative_heating_field = (
        (radiative_eq - lower) / (float(radiative_relaxation_days) * 86400.0)
    )

    latent_heating_rate = float(np.mean(latent_heating_field[tropical]))
    radiative_heating_rate = float(np.mean(radiative_heating_field[tropical]))
    total_heating_rate = latent_heating_rate + radiative_heating_rate

    stability_k_pa = np.maximum(
        np.mean(np.abs(lower - middle)[tropical]) / float(layer_pressure_depth_pa),
        1.0e-4,
    )
    omega_pa_s = total_heating_rate / stability_k_pa
    speed = (
        omega_pa_s * float(radius_m) * np.radians(float(hadley_edge_deg))
        / float(layer_pressure_depth_pa)
    )
    bounded_speed = float(np.clip(speed, 0.0, float(max_speed_m_s)))
    return MoistStaticEnergyOverturning(
        bounded_speed,
        latent_heating_rate,
        radiative_heating_rate,
        total_heating_rate,
        float(omega_pa_s),
    )


def diabatic_overturning_speed(
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    *,
    radius_m: float,
    hadley_edge_deg: float,
    layer_pressure_depth_pa: float,
    midlevel_height_m: float,
    relaxation_days: float,
    max_speed_m_s: float,
) -> float:
    """Infer a bounded Hadley lower-branch speed from stored latent heating.

    The persistent midlevel temperature anomaly above a reference lapse-rate
    profile is a diabatic-heating memory.  Dividing it by its relaxation time
    gives Q [K s-1]; Q/static-stability gives pressure velocity, and continuity
    over the diagnosed cell width converts that to a meridional branch speed.
    This is intentionally a scalar, zonal-mean closure: it supplies unresolved
    overturning while regional winds remain prognostic.
    """
    lower = np.asarray(lower_temperature_k, dtype=np.float64)
    middle = np.asarray(midlevel_temperature_k, dtype=np.float64)
    if lower.shape != middle.shape or lower.ndim != 2 or lower.shape[1] != 2 * lower.shape[0]:
        raise ValueError("lower and middle temperatures must share a 2:1 grid")
    if min(radius_m, hadley_edge_deg, layer_pressure_depth_pa, midlevel_height_m, relaxation_days, max_speed_m_s) <= 0.0:
        raise ValueError("diabatic-overturning scales must be positive")
    reference_middle = lower - 6.5e-3 * float(midlevel_height_m)
    heating_k = np.maximum(middle - reference_middle, 0.0)
    h = lower.shape[0]
    latitude = 90.0 - (np.arange(h, dtype=np.float64) + 0.5) * 180.0 / h
    tropical = np.abs(latitude) <= float(hadley_edge_deg)
    heating_rate = float(np.mean(heating_k[tropical])) / (float(relaxation_days) * 86400.0)
    stability_k_pa = np.maximum(
        np.mean(np.abs(lower - middle)[tropical]) / float(layer_pressure_depth_pa),
        1.0e-4,
    )
    omega_pa_s = heating_rate / stability_k_pa
    speed = (
        omega_pa_s * float(radius_m) * np.radians(float(hadley_edge_deg))
        / float(layer_pressure_depth_pa)
    )
    return float(np.clip(speed, 0.0, float(max_speed_m_s)))
