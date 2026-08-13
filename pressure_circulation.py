"""Horizontally resolved divergent-wind closure for a pressure column.

The prognostic wind solver carries rotational and divergent components together.
For an optional three-level pressure-column experiment, this module diagnoses the
layer-weighted horizontal mass-flux residual and applies only the minimum
diagnostic divergent correction needed to the upper level.  The correction is
constructed in longitude Fourier space plus a zonally symmetric meridional
flux, so its cost scales linearly with horizontal cell count rather than with a
slow global relaxation solve.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import numpy as np


_STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8


class HorizontalMassClosure(NamedTuple):
    upper_u_correction: np.ndarray
    upper_v_correction: np.ndarray
    residual_before_s: float
    residual_after_s: float
    equatorial_throughflow_before_m_s: float
    equatorial_throughflow_after_m_s: float


class DiabaticInterfaceMassFlux(NamedTuple):
    """Zonal-mean, mass-consistent pressure-interface circulation diagnosis."""

    omega_lower_mid_pa_s: np.ndarray
    omega_mid_upper_pa_s: np.ndarray
    lower_divergence_s: np.ndarray
    midlevel_divergence_s: np.ndarray
    upperlevel_divergence_s: np.ndarray
    latent_heating_w_m2: np.ndarray
    lower_mid_vertical_courant_max: float
    mid_upper_vertical_courant_max: float


class PressureCoordinateCirculation(NamedTuple):
    """One mass-consistent three-level horizontal/vertical circulation state."""

    lower_u: np.ndarray
    lower_v: np.ndarray
    midlevel_u: np.ndarray
    midlevel_v: np.ndarray
    upperlevel_u: np.ndarray
    upperlevel_v: np.ndarray
    interface_mass_flux: DiabaticInterfaceMassFlux


class LargeScaleHeatingStep(NamedTuple):
    """Radiatively adjusted latent-heating anomaly for diagnosed overturning."""

    heating_w_m2: np.ndarray
    radiative_adjustment_time_s: np.ndarray


def diabatic_interface_mass_flux(
    precipitation_mm_day: np.ndarray | None,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    *,
    dt_seconds: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    latent_heat_j_kg: float = 2.5e6,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> DiabaticInterfaceMassFlux:
    """Diagnose a vertically closed large-scale omega from diabatic heating.

    The existing raw independent layer winds contain divergent modes that imply
    pressure-interface mass transfer of tens of complete layers per day.  This
    operator deliberately does not cap those omegas.  It bypasses them and
    derives the large-scale overturning from the preceding raw-column
    precipitation: ``Lv * P`` is deposited uniformly per unit free-troposphere
    mass, then balanced against each interface's resolved static stability.

    Precipitation and temperatures are zonally averaged before the diagnosis.
    That is intentional: a three-level climate column represents the Hadley/
    seasonal mean overturning, whereas individual grid-cell convective plumes
    belong to the condensate closure.  The returned layer divergences solve the
    same centred continuity relations as the production omega calculation and
    have zero 0.40/0.35/0.25 mass-weighted column divergence at every cell.
    """
    if dt_seconds <= 0.0 or surface_pressure_pa <= 0.0:
        raise ValueError("dt_seconds and surface_pressure_pa must be positive")
    if lower_mid_pressure_depth_pa <= 0.0 or mid_upper_pressure_depth_pa <= 0.0:
        raise ValueError("pressure depths must be positive")
    if gravity_m_s2 <= 0.0 or cp_dry_j_kg_k <= 0.0 or latent_heat_j_kg <= 0.0:
        raise ValueError("physical constants must be positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(np.sum(fractions), 1.0):
        raise ValueError("layer mass fractions must be three positive values summing to one")
    lower = np.asarray(lower_temperature_k, dtype=np.float64)
    middle = np.asarray(midlevel_temperature_k, dtype=np.float64)
    upper = np.asarray(upperlevel_temperature_k, dtype=np.float64)
    if lower.ndim != 2 or lower.shape[1] != 2 * lower.shape[0] or middle.shape != lower.shape or upper.shape != lower.shape:
        raise ValueError("temperature layers must share a two-dimensional 2:1 grid")
    if precipitation_mm_day is None:
        precipitation = np.zeros_like(lower)
    else:
        precipitation = np.asarray(precipitation_mm_day, dtype=np.float64)
        if precipitation.shape != lower.shape:
            raise ValueError("precipitation must match the temperature grid")
        if not np.all(np.isfinite(precipitation)):
            raise ValueError("precipitation must be finite")

    # kg m-2 day-1 is numerically equal to mm day-1.  The planetwide mean
    # latent release is balanced by the host radiation/surface step; only its
    # cosine-area-weighted anomaly can drive a closed large-scale overturning.
    # Without that removal every latitude ascends, which cannot be represented
    # by a horizontal circulation and leads to arbitrary polar return flow.
    zonal_precipitation = np.maximum(np.mean(precipitation, axis=1, keepdims=True), 0.0)
    cos_lat = np.cos(np.radians(90.0 - (np.arange(lower.shape[0]) + 0.5) * 180.0 / lower.shape[0]))[:, None]
    mean_precipitation = np.sum(zonal_precipitation * cos_lat) / np.sum(cos_lat)
    latent_flux = float(latent_heat_j_kg) * (
        zonal_precipitation - mean_precipitation
    ) / 86400.0
    return diabatic_interface_mass_flux_from_heating(
        np.broadcast_to(latent_flux, lower.shape), lower, middle, upper,
        dt_seconds=dt_seconds, surface_pressure_pa=surface_pressure_pa,
        lower_mid_pressure_depth_pa=lower_mid_pressure_depth_pa,
        mid_upper_pressure_depth_pa=mid_upper_pressure_depth_pa,
        gravity_m_s2=gravity_m_s2, cp_dry_j_kg_k=cp_dry_j_kg_k,
        layer_mass_fractions=layer_mass_fractions,
    )


def diabatic_interface_mass_flux_from_heating(
    large_scale_heating_w_m2: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    *,
    dt_seconds: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> DiabaticInterfaceMassFlux:
    """Diagnose closed interface mass flux from a signed heating anomaly."""
    lower = np.asarray(lower_temperature_k, dtype=np.float64)
    middle = np.asarray(midlevel_temperature_k, dtype=np.float64)
    upper = np.asarray(upperlevel_temperature_k, dtype=np.float64)
    heating = np.asarray(large_scale_heating_w_m2, dtype=np.float64)
    if lower.ndim != 2 or lower.shape[1] != 2 * lower.shape[0] or middle.shape != lower.shape or upper.shape != lower.shape:
        raise ValueError("temperature layers must share a two-dimensional 2:1 grid")
    if heating.shape != lower.shape or not np.all(np.isfinite(heating)):
        raise ValueError("large-scale heating must be finite and match the temperature grid")
    if dt_seconds <= 0.0 or surface_pressure_pa <= 0.0 or lower_mid_pressure_depth_pa <= 0.0 or mid_upper_pressure_depth_pa <= 0.0:
        raise ValueError("time, pressure, and pressure depths must be positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(np.sum(fractions), 1.0):
        raise ValueError("layer mass fractions must be three positive values summing to one")
    # The forcing is a large-scale zonal anomaly.  Re-average here so callers
    # cannot accidentally turn local plume noise into a global circulation.
    latent_flux = np.broadcast_to(np.mean(heating, axis=1, keepdims=True), lower.shape).copy()
    column_mass = float(surface_pressure_pa) / float(gravity_m_s2)
    free_troposphere_mass = column_mass * (fractions[1] + fractions[2])
    heating_k_s = latent_flux / (float(cp_dry_j_kg_k) * free_troposphere_mass)
    lower_zonal = np.mean(lower, axis=1, keepdims=True)
    middle_zonal = np.mean(middle, axis=1, keepdims=True)
    upper_zonal = np.mean(upper, axis=1, keepdims=True)
    # Static stability is the vertical *potential-temperature* gradient, not
    # a raw temperature difference. A normally stratified troposphere cools
    # with height, so raw T differences can approach zero even while theta
    # increases strongly upward. Using raw T produced a singular omega during
    # phase-heating events. Layer-centre pressures follow the documented
    # 0.40/0.35/0.25 pressure-mass partition; no stability floor or omega cap
    # is introduced.
    layer_edges = float(surface_pressure_pa) * np.array(
        (1.0, 1.0 - fractions[0], fractions[2], 0.0), dtype=np.float64
    )
    layer_pressures = 0.5 * (layer_edges[:-1] + layer_edges[1:])
    kappa = 287.05 / float(cp_dry_j_kg_k)
    reference_pressure = float(surface_pressure_pa)
    lower_theta = lower_zonal * (reference_pressure / layer_pressures[0]) ** kappa
    middle_theta = middle_zonal * (reference_pressure / layer_pressures[1]) ** kappa
    upper_theta = upper_zonal * (reference_pressure / layer_pressures[2]) ** kappa
    lower_mid_stability = (middle_theta - lower_theta) / float(lower_mid_pressure_depth_pa)
    mid_upper_stability = (upper_theta - middle_theta) / float(mid_upper_pressure_depth_pa)
    # An unstably stratified interface is a microphysical convective-adjustment
    # problem, not a hydrostatic large-scale omega diagnostic.  Leave that
    # branch at zero instead of hiding an arbitrary stability floor here.
    omega_lower_mid = np.divide(
        -heating_k_s, lower_mid_stability,
        out=np.zeros_like(heating_k_s), where=lower_mid_stability > 0.0,
    )
    omega_mid_upper = np.divide(
        -heating_k_s, mid_upper_stability,
        out=np.zeros_like(heating_k_s), where=mid_upper_stability > 0.0,
    )
    omega_lower_mid = np.broadcast_to(omega_lower_mid, lower.shape).copy()
    omega_mid_upper = np.broadcast_to(omega_mid_upper, lower.shape).copy()

    # Invert the centred continuity relations used by the pressure column:
    # omega_lm=0.5*dp_lm*(d_lower-d_mid), omega_mu=0.5*dp_mu*(d_mid-d_upper),
    # plus weighted column divergence exactly equal to zero.
    lower_mid_difference = 2.0 * omega_lower_mid / float(lower_mid_pressure_depth_pa)
    mid_upper_difference = 2.0 * omega_mid_upper / float(mid_upper_pressure_depth_pa)
    mid_divergence = -fractions[0] * lower_mid_difference + fractions[2] * mid_upper_difference
    lower_divergence = mid_divergence + lower_mid_difference
    upper_divergence = mid_divergence - mid_upper_difference
    return DiabaticInterfaceMassFlux(
        omega_lower_mid.astype(np.float32),
        omega_mid_upper.astype(np.float32),
        lower_divergence.astype(np.float32),
        mid_divergence.astype(np.float32),
        upper_divergence.astype(np.float32),
        np.broadcast_to(latent_flux, lower.shape).astype(np.float32),
        float(np.max(np.abs(omega_lower_mid)) * dt_seconds / float(lower_mid_pressure_depth_pa)),
        float(np.max(np.abs(omega_mid_upper)) * dt_seconds / float(mid_upper_pressure_depth_pa)),
    )


def evolve_large_scale_heating_reservoir(
    previous_heating_w_m2: np.ndarray | None,
    condensation_mm_day: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    *,
    dt_seconds: float,
    surface_pressure_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    latent_heat_j_kg: float = 2.5e6,
) -> LargeScaleHeatingStep:
    """Low-pass large-scale latent heating using a derived radiative timescale.

    The reservoir stores only cosine-area-balanced zonal anomalies.  Its
    e-folding time is free-tropospheric heat capacity divided by linearized
    thermal emission (``4 sigma T^3``), so it introduces no configurable
    damping or circulation-strength coefficient.
    """
    lower, middle, upper, condensation = (
        np.asarray(value, dtype=np.float64)
        for value in (lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k, condensation_mm_day)
    )
    if lower.ndim != 2 or lower.shape[1] != 2 * lower.shape[0] or any(value.shape != lower.shape for value in (middle, upper, condensation)):
        raise ValueError("heating-reservoir fields must share a two-dimensional 2:1 grid")
    if dt_seconds <= 0.0 or surface_pressure_pa <= 0.0 or gravity_m_s2 <= 0.0 or cp_dry_j_kg_k <= 0.0:
        raise ValueError("heating-reservoir constants must be positive")
    previous = np.zeros_like(lower) if previous_heating_w_m2 is None else np.asarray(previous_heating_w_m2, dtype=np.float64)
    if previous.shape != lower.shape or not np.all(np.isfinite(previous)):
        raise ValueError("previous heating must be finite and match the temperature grid")
    zonal_condensation = np.maximum(np.mean(condensation, axis=1, keepdims=True), 0.0)
    cos_lat = np.cos(np.radians(90.0 - (np.arange(lower.shape[0]) + 0.5) * 180.0 / lower.shape[0]))[:, None]
    mean_condensation = np.sum(zonal_condensation * cos_lat) / np.sum(cos_lat)
    forcing = float(latent_heat_j_kg) * (zonal_condensation - mean_condensation) / 86400.0
    free_temperature = 0.35 * np.mean(middle, axis=1, keepdims=True) + 0.25 * np.mean(upper, axis=1, keepdims=True)
    free_mass = float(surface_pressure_pa) / float(gravity_m_s2) * 0.60
    radiative_stiffness = 4.0 * _STEFAN_BOLTZMANN_W_M2_K4 * np.maximum(free_temperature, 180.0) ** 3
    adjustment_time = free_mass * float(cp_dry_j_kg_k) / radiative_stiffness
    decay = np.exp(-float(dt_seconds) / adjustment_time)
    updated = decay * np.mean(previous, axis=1, keepdims=True) + (1.0 - decay) * forcing
    return LargeScaleHeatingStep(
        np.broadcast_to(updated, lower.shape).astype(np.float32),
        np.broadcast_to(adjustment_time, lower.shape).astype(np.float32),
    )


def shared_pressure_coordinate_circulation(
    precipitation_mm_day: np.ndarray | None,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_zonal_wind_m_s: np.ndarray,
    midlevel_zonal_wind_m_s: np.ndarray,
    upperlevel_zonal_wind_m_s: np.ndarray,
    *,
    dt_seconds: float,
    radius_m: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    large_scale_heating_w_m2: np.ndarray | None = None,
) -> PressureCoordinateCirculation:
    """Return winds and omega from one pressure-coordinate continuity solve.

    The zonal component keeps only each raw wind's longitude mean, which is a
    non-divergent zonal flow on this grid.  Meridional winds are then the
    minimum-norm fields whose production spherical divergence matches the
    diabatic interface solve exactly.  Thus the horizontal energy carrier and
    the vertical water/energy exchange are no longer two incompatible systems.
    """
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    fields = tuple(np.asarray(value, dtype=np.float64) for value in (
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        lower_zonal_wind_m_s, midlevel_zonal_wind_m_s, upperlevel_zonal_wind_m_s,
    ))
    shape = fields[0].shape
    if any(value.shape != shape for value in fields) or len(shape) != 2 or shape[1] != 2 * shape[0]:
        raise ValueError("all pressure-coordinate fields must share a two-dimensional 2:1 grid")
    if large_scale_heating_w_m2 is None:
        interface = diabatic_interface_mass_flux(
            precipitation_mm_day, fields[0], fields[1], fields[2],
            dt_seconds=dt_seconds, surface_pressure_pa=surface_pressure_pa,
            lower_mid_pressure_depth_pa=lower_mid_pressure_depth_pa,
            mid_upper_pressure_depth_pa=mid_upper_pressure_depth_pa,
            gravity_m_s2=gravity_m_s2, cp_dry_j_kg_k=cp_dry_j_kg_k,
        )
    else:
        interface = diabatic_interface_mass_flux_from_heating(
            large_scale_heating_w_m2, fields[0], fields[1], fields[2],
            dt_seconds=dt_seconds, surface_pressure_pa=surface_pressure_pa,
            lower_mid_pressure_depth_pa=lower_mid_pressure_depth_pa,
            mid_upper_pressure_depth_pa=mid_upper_pressure_depth_pa,
            gravity_m_s2=gravity_m_s2, cp_dry_j_kg_k=cp_dry_j_kg_k,
        )
    latitude = np.radians(90.0 - (np.arange(shape[0]) + 0.5) * 180.0 / shape[0])
    cos_lat = np.cos(latitude)

    def layer(zonal_wind: np.ndarray, divergence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.broadcast_to(np.mean(zonal_wind, axis=1, keepdims=True), shape).copy()
        v_1d = _zonal_mean_meridional_wind(divergence[:, 0], radius_m, cos_lat)
        v = np.broadcast_to(v_1d[:, None], shape).copy()
        return u.astype(np.float32), v.astype(np.float32)

    lower_u, lower_v = layer(fields[3], interface.lower_divergence_s)
    mid_u, mid_v = layer(fields[4], interface.midlevel_divergence_s)
    upper_u, upper_v = layer(fields[5], interface.upperlevel_divergence_s)
    return PressureCoordinateCirculation(
        lower_u, lower_v, mid_u, mid_v, upper_u, upper_v, interface,
    )


def spherical_divergence(u: np.ndarray, v: np.ndarray, radius_m: float) -> np.ndarray:
    """Production-compatible signed divergence on a north-to-south 2:1 grid.

    This is deliberately the unit-mass form of
    :func:`atmosphere.flux_divergence_spherical`: the pressure-column exchange
    must close the same discrete operator that diagnoses its interface omega.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    h, w = u.shape
    lat = np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h)
    cos = np.maximum(np.cos(lat), 1.0e-3)[:, None]
    dphi, dlon = np.pi / h, 2.0 * np.pi / w
    du = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dlon)
    vcos = v * cos
    dvcos = np.empty_like(vcos)
    dvcos[1:-1] = 0.5 * (vcos[2:] - vcos[:-2])
    dvcos[0] = vcos[1] - vcos[0]
    dvcos[-1] = vcos[-1] - vcos[-2]
    # Grid rows run north to south, while latitude increases northward.
    dvcos /= -dphi
    return (du + dvcos) / (float(radius_m) * cos)


def smooth_spherical_scalar(
    field: np.ndarray, *, strength: float, passes: int = 1,
) -> np.ndarray:
    """Apply a conservative, scale-selective filter on a regular spherical grid.

    The filter is for diagnosed divergence, where unresolved one- and two-cell
    noise otherwise enters pressure-interface omega directly.  Each pass
    preserves the cosine-area-weighted global mean exactly, so it cannot create
    a net column mass source or sink.
    """
    if passes < 0 or not 0.0 <= strength <= 1.0:
        raise ValueError("passes must be non-negative and strength must lie in [0, 1]")
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 2 * values.shape[0]:
        raise ValueError("field must use a two-dimensional 2:1 global grid")
    h = values.shape[0]
    weights = np.cos(np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h))[:, None]
    for _ in range(passes):
        north = np.vstack((values[:1], values[:-1]))
        south = np.vstack((values[1:], values[-1:]))
        neighbour_mean = 0.25 * (np.roll(values, 1, 1) + np.roll(values, -1, 1) + north + south)
        filtered = (1.0 - strength) * values + strength * neighbour_mean
        # Correct roundoff and boundary-stencil asymmetry in the only global
        # moment pressure continuity requires us to retain.
        filtered += (
            np.sum(values * weights) - np.sum(filtered * weights)
        ) / np.sum(np.broadcast_to(weights, values.shape))
        values = filtered
    return values


def balanced_thermal_wind_u(
    lower_u: np.ndarray,
    temperature_k: np.ndarray,
    *,
    radius_m: float,
    sidereal_day_hours: float,
    surface_pressure_pa: float,
    upper_pressure_pa: float,
    hadley_edge_deg: float,
    gas_constant_dry: float = 287.05,
) -> np.ndarray:
    """Return upper-level zonal wind implied by zonal-mean thermal-wind shear.

    The hydrostatic thermal-wind relation gives
    ``u_upper - u_lower = -(R/f) log(p_s/p_u) dT/dy``.  It is diagnosed from
    the zonal mean, which prevents terrain-scale temperature noise from being
    converted into an unphysical free-tropospheric jet.  A smooth equatorial
    taper suppresses the singular geostrophic approximation inside the Hadley
    cell rather than relying on an arbitrary finite ``f`` there.
    """
    lower_u = np.asarray(lower_u, dtype=np.float64)
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    if lower_u.shape != temperature_k.shape or lower_u.ndim != 2:
        raise ValueError("lower wind and temperature must share one two-dimensional shape")
    if radius_m <= 0.0 or sidereal_day_hours <= 0.0:
        raise ValueError("radius and sidereal day must be positive")
    if not 0.0 < upper_pressure_pa < surface_pressure_pa:
        raise ValueError("upper pressure must lie between zero and surface pressure")
    if hadley_edge_deg <= 0.0:
        raise ValueError("Hadley edge must be positive")
    h, _ = lower_u.shape
    latitude = np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h)
    dy_m = float(radius_m) * np.pi / h
    zonal_temperature = np.mean(temperature_k, axis=1)
    dtemp_dy = -np.gradient(zonal_temperature, dy_m, edge_order=1)
    omega = 2.0 * np.pi / (float(sidereal_day_hours) * 3600.0)
    coriolis = 2.0 * omega * np.sin(latitude)
    edge_rad = np.radians(float(hadley_edge_deg))
    f_edge = max(2.0 * omega * np.sin(edge_rad), 1.0e-8)
    # The taper goes rapidly to zero inside the direct circulation but does
    # not impose a discontinuity at its edge.
    taper = 1.0 - np.exp(-((np.abs(latitude) / edge_rad) ** 4))
    f_safe = np.copysign(np.maximum(np.abs(coriolis), f_edge), coriolis)
    shear = (
        -float(gas_constant_dry)
        * np.log(float(surface_pressure_pa) / float(upper_pressure_pa))
        * dtemp_dy
        / f_safe
        * taper
    )
    return (lower_u + shear[:, None]).astype(np.float32)


@lru_cache(maxsize=8)
def _meridional_gradient_pseudoinverse(height: int) -> np.ndarray:
    """Cache the fixed north-to-south derivative stencil's pseudoinverse."""
    if height < 2:
        raise ValueError("at least two latitude rows are required")
    dphi = np.pi / height
    h = height
    gradient = np.zeros((h, h), dtype=np.float64)
    gradient[0, 0], gradient[0, 1] = -1.0 / dphi, 1.0 / dphi
    gradient[-1, -2], gradient[-1, -1] = -1.0 / dphi, 1.0 / dphi
    row = np.arange(1, h - 1)
    gradient[row, row - 1] = -0.5 / dphi
    gradient[row, row + 1] = 0.5 / dphi
    # The constant null mode is immaterial.  The pseudoinverse selects its
    # minimum-norm representative while retaining the exact production stencil.
    return np.linalg.pinv(gradient)


def _zonal_mean_meridional_wind(
    target_divergence_s: np.ndarray, radius_m: float, cos_lat: np.ndarray,
) -> np.ndarray:
    """Return v whose discrete meridional divergence matches a zonal mean target."""
    vcos = _meridional_gradient_pseudoinverse(target_divergence_s.size) @ (
        -target_divergence_s * float(radius_m) * cos_lat
    )
    return vcos / cos_lat


def _equatorial_column_throughflow(
    lower_v: np.ndarray,
    mid_v: np.ndarray,
    upper_v: np.ndarray,
    cos_lat: np.ndarray,
) -> float:
    """Return the column-integrated zonal-mean ``v cos(latitude)`` at the equator.

    A zonally symmetric constant of ``v cos(latitude)`` belongs to the null
    space of the spherical-divergence stencil.  It is invisible to a pressure
    solve but represents an impossible pole-to-pole atmospheric throughflow.
    The equatorial pair is the least singular place to identify that gauge
    mode on PlanetSim's even-height grid.
    """
    column_vcos = np.mean(
        (0.40 * lower_v + 0.35 * mid_v + 0.25 * upper_v) * cos_lat[:, None],
        axis=1,
    )
    h = column_vcos.size
    return float(np.mean(column_vcos[(h // 2 - 1):(h // 2 + 1)]))


def close_upper_mass_flux(
    lower_u: np.ndarray, lower_v: np.ndarray, mid_u: np.ndarray, mid_v: np.ndarray,
    upper_u: np.ndarray, upper_v: np.ndarray, *, radius_m: float,
    strength: float = 1.0, max_speed_m_s: float = 12.0,
    throughflow_max_speed_m_s: float = 80.0,
) -> HorizontalMassClosure:
    """Return a bounded upper-level correction that closes weighted divergence.

    The lower, middle, and upper layers carry weights 0.40, 0.35, and 0.25.
    Zonal Fourier inversion exactly removes every resolved non-zonal divergence
    mode under the centred stencil.  A small least-squares solve handles the
    zonal-mean meridional flux; no arbitrary number of global relaxation
    iterations is required.
    """
    if max_speed_m_s <= 0.0 or throughflow_max_speed_m_s <= 0.0:
        raise ValueError("mass-flux correction speed limits must be positive")
    lower_u = np.asarray(lower_u, dtype=np.float64)
    h, w = lower_u.shape
    if any(np.asarray(field).shape != (h, w) for field in (lower_v, mid_u, mid_v, upper_u, upper_v)):
        raise ValueError("all layer winds must have the same two-dimensional shape")
    dl = spherical_divergence(lower_u, lower_v, radius_m)
    dm = spherical_divergence(mid_u, mid_v, radius_m)
    du = spherical_divergence(upper_u, upper_v, radius_m)
    residual = 0.40 * dl + 0.35 * dm + 0.25 * du
    target = -float(strength) * residual / 0.25

    lat = np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h)
    cos = np.cos(lat)
    dlon = 2.0 * np.pi / w
    zonal_mean = np.mean(target, axis=1)
    eddy = target - zonal_mean[:, None]
    mode = np.fft.fftfreq(w, d=1.0 / w)
    eigenvalue = 1j * np.sin(2.0 * np.pi * mode / w) / (dlon * float(radius_m) * cos[:, None])
    correction_u_hat = np.zeros((h, w), dtype=np.complex128)
    usable = np.abs(eigenvalue) > 1.0e-16
    correction_u_hat[usable] = np.fft.fft(eddy, axis=1)[usable] / eigenvalue[usable]
    correction_u = np.fft.ifft(correction_u_hat, axis=1).real
    correction_v = _zonal_mean_meridional_wind(zonal_mean, radius_m, cos)[:, None]
    correction_v = np.broadcast_to(correction_v, (h, w)).copy()

    speed = np.hypot(correction_u, correction_v)
    limiter = np.minimum(1.0, float(max_speed_m_s) / np.maximum(speed, 1e-12))
    correction_u *= limiter
    correction_v *= limiter
    # The divergence inversion cannot determine its zonally symmetric
    # v*cos(latitude) null mode. Remove the column-integrated component with
    # a separately bounded upper-wind correction; otherwise a spurious net
    # pole-to-pole transport can survive despite a locally closed divergence.
    throughflow_before = _equatorial_column_throughflow(
        np.asarray(lower_v, dtype=np.float64), np.asarray(mid_v, dtype=np.float64),
        np.asarray(upper_v, dtype=np.float64) + correction_v, cos,
    )
    throughflow_correction_v = -throughflow_before / (0.25 * np.maximum(cos, 1.0e-3))
    correction_v += np.clip(
        throughflow_correction_v,
        -float(throughflow_max_speed_m_s),
        float(throughflow_max_speed_m_s),
    )[:, None]
    throughflow_after = _equatorial_column_throughflow(
        np.asarray(lower_v, dtype=np.float64), np.asarray(mid_v, dtype=np.float64),
        np.asarray(upper_v, dtype=np.float64) + correction_v, cos,
    )
    after = 0.40 * dl + 0.35 * dm + 0.25 * spherical_divergence(
        np.asarray(upper_u, dtype=np.float64) + correction_u,
        np.asarray(upper_v, dtype=np.float64) + correction_v,
        radius_m,
    )
    return HorizontalMassClosure(
        correction_u.astype(np.float32),
        correction_v.astype(np.float32),
        float(np.sqrt(np.mean(residual**2))),
        float(np.sqrt(np.mean(after**2))),
        throughflow_before,
        throughflow_after,
    )
