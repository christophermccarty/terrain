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


class HorizontalMassClosure(NamedTuple):
    upper_u_correction: np.ndarray
    upper_v_correction: np.ndarray
    residual_before_s: float
    residual_after_s: float
    equatorial_throughflow_before_m_s: float
    equatorial_throughflow_after_m_s: float


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
