"""Conserved horizontal column-water budget primitives.

This is deliberately independent from precipitation generation while the
legacy row-target rescale remains active.  It supplies the mass-conserving
kernel and diagnostics needed to migrate without mixing both closures.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ColumnWaterStep(NamedTuple):
    water_mm: np.ndarray
    transport_tendency_mm_day: np.ndarray
    residual_mm: float
    relative_residual: float
    substeps: int
    maximum_outgoing_courant: float


def evolve_column_water(
    water_mm: np.ndarray,
    evaporation_mm_day: np.ndarray,
    precipitation_mm_day: np.ndarray,
    wind_u_m_s: np.ndarray,
    wind_v_m_s: np.ndarray,
    *,
    dx_m: np.ndarray | float,
    dy_m: float,
    dt_days: float,
    cell_area_m2: np.ndarray | float | None = None,
    x_face_length_m: np.ndarray | float | None = None,
    y_face_length_m: np.ndarray | float | None = None,
    max_courant: float = 0.5,
) -> ColumnWaterStep:
    """Advance ``dW/dt = E - P - div(W v)`` with conservative fluxes.

    ``water_mm`` is a depth, while conservation applies to ``water_mm * cell
    area``.  Callers on a regular Cartesian grid can omit the optional geometry
    arguments.  Spherical callers should provide exact cell areas plus the
    meridional length of east/west faces (``x_face_length_m``) and the zonal
    length of north/south faces (``y_face_length_m``, shape ``(H + 1, W)``).

    Longitude is periodic and latitude has closed boundaries.  Fluxes live on
    shared faces, so an amount leaving one cell enters its neighbour exactly.
    The integration is split until every combined outgoing Courant fraction is
    at most ``max_courant``; this preserves non-negative water for transport
    alone instead of hiding an unstable step behind a clipping loss.
    """
    if dt_days <= 0.0 or dy_m <= 0.0 or not 0.0 < max_courant <= 1.0:
        raise ValueError("dt_days/dy_m must be positive and max_courant in (0, 1]")
    w = np.clip(np.asarray(water_mm, dtype=np.float64), 0.0, None)
    evap = np.asarray(evaporation_mm_day, dtype=np.float64)
    precip = np.asarray(precipitation_mm_day, dtype=np.float64)
    u, v = np.asarray(wind_u_m_s, dtype=np.float64), np.asarray(wind_v_m_s, dtype=np.float64)
    if not (w.shape == evap.shape == precip.shape == u.shape == v.shape) or w.ndim != 2:
        raise ValueError("column-water fields must be same-shape 2-D arrays")
    dx = np.broadcast_to(np.asarray(dx_m, dtype=np.float64), w.shape)
    if np.any(dx <= 0.0):
        raise ValueError("dx_m must be positive")
    area = np.broadcast_to(
        np.asarray(cell_area_m2 if cell_area_m2 is not None else dx * dy_m, dtype=np.float64),
        w.shape,
    )
    if np.any(area <= 0.0):
        raise ValueError("cell_area_m2 must be positive")
    x_length = np.broadcast_to(
        np.asarray(x_face_length_m if x_face_length_m is not None else dy_m, dtype=np.float64),
        w.shape,
    )
    if y_face_length_m is None:
        y_length = np.zeros((w.shape[0] + 1, w.shape[1]), dtype=np.float64)
        y_length[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    else:
        y_length = np.broadcast_to(
            np.asarray(y_face_length_m, dtype=np.float64),
            (w.shape[0] + 1, w.shape[1]),
        )
    if np.any(x_length <= 0.0) or np.any(y_length < 0.0):
        raise ValueError("face lengths must be non-negative (and x faces positive)")

    # Face velocities are shared by neighbouring cells.  Positive v is
    # northward; rows are indexed north-to-south, hence the north face of row
    # i lies at y_face[i].
    u_east = 0.5 * (u + np.roll(u, -1, axis=1))
    v_north = np.zeros((w.shape[0] + 1, w.shape[1]), dtype=np.float64)
    v_north[1:-1] = 0.5 * (v[:-1] + v[1:])

    # The sum of outward Courant fractions bounds the explicit donor-cell
    # update.  Sources/sinks are intentionally not folded into this limiter;
    # any physical source/sink imbalance is returned in the residual rather
    # than silently reinterpreted as transport.
    u_west = np.roll(u_east, 1, axis=1)
    v_south = v_north[1:]
    v_north_cell = v_north[:-1]
    outbound_rate = (
        np.maximum(u_east, 0.0) * x_length
        + np.maximum(-u_west, 0.0) * x_length
        + np.maximum(v_north_cell, 0.0) * y_length[:-1]
        + np.maximum(-v_south, 0.0) * y_length[1:]
    ) / area
    n_substeps = max(1, int(np.ceil(float(np.max(outbound_rate)) * dt_days * 86400.0 / max_courant)))
    dt_sub_seconds = dt_days * 86400.0 / n_substeps
    source_rate = evap - precip
    mass = w * area
    for _ in range(n_substeps):
        depth = mass / area
        east_flux = np.where(u_east >= 0.0, u_east * depth, u_east * np.roll(depth, -1, axis=1)) * x_length
        west_flux = np.roll(east_flux, 1, axis=1)
        north_flux = np.zeros_like(v_north)
        north_flux[1:-1] = np.where(
            v_north[1:-1] >= 0.0,
            v_north[1:-1] * depth[1:],
            v_north[1:-1] * depth[:-1],
        ) * y_length[1:-1]
        south_flux = north_flux[1:]
        mass = mass + dt_sub_seconds * (
            west_flux - east_flux + south_flux - north_flux[:-1]
        ) + (dt_sub_seconds / 86400.0) * source_rate * area

    water_next = np.maximum(mass / area, 0.0)
    transport = (water_next - w - dt_days * source_rate) / dt_days
    expected_mass = np.sum((w + dt_days * source_rate) * area)
    final_mass = float(np.sum(water_next * area))
    residual = final_mass - float(expected_mass)
    relative_residual = residual / max(abs(final_mass), abs(float(expected_mass)), 1.0)
    return ColumnWaterStep(
        water_next.astype(np.float32), transport.astype(np.float32), residual, relative_residual,
        n_substeps, float(np.max(outbound_rate) * dt_days * 86400.0 / n_substeps),
    )
