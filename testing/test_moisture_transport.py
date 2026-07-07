"""test_moisture_transport.py -- semi-Lagrangian scalar advection unit tests.

Covers `atmosphere._advect_scalar_semi_lagrangian` / `_semi_lagrangian_departure`,
added to fix the continental-interior/desert moisture-transport gap: the old
`generate_precipitation` humidity advection moved a fixed ~3 grid cells per
call regardless of wind speed or dt_days (see known-physics-gaps.md /
ROADMAP.md's "CFL-linked humidity advection" item). These are mechanism-level
tests independent of the full coupled simulation, following the same pattern
as test_upper_layer_wind.py's direct `evolve_wind_aloft` unit tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import atmosphere as atmo


def test_zero_wind_is_noop():
    """With no wind, the departure point is the grid point itself -- field unchanged."""
    H, W = 16, 32
    rng = np.random.default_rng(0)
    field = rng.random((H, W)).astype(np.float32)
    u = np.zeros((H, W), dtype=np.float32)
    v = np.zeros((H, W), dtype=np.float32)
    dx = np.full((H, W), 30000.0, dtype=np.float32)
    dy = 30000.0

    out = atmo._advect_scalar_semi_lagrangian(field, u, v, dt_seconds=86400.0, dx_meters=dx, dy_meters=dy)
    assert np.allclose(out, field, atol=1e-5)


def test_uniform_field_stays_uniform():
    """A spatially uniform field has no gradient to move -- any wind leaves it unchanged."""
    H, W = 16, 32
    field = np.full((H, W), 0.007, dtype=np.float32)
    rng = np.random.default_rng(1)
    u = rng.uniform(-20.0, 20.0, (H, W)).astype(np.float32)
    v = rng.uniform(-10.0, 10.0, (H, W)).astype(np.float32)
    dx = np.full((H, W), 30000.0, dtype=np.float32)
    dy = 30000.0

    out = atmo._advect_scalar_semi_lagrangian(field, u, v, dt_seconds=86400.0, dx_meters=dx, dy_meters=dy)
    assert np.allclose(out, 0.007, atol=1e-5)


def test_zonal_shift_matches_courant_formula():
    """A pure zonal (eastward) wind should shift a localized bump by the wind's own
    Courant number (wind_speed * dt / dx) of grid columns, wrapping periodically --
    this is the whole point of the fix: transport distance scales with wind*dt,
    not a fixed ~3-cell blend."""
    H, W = 8, 64
    dx_val = 40000.0  # meters/cell
    u_speed = 10.0  # m/s
    dt_seconds = 8 * 86400.0  # 8 days, matching simulate.py's _PRECIP_SUBSTEP_DAYS chunk
    expected_shift = round(u_speed * dt_seconds / dx_val)  # cells

    field = np.zeros((H, W), dtype=np.float32)
    j0 = 10
    field[:, j0] = 1.0
    u = np.full((H, W), u_speed, dtype=np.float32)
    v = np.zeros((H, W), dtype=np.float32)
    dx = np.full((H, W), dx_val, dtype=np.float32)
    dy = dx_val

    out = atmo._advect_scalar_semi_lagrangian(field, u, v, dt_seconds=dt_seconds, dx_meters=dx, dy_meters=dy)
    peak_col = int(np.argmax(out[0, :]))
    expected_col = (j0 + expected_shift) % W
    assert peak_col == expected_col, (
        f"bump at col {j0} with u={u_speed} m/s over {dt_seconds/86400:.0f} days "
        f"landed at col {peak_col}, expected {expected_col} (shift={expected_shift} cells)"
    )


def test_scalar_advection_matches_wind_advection_for_shared_field():
    """Advecting the `u` field itself as a passive scalar must match the u-component
    of `_advect_wind_semi_lagrangian` exactly -- both share `_semi_lagrangian_departure`,
    so this is a direct regression check that the extraction didn't diverge."""
    H, W = 12, 24
    rng = np.random.default_rng(2)
    u = rng.uniform(-15.0, 15.0, (H, W)).astype(np.float32)
    v = rng.uniform(-8.0, 8.0, (H, W)).astype(np.float32)
    dx = (30000.0 * (0.3 + np.cos(np.linspace(-1.4, 1.4, H))[:, None] * np.ones((1, W)))).astype(np.float32)
    dy = 28000.0

    u_new_wind, _ = atmo._advect_wind_semi_lagrangian(u, v, dt_seconds=172800.0, dx_meters=dx, dy_meters=dy)
    u_new_scalar = atmo._advect_scalar_semi_lagrangian(u, u, v, dt_seconds=172800.0, dx_meters=dx, dy_meters=dy)

    assert np.allclose(u_new_wind, u_new_scalar, atol=1e-4)


def test_no_new_extrema_introduced():
    """Bilinear sampling (order=1) is a convex combination of neighbors -- it must
    not overshoot beyond the input field's own min/max."""
    H, W = 20, 40
    rng = np.random.default_rng(3)
    field = rng.uniform(0.0, 0.02, (H, W)).astype(np.float32)
    u = rng.uniform(-25.0, 25.0, (H, W)).astype(np.float32)
    v = rng.uniform(-15.0, 15.0, (H, W)).astype(np.float32)
    dx = np.full((H, W), 35000.0, dtype=np.float32)
    dy = 35000.0

    out = atmo._advect_scalar_semi_lagrangian(field, u, v, dt_seconds=5 * 86400.0, dx_meters=dx, dy_meters=dy)
    assert out.min() >= field.min() - 1e-6
    assert out.max() <= field.max() + 1e-6
    assert np.all(np.isfinite(out))
