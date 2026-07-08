"""test_moisture_flux_transport.py -- CFL-safe Eulerian scalar advection tests.

Covers `atmosphere._advect_scalar_flux_eulerian` (and its
`_advect_scalar_cfl_step_numba`/`_advect_scalar_cfl_step` kernels), added to fix the
jump-dilution bug diagnosed in the moisture-advection-jump-dilution-2026-07 project
memory: the old `_advect_scalar_semi_lagrangian` sampled a single backward-trajectory
point covering the *entire* substep's distance at once (~5000km at real MONTHLY-mode
substep dt), which measurably diluted even the ocean source. This scheme instead
integrates the advection equation `dq/dt = -u*dq/dx - v*dq/dy` forward in many small,
CFL-respecting substeps (deliberately advection, not flux-divergence, form -- q is a
mixing ratio with no companion air-density continuity equation in this model; see the
function's own docstring for why a conservative form blows up here). Mirrors
test_moisture_transport.py's mechanism-level testing pattern (independent of the full
coupled simulation).

The tests below are mechanism-level (synthetic fields) -- they check the properties
this function is built to guarantee (no-op under no/uniform conditions, bounded
output, correct net transport distance). The actual bug-fix claim was verified
separately against real terrain (`saves/earth.pkl`, the exact save the diagnosing
session used), reproducing its 50 deg N Atlantic->Central Europe transect at the real
~7.61-day production substep dt: old mechanism ocean-cell RH fell 92%->78.6%->59.3%
as `moisture_advection_scale` went 0->0.3->0.7 (matches the memory's ~66%-at-0.7
finding); new mechanism stayed 92%->93.6%->94.2% -- no measurable dilution. Not
committed as an automated test since it depends on a large real-terrain save file,
consistent with this project's existing convention for this kind of check (see
moisture-advection-jump-dilution-2026-07 project memory's own "reusable tooling"
section).
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
    """With no wind, there is no flux at any face -- field must be unchanged."""
    H, W = 12, 32
    rng = np.random.default_rng(0)
    field = rng.random((H, W)).astype(np.float32)
    u = np.zeros((H, W), dtype=np.float32)
    v = np.zeros((H, W), dtype=np.float32)
    dx = np.full((H, W), 30000.0, dtype=np.float32)
    dy = 30000.0

    out = atmo._advect_scalar_flux_eulerian(field, u, v, dt_seconds=86400.0, dx_meters=dx, dy_meters=dy)
    assert np.allclose(out, field, atol=1e-6)


def test_uniform_field_stays_uniform():
    """A spatially uniform field has zero flux divergence everywhere -- any wind
    (even highly variable) leaves it unchanged."""
    H, W = 12, 32
    field = np.full((H, W), 0.007, dtype=np.float32)
    rng = np.random.default_rng(1)
    u = rng.uniform(-20.0, 20.0, (H, W)).astype(np.float32)
    v = rng.uniform(-10.0, 10.0, (H, W)).astype(np.float32)
    dx = np.full((H, W), 30000.0, dtype=np.float32)
    dy = 30000.0

    out = atmo._advect_scalar_flux_eulerian(field, u, v, dt_seconds=6 * 86400.0, dx_meters=dx, dy_meters=dy)
    assert np.allclose(out, 0.007, atol=1e-4)


def test_bounded_no_blowup_under_realistic_divergence():
    """A genuine risk for a naive flux-divergence ('conservative') scheme: a
    smooth, perfectly ordinary wind divergence sustained over a multi-day
    integration would compound its `q*div(V)` reaction term exponentially
    (found directly while building this function -- see its docstring). Pure
    advection form has no such term: the output must stay within the input
    field's own range regardless of how much the wind field diverges."""
    H, W = 14, 40
    field = np.random.default_rng(3).uniform(0.0, 0.02, (H, W)).astype(np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False)
    u = np.tile((20.0 * np.sin(x)).astype(np.float32), (H, 1))
    v = np.tile((10.0 * np.cos(x)).astype(np.float32), (H, 1))
    dx = np.full((H, W), 35000.0, dtype=np.float32)
    dy = 35000.0

    out = atmo._advect_scalar_flux_eulerian(field, u, v, dt_seconds=5 * 86400.0, dx_meters=dx, dy_meters=dy)
    assert out.min() >= field.min() - 1e-6
    assert out.max() <= field.max() + 1e-6
    assert np.all(np.isfinite(out))


def test_no_new_extrema_introduced():
    """An upwind scheme respecting its CFL bound is monotone -- it must not
    overshoot beyond the input field's own min/max."""
    H, W = 14, 40
    rng = np.random.default_rng(3)
    field = rng.uniform(0.0, 0.02, (H, W)).astype(np.float32)
    u = rng.uniform(-25.0, 25.0, (H, W)).astype(np.float32)
    v = rng.uniform(-15.0, 15.0, (H, W)).astype(np.float32)
    dx = np.full((H, W), 35000.0, dtype=np.float32)
    dy = 35000.0

    out = atmo._advect_scalar_flux_eulerian(field, u, v, dt_seconds=5 * 86400.0, dx_meters=dx, dy_meters=dy)
    assert out.min() >= field.min() - 1e-6
    assert out.max() <= field.max() + 1e-6
    assert np.all(np.isfinite(out))


def test_centroid_shift_matches_courant_distance():
    """A pure zonal wind should shift a localized bump's mass-weighted centroid
    by close to the wind's own Courant distance (wind_speed * dt / dx), wrapping
    periodically -- confirms many small CFL-bounded substeps still add up to the
    physically correct net transport distance, not something shorter/longer."""
    H, W = 4, 128
    dx_val = 30000.0
    u_speed = 8.0
    dt_seconds = 5 * 86400.0
    expected_shift = u_speed * dt_seconds / dx_val

    field = np.zeros((H, W), dtype=np.float32)
    j0 = 20
    field[:, j0] = 1.0
    u = np.full((H, W), u_speed, dtype=np.float32)
    v = np.zeros((H, W), dtype=np.float32)
    dx = np.full((H, W), dx_val, dtype=np.float32)
    dy = dx_val

    out = atmo._advect_scalar_flux_eulerian(field, u, v, dt_seconds=dt_seconds, dx_meters=dx, dy_meters=dy)

    row = out[0, :].astype(np.float64)
    theta = 2.0 * np.pi * np.arange(W) / W
    centroid_angle = np.arctan2(np.sum(row * np.sin(theta)), np.sum(row * np.cos(theta))) % (2 * np.pi)
    centroid_col = centroid_angle / (2 * np.pi) * W

    expected_col = (j0 + expected_shift) % W
    diff = min(abs(centroid_col - expected_col), W - abs(centroid_col - expected_col))
    assert diff < 2.0, (
        f"centroid landed at col {centroid_col:.1f}, expected ~{expected_col:.1f} "
        f"(shift={expected_shift:.1f} cells)"
    )


def test_takes_many_substeps_at_real_production_dt(monkeypatch):
    """The whole point of this scheme: at real MONTHLY-mode substep dt and a
    realistic mid-latitude wind speed, it must take many small internal steps
    (not one single ~5000km jump like the old semi-Lagrangian sampler).
    Spies on the per-substep kernel to count actual calls, rather than
    recomputing the sizing formula independently (which would just check the
    implementation against itself)."""
    H, W = 4, 64
    dx_val = 25100.0  # ~50 deg N grid spacing
    dy_val = dx_val
    dt_seconds = 7.61 * 86400.0  # real MONTHLY-mode precip substep

    field = np.random.default_rng(2).uniform(0.0, 0.02, (H, W)).astype(np.float32)
    u = np.full((H, W), 7.5, dtype=np.float32)  # typical mid-latitude wind speed
    v = np.zeros((H, W), dtype=np.float32)
    dx = np.full((H, W), dx_val, dtype=np.float32)

    calls = {"n": 0}
    step_fn = atmo._advect_scalar_cfl_step_numba if atmo.NUMBA_AVAILABLE else atmo._advect_scalar_cfl_step

    def counting_step(*args, **kwargs):
        calls["n"] += 1
        return step_fn(*args, **kwargs)

    attr = "_advect_scalar_cfl_step_numba" if atmo.NUMBA_AVAILABLE else "_advect_scalar_cfl_step"
    monkeypatch.setattr(atmo, attr, counting_step)

    atmo._advect_scalar_flux_eulerian(field, u, v, dt_seconds=dt_seconds, dx_meters=dx, dy_meters=dy_val)

    # ~196 cells of total travel at max_courant=0.5 implies ~2 substeps/cell;
    # require at least an order of magnitude more than "a handful" to confirm
    # this isn't collapsing back to a small number of large jumps.
    assert calls["n"] > 50, f"expected many small substeps, only took {calls['n']}"
