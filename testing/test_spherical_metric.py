"""test_spherical_metric.py — closed-form checks on the spherical flux divergence.

These tests are *specifications*, not regression guards: every expected value is
derived analytically, so none of them depends on a tuning constant. That is the
same principle as `test_derivative_signs.py`, and it is what makes them able to
catch the metric-factor omission that aggregate precipitation metrics cannot.

The gap they pin down (ROADMAP Theme 1, "spherical metric completeness in
precipitation"): `_moisture_convergence_numba` takes raw index differences, so

  * the zonal term is under-weighted by 1/cos(phi) -- x2 at 60 deg, x3.9 at 75 deg
  * the meridional term is missing the cos(phi) flux weighting for converging
    meridians
  * both pole rows are left identically zero (`prange(1, H-1)`)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RADIUS = 6.371e6


def _grid(H: int = 180, W: int = 360):
    """Cell-centred lat/lon grid; row 0 is the north pole side."""
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi          # +pi/2 .. -pi/2
    lon = (np.arange(W) + 0.5) / W * 2.0 * np.pi            # 0 .. 2pi
    return lat, lon, np.meshgrid(lon, lat)                   # (lon2d, lat2d) broadcast


def _interior(field, lat, max_abs_lat_deg=75.0):
    """Mask off the polar caps, where centred differences on a coarse grid and
    the cos-floor both degrade accuracy. The pole rows get their own test."""
    keep = np.abs(np.degrees(lat)) <= max_abs_lat_deg
    return field[keep]


def test_positive_driver_normalization_is_scale_invariant():
    from atmosphere import _normalize_positive_driver

    field = np.array([[0.0, 1.0, 3.0], [2.0, 0.0, 6.0]], dtype=np.float64)
    normalized = _normalize_positive_driver(field)
    normalized_si = _normalize_positive_driver(field * 1e-9)

    np.testing.assert_allclose(normalized_si, normalized, rtol=1e-6, atol=0.0)
    assert float(np.mean(normalized)) == pytest.approx(1.0)


def test_positive_driver_normalization_keeps_calm_field_zero():
    from atmosphere import _normalize_positive_driver

    normalized = _normalize_positive_driver(np.zeros((8, 16), dtype=np.float64))
    assert normalized.dtype == np.float32
    assert not np.any(normalized)


# ---------------------------------------------------------------------------
# 1. Solid-body rotation must be exactly non-divergent
# ---------------------------------------------------------------------------

def test_solid_body_rotation_is_nondivergent():
    """u = U*cos(phi), v = 0, q = 1 has zero divergence everywhere.

    A rigid zonal rotation moves mass along latitude circles without piling it
    up anywhere, so any nonzero result is a pure artefact of the discretisation.
    """
    from atmosphere import flux_divergence_spherical

    H, W = 180, 360
    lat, _lon, (_lon2d, lat2d) = _grid(H, W)
    q = np.ones((H, W))
    u = 20.0 * np.cos(lat2d)
    v = np.zeros((H, W))

    div = flux_divergence_spherical(q, u, v, lat, radius_m=RADIUS)
    assert np.all(np.isfinite(div))
    assert np.max(np.abs(div)) < 1e-18, (
        f"solid-body rotation should be non-divergent, got max |div| = {np.max(np.abs(div)):.3e}"
    )


# ---------------------------------------------------------------------------
# 2. Zonally-varying zonal flow: divergence must scale as 1/cos(phi)
# ---------------------------------------------------------------------------

def test_zonal_flow_divergence_scales_as_inverse_cos_lat():
    """q = 1, v = 0, u = U*sin(lambda)  =>  div = U*cos(lambda) / (a*cos(phi)).

    This is THE discriminating case for the metric factor: the metric-free
    kernel returns a latitude-independent answer, while the correct divergence
    grows as 1/cos(phi) toward the poles.
    """
    from atmosphere import flux_divergence_spherical

    H, W = 180, 360
    lat, _lon, (lon2d, _lat2d) = _grid(H, W)
    U = 15.0
    q = np.ones((H, W))
    u = U * np.sin(lon2d)
    v = np.zeros((H, W))

    div = flux_divergence_spherical(q, u, v, lat, radius_m=RADIUS)
    expected = U * np.cos(lon2d) / (RADIUS * np.cos(lat)[:, None])

    got_i = _interior(div, lat)
    exp_i = _interior(expected, lat)
    rel = np.max(np.abs(got_i - exp_i)) / np.max(np.abs(exp_i))
    assert rel < 0.01, f"relative error vs analytic solution = {rel:.4f}"

    # And the latitude dependence itself: 60 deg must be ~2x the equator.
    def band_amp(deg):
        row = int(np.argmin(np.abs(np.degrees(lat) - deg)))
        return np.max(np.abs(div[row]))

    ratio = band_amp(60.0) / band_amp(0.0)
    assert 1.9 < ratio < 2.1, f"60deg/equator divergence ratio = {ratio:.3f}, expected ~2.0"


# ---------------------------------------------------------------------------
# 3. Uniform meridional flow converges as meridians converge
# ---------------------------------------------------------------------------

def test_uniform_meridional_flow_has_tan_lat_convergence():
    """q = 1, u = 0, v = V  =>  div = -V*tan(phi)/a.

    A uniform poleward flow *must* converge in the hemisphere it flows toward,
    purely because the meridians do. The metric-free kernel reports exactly zero
    here, since d(V)/di = 0 without the cos(phi) weighting.
    """
    from atmosphere import flux_divergence_spherical

    H, W = 180, 360
    lat, _lon, _ = _grid(H, W)
    V = 5.0
    q = np.ones((H, W))
    u = np.zeros((H, W))
    v = np.full((H, W), V)

    div = flux_divergence_spherical(q, u, v, lat, radius_m=RADIUS)
    expected = (-V * np.tan(lat) / RADIUS)[:, None] * np.ones((1, W))

    got_i = _interior(div, lat, 70.0)
    exp_i = _interior(expected, lat, 70.0)
    rel = np.max(np.abs(got_i - exp_i)) / np.max(np.abs(exp_i))
    assert rel < 0.02, f"relative error vs -V*tan(phi)/a = {rel:.4f}"

    # Northward flow converges in the NH and diverges in the SH.
    nh = np.degrees(lat) > 20.0
    sh = np.degrees(lat) < -20.0
    assert np.all(div[nh] < 0.0), "uniform northward flow must converge in the NH"
    assert np.all(div[sh] > 0.0), "uniform northward flow must diverge in the SH"


# ---------------------------------------------------------------------------
# 4. Pole rows must carry real values, not zeros
# ---------------------------------------------------------------------------

def test_pole_rows_are_not_identically_zero():
    """`_moisture_convergence_numba` uses `prange(1, H-1)`, leaving rows 0 and
    H-1 at exactly zero regardless of the flow. Polar precipitation is then
    structurally impossible, which matters for high-obliquity worlds."""
    from atmosphere import flux_divergence_spherical

    H, W = 90, 180
    lat, _lon, (lon2d, _lat2d) = _grid(H, W)
    q = np.ones((H, W))
    u = 12.0 * np.sin(lon2d)
    v = np.full((H, W), 3.0)

    div = flux_divergence_spherical(q, u, v, lat, radius_m=RADIUS)
    assert np.all(np.isfinite(div)), "pole rows must stay finite despite 1/cos(phi)"
    assert np.any(div[0] != 0.0), "north pole row is identically zero"
    assert np.any(div[-1] != 0.0), "south pole row is identically zero"


def test_spherical_divergence_is_globally_conservative():
    from atmosphere import flux_divergence_spherical

    H, W = 90, 180
    lat, _lon, _ = _grid(H, W)
    rng = np.random.default_rng(20260726)
    q = rng.uniform(0.001, 0.02, size=(H, W))
    u = rng.normal(0.0, 12.0, size=(H, W))
    v = rng.normal(0.0, 5.0, size=(H, W))

    div = flux_divergence_spherical(q, u, v, lat, radius_m=RADIUS)
    area_weight = np.cos(lat)[:, None]
    net = abs(float(np.sum(div * area_weight)))
    gross = float(np.sum(np.abs(div) * area_weight))
    assert net / gross < 1e-4


# ---------------------------------------------------------------------------
# 5. The current production kernel demonstrably lacks the metric
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not getattr(__import__("atmosphere"), "NUMBA_AVAILABLE", False),
    reason="numba kernel not available",
)
def test_legacy_kernel_is_latitude_flat_documenting_the_gap():
    """Documents *why* the gate exists: on the case from test 2 the legacy
    kernel's convergence is flat in latitude where the true answer grows as
    1/cos(phi). If this test ever fails, the legacy kernel gained a metric
    factor and this whole module should be revisited."""
    from atmosphere import _moisture_convergence_numba

    H, W = 180, 360
    lat, _lon, (lon2d, _lat2d) = _grid(H, W)
    q = np.ones((H, W), dtype=np.float32)
    u = (15.0 * np.sin(lon2d)).astype(np.float32)
    v = np.zeros((H, W), dtype=np.float32)

    conv = _moisture_convergence_numba(q, u, v)

    def band_amp(deg):
        row = int(np.argmin(np.abs(np.degrees(lat) - deg)))
        return float(np.max(np.abs(conv[row])))

    ratio = band_amp(60.0) / max(band_amp(0.0), 1e-30)
    assert ratio < 1.05, (
        f"legacy kernel unexpectedly shows latitude weighting (60deg/equator = {ratio:.3f})"
    )
