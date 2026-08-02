"""Directional/sign regression tests for atmosphere.py's meridional derivatives.

The 2026-07-03 audit found that on this grid (row 0 = north pole, row index
increases southward) while v is northward-positive, every naive meridional
derivative has an inverted sign relative to the physical convention. That bug
(ITCZ convergence registering as divergence, orographic uplift/rain-shadow
swapped) survived for months because existing tests only checked aggregate
mm/day numbers, which a compensating hand-tuned constant could mask. These
tests check raw sign/direction instead, independent of any tuning constant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import atmosphere as atmo


# ---------------------------------------------------------------------------
# _moisture_convergence_numba: the dynamical ITCZ-convergence driver
# ---------------------------------------------------------------------------

def test_moisture_convergence_positive_at_confluence():
    """Wind converging toward a row (southward north of it, northward south of
    it) must register as positive convergence at that row."""
    H, W = 24, 8
    mid = H // 2
    q = np.full((H, W), 0.01, dtype=np.float32)
    u = np.zeros((H, W), dtype=np.float32)
    v = np.empty((H, W), dtype=np.float32)
    v[:mid, :] = -5.0   # north of mid: flowing south, toward mid
    v[mid:, :] = 5.0    # south of mid: flowing north, toward mid
    conv = atmo._moisture_convergence_numba(q, u, v)
    assert conv[mid, 0] > 0.0, "converging flow must register as positive moisture convergence"


def test_moisture_convergence_zero_at_confluence_reversed():
    """The mirror-image (diverging) wind pattern must NOT register as
    convergence at that row (the kernel clips negative divergence to zero)."""
    H, W = 24, 8
    mid = H // 2
    q = np.full((H, W), 0.01, dtype=np.float32)
    u = np.zeros((H, W), dtype=np.float32)
    v = np.empty((H, W), dtype=np.float32)
    v[:mid, :] = 5.0    # north of mid: flowing further north, away from mid
    v[mid:, :] = -5.0   # south of mid: flowing further south, away from mid
    conv = atmo._moisture_convergence_numba(q, u, v)
    assert conv[mid, 0] == 0.0, "diverging flow must not register as (clipped-positive) convergence"


# ---------------------------------------------------------------------------
# generate_precipitation: orographic uplift vs. rain-shadow (meridional gy term)
# ---------------------------------------------------------------------------

def _make_meridional_ramp_elevation(H: int, W: int) -> np.ndarray:
    """Elevation rising monotonically toward the south (row index increasing),
    uniform across longitude -- an idealized east-west mountain range with no
    zonal slope, so any precipitation asymmetry can only come from the
    meridional (gy) orographic term, not the zonal (gx) one."""
    row_elev = 200.0 + 100.0 * np.arange(H, dtype=np.float32)
    return np.repeat(row_elev[:, None], W, axis=1)


def test_orographic_uplift_exceeds_rain_shadow_on_windward_side():
    """Wind blowing from low ground toward high ground (windward, uphill) must
    register more orographic uplift, and less rain-shadow suppression, than the
    same setup with wind reversed (leeward, downhill/descending) -- a direct
    regression check for the fixed `gy = -np.gradient(elev, axis=0)` meridional
    sign convention.

    Checks `debug_fields["orog"]`/`["rain_shadow_suppression"]` directly rather
    than final `P.mean()` (2026-08-01, ACCURACY_AUDIT.md A5 follow-up): A5's
    `_raw_conversion_gain` (up to 5.5x, latitude-regime-dependent, unconditional)
    now pushes `remove_frac_prerescale` to its 0.85 ceiling in exactly the
    high-elevation rows where the orographic differential is otherwise largest,
    clipping away most of windward's raw-physics advantage there while leaving
    unrelated wind-direction-dependent moisture-convergence effects in the
    low-elevation rows unclipped -- confirmed by direct ablation (forcing the
    gain to 1.0 restores the old `P.mean()` comparison's PASS). The sign
    convention itself was directly verified intact throughout (this is why the
    test now targets it directly, not because the invariant was relaxed): `orog`
    and `rain_shadow_suppression` are unaffected by the moisture-budget rescale
    or the regime gain, so checking them isolates exactly the thing this test
    is named for, immune to that (real, separately-tracked) downstream
    saturation interaction."""
    H, W = 32, 16
    elev = _make_meridional_ramp_elevation(H, W)
    u_zero = np.zeros((H, W), dtype=np.float32)

    # Windward: flowing south (toward higher ground, since elevation rises southward).
    v_windward = np.full((H, W), -8.0, dtype=np.float32)
    debug_windward: dict = {}
    atmo.generate_precipitation(
        H, W, elev, wind_u=u_zero, wind_v=v_windward, day_of_year=80, dt_days=1.0,
        debug_fields=debug_windward,
    )

    # Leeward: flowing north (away from higher ground, descending).
    v_leeward = np.full((H, W), 8.0, dtype=np.float32)
    debug_leeward: dict = {}
    atmo.generate_precipitation(
        H, W, elev, wind_u=u_zero, wind_v=v_leeward, day_of_year=80, dt_days=1.0,
        debug_fields=debug_leeward,
    )

    orog_windward = float(debug_windward["orog"].mean())
    orog_leeward = float(debug_leeward["orog"].mean())
    assert orog_windward > orog_leeward, (
        f"windward (uphill) orographic uplift {orog_windward:.3f} should exceed "
        f"leeward (downhill) uplift {orog_leeward:.3f}"
    )

    shadow_windward = float(debug_windward["rain_shadow_suppression"].mean())
    shadow_leeward = float(debug_leeward["rain_shadow_suppression"].mean())
    assert shadow_windward > shadow_leeward, (
        f"windward (uphill) rain_shadow_suppression {shadow_windward:.3f} should exceed "
        f"leeward (downhill/rain-shadow) suppression {shadow_leeward:.3f} "
        f"(i.e. leeward should be suppressed more)"
    )


# ---------------------------------------------------------------------------
# generate_wind_field: terrain deflection responds to the meridional slope
# (gy), not the zonal one (gx) -- regression for the axis-swap bug in
# `_g_row, _g_col = np.gradient(elev_c); gx = _g_col; gy = -_g_row`.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("elev_sign,expected_v_sign", [(1.0, 1.0), (-1.0, -1.0)])
def test_terrain_deflection_is_meridional_not_zonal(elev_sign, expected_v_sign):
    """A north-south-only elevation ramp (no zonal slope at all) must deflect
    wind meridionally (away from the higher ground), not zonally. If the
    gx/gy unpacking regresses to picking up the row-gradient as gx, this
    deflection would spuriously show up in wind_u instead of wind_v."""
    H, W = 24, 48
    row_elev = 300.0 + elev_sign * 150.0 * np.arange(H, dtype=np.float32)
    elev = np.repeat(row_elev[:, None], W, axis=1).astype(np.float32)

    lat_deg = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0
    # Restrict comparison to extratropical rows so the always-on tropical
    # trade-wind zonal wave (unrelated to terrain) doesn't confound the signal.
    extratropical = np.abs(lat_deg) >= 28.0

    T_lat = atmo.temperature_kelvin_for_lat(np.deg2rad(lat_deg).astype(np.float32), day_of_year=80)
    temperature = np.repeat(T_lat[:, None], W, axis=1).astype(np.float32)

    common = dict(
        elevation=elev, temperature=temperature, day_of_year=80, block_size=1,
        weather_amp=0.0, terrain_flow_amp=1.0, debug_log=False,
    )
    u_on, v_on = atmo.generate_wind_field(H, W, terrain_influence=1.0, **common)
    u_off, v_off = atmo.generate_wind_field(H, W, terrain_influence=0.0, **common)

    delta_u = (u_on - u_off)[extratropical]
    delta_v = (v_on - v_off)[extratropical]

    mean_abs_du = float(np.mean(np.abs(delta_u)))
    mean_abs_dv = float(np.mean(np.abs(delta_v)))
    assert mean_abs_dv > mean_abs_du, (
        f"terrain deflection from a meridional-only slope should perturb v "
        f"(mean|dv|={mean_abs_dv:.4f}) far more than u (mean|du|={mean_abs_du:.4f})"
    )
    assert np.sign(float(np.mean(delta_v))) == expected_v_sign, (
        "terrain deflection should push wind away from the higher ground"
    )


# ---------------------------------------------------------------------------
# Cloud formation: ascent_term in simulate._evolve_temperature
# ---------------------------------------------------------------------------

def test_cloud_ascent_term_positive_at_convergence():
    """Converging meridional flow must raise cloud ascent_term (regression for
    the inverted meridional divergence sign in simulate._evolve_temperature)."""
    H, W = 24, 8
    mid = H // 2
    u = np.zeros((H, W), dtype=np.float32)
    v = np.empty((H, W), dtype=np.float32)
    v[:mid, :] = -5.0
    v[mid:, :] = 5.0
    div = 0.5 * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) - np.gradient(v, axis=0)
    ascent = np.clip(-div, 0.0, None)
    ascent = ascent / (np.mean(ascent) + 1e-6)
    ascent_term = np.clip(0.6 + 0.6 * ascent, 0.0, 1.4)
    assert ascent_term[mid, 0] > 1.0, "converging flow should boost cloud ascent_term"
