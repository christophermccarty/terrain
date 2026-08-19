"""Tests for the (A10) tropopause radiative closure (`sesam_tropopause.py`),
the last piece of SESAM stage P5 (docs/SESAM_GAP_ANALYSIS.md section 7).

See the module docstring for the full provenance: Rstr,net's formula is a
read-only disambiguation from the reference Fortran (the paper states it
only in prose), the ozone climatology is an explicit constant-global
placeholder (not sourced from the paper), and c1tp's per-day folding is
resolved from the paper's own Conclusions text.

The supported climate path never calls this module (shares
``PlanetParams.enable_sesam_radiation``, no new gate added), so these tests
exercise a default-off diagnostic only.
"""
from __future__ import annotations

import numpy as np
import pytest

import sesam_tropopause as st
import sesam_vertical as sv
from planet_params import PlanetParams


def test_sesam_radiation_gate_defaults_off():
    assert PlanetParams().enable_sesam_radiation is False


# ---------------------------------------------------------------------------
# Ozone climatology
# ---------------------------------------------------------------------------


def test_ozone_profile_column_matches_target_du():
    levels = np.linspace(0.0, 30000.0, 61)
    rho = 1.2 * np.exp(-levels / 8000.0)
    rho_hw = rho[:, None, None]
    mmr = st.standard_ozone_mixing_ratio_profile_kgkg(levels, rho_hw, total_column_du=300.0)
    column_kg_m2 = st._trapezoid(mmr[:, 0, 0] * rho, levels)
    target_kg_m2 = 300.0 * st._DU_TO_KG_M2
    assert column_kg_m2 == pytest.approx(target_kg_m2, rel=1e-6)


def test_ozone_profile_nonnegative_and_peaks_near_stratosphere():
    """Checks the ozone *density* (mmr*rho_air), not the mixing ratio itself
    -- mixing ratio keeps rising toward the domain top since air density
    decays faster than the Gaussian ozone layer, which is physically
    expected (real stratospheric O3 mixing ratio does increase with height
    well above the density peak); the mass concentration is the physically
    meaningful "where is the ozone layer" check."""
    levels = np.linspace(0.0, 30000.0, 61)
    rho = 1.2 * np.exp(-levels / 8000.0)
    rho_hw = rho[:, None, None]
    mmr = st.standard_ozone_mixing_ratio_profile_kgkg(levels, rho_hw)
    assert np.all(mmr >= 0.0)
    density = mmr[:, 0, 0] * rho
    peak_level = levels[np.argmax(density)]
    assert 15000.0 < peak_level < 30000.0


def test_ozone_profile_scales_linearly_with_column():
    levels = np.linspace(0.0, 30000.0, 31)
    rho_hw = (1.2 * np.exp(-levels / 8000.0))[:, None, None]
    mmr_300 = st.standard_ozone_mixing_ratio_profile_kgkg(levels, rho_hw, total_column_du=300.0)
    mmr_600 = st.standard_ozone_mixing_ratio_profile_kgkg(levels, rho_hw, total_column_du=600.0)
    assert mmr_600 == pytest.approx(2.0 * mmr_300, rel=1e-9)


# ---------------------------------------------------------------------------
# Level-profile interpolation
# ---------------------------------------------------------------------------


def test_interpolate_level_profile_hand_value_midpoint():
    levels = np.array([0.0, 1000.0, 2000.0])
    profile = np.array([[[10.0]], [[20.0]], [[40.0]]])
    height = np.array([[500.0]])
    result = st.interpolate_level_profile(levels, profile, height)
    assert result[0, 0] == pytest.approx(15.0)


def test_interpolate_level_profile_exact_at_grid_point():
    levels = np.array([0.0, 1000.0, 2000.0])
    profile = np.array([[[10.0]], [[20.0]], [[40.0]]])
    height = np.array([[1000.0]])
    result = st.interpolate_level_profile(levels, profile, height)
    assert result[0, 0] == pytest.approx(20.0)


def test_interpolate_level_profile_clamps_outside_grid():
    levels = np.array([0.0, 1000.0, 2000.0])
    profile = np.array([[[10.0]], [[20.0]], [[40.0]]])
    below = st.interpolate_level_profile(levels, profile, np.array([[-500.0]]))
    above = st.interpolate_level_profile(levels, profile, np.array([[5000.0]]))
    assert below[0, 0] == pytest.approx(10.0)
    assert above[0, 0] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Rstr,net
# ---------------------------------------------------------------------------


def test_stratospheric_residual_hand_value():
    # 3 levels: surface, tropopause (exactly on-grid), model top.
    levels = np.array([0.0, 12000.0, 30000.0])
    down = np.array([[[300.0]], [[50.0]], [[0.0]]])
    up = np.array([[[400.0]], [[200.0]], [[220.0]]])
    ht = np.array([[12000.0]])
    toa_sw = np.array([[340.0]])
    result = st.stratospheric_net_radiative_residual(down, up, levels, ht, toa_sw)

    net_top = 0.0 - 220.0
    net_tropopause = 50.0 - 200.0
    expected_lw = net_top - net_tropopause
    expected_sw = st.FRAC_VU * (1.0 - st.I_O3_VU) * 340.0
    assert result[0, 0] == pytest.approx(expected_lw + expected_sw, rel=1e-9)


def test_stratospheric_residual_zero_sw_when_no_insolation():
    levels = np.array([0.0, 12000.0, 30000.0])
    down = np.zeros((3, 1, 1))
    up = np.zeros((3, 1, 1))
    ht = np.array([[12000.0]])
    result = st.stratospheric_net_radiative_residual(down, up, levels, ht, np.array([[0.0]]))
    assert result[0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Full (A10)/(A11) advance
# ---------------------------------------------------------------------------


def _flat_flux(levels, down_top, up_top, down_tro, up_tro, tropopause_m):
    """Two-level-bracket flux profile (surface value unused by the residual,
    included only so the array shapes are self-consistent)."""
    n = levels.shape[0]
    down = np.zeros((n, 1, 1))
    up = np.zeros((n, 1, 1))
    down[-1] = down_top
    up[-1] = up_top
    idx = np.searchsorted(levels, tropopause_m)
    down[idx] = down_tro
    up[idx] = up_tro
    return down, up


def test_advance_tropopause_height_direction_matches_a10_sign():
    """(A10): dHT/dt = -c1tp*(Rstr,net + S). A strongly positive net
    radiative residual (net input into the stratosphere) must lower HT."""
    # net_top ~ 0 (down_top=up_top=0); net_tropopause strongly negative
    # (up_tro=300 >> down_tro=0) -> lw_term = net_top - net_tropopause = +300,
    # i.e. the stratosphere is gaining net radiative energy.
    levels = np.array([0.0, 6000.0, 12000.0, 30000.0])
    down, up = _flat_flux(levels, down_top=0.0, up_top=0.0, down_tro=0.0, up_tro=300.0, tropopause_m=12000.0)
    ht0 = np.array([[12000.0]])
    lat = np.array([[0.0]])
    result = st.advance_tropopause_height(
        ht0, lat, itcz_latitude_rad=0.0, hadley_width_rad=np.deg2rad(30.0),
        longwave_downward_w_m2=down, longwave_upward_w_m2=up, levels_m=levels,
        toa_incoming_shortwave_w_m2=np.array([[340.0]]), dt_days=1.0,
        surface_elevation_m=np.array([[0.0]]),
    )
    assert result.r_strat_net_w_m2[0, 0] > 0.0
    assert result.tropopause_height_m[0, 0] < ht0[0, 0]


def test_advance_tropopause_height_clips_to_physical_bounds():
    levels = np.array([0.0, 6000.0, 12000.0, 30000.0])
    down, up = _flat_flux(levels, down_top=0.0, up_top=1.0e6, down_tro=0.0, up_tro=0.0, tropopause_m=12000.0)
    ht0 = np.array([[12000.0]])
    result = st.advance_tropopause_height(
        ht0, np.array([[0.0]]), itcz_latitude_rad=0.0, hadley_width_rad=np.deg2rad(30.0),
        longwave_downward_w_m2=down, longwave_upward_w_m2=up, levels_m=levels,
        toa_incoming_shortwave_w_m2=np.array([[340.0]]), dt_days=100.0,
        surface_elevation_m=np.array([[0.0]]), min_thickness_m=3000.0,
    )
    assert result.tropopause_height_m[0, 0] >= 3000.0
    assert result.tropopause_height_m[0, 0] <= st.MODEL_TOP_M


def test_advance_tropopause_height_agrees_with_vertical_module_pieces():
    """Cross-check: the assembled result must equal calling
    sesam_vertical's own tropopause_shape_s/tropopause_tendency directly
    with this module's Rstr,net -- advance_tropopause_height should not
    silently diverge from the P1 kernels it wraps."""
    levels = np.array([0.0, 6000.0, 12000.0, 30000.0])
    down, up = _flat_flux(levels, down_top=0.0, up_top=50.0, down_tro=10.0, up_tro=30.0, tropopause_m=12000.0)
    ht0 = np.array([[12000.0]])
    lat = np.array([[0.3]])
    result = st.advance_tropopause_height(
        ht0, lat, itcz_latitude_rad=0.0, hadley_width_rad=np.deg2rad(30.0),
        longwave_downward_w_m2=down, longwave_upward_w_m2=up, levels_m=levels,
        toa_incoming_shortwave_w_m2=np.array([[200.0]]), dt_days=0.5,
        surface_elevation_m=np.array([[0.0]]),
    )
    r_strat = st.stratospheric_net_radiative_residual(down, up, levels, ht0, np.array([[200.0]]))
    shape_s = sv.tropopause_shape_s(lat, 0.0, np.deg2rad(30.0))
    tendency = sv.tropopause_tendency(r_strat, shape_s)
    expected = np.clip(ht0 + tendency * 0.5, 3000.0, st.MODEL_TOP_M)
    assert result.tropopause_height_m[0, 0] == pytest.approx(expected[0, 0], rel=1e-9)
