"""Tests for SESAM stage P2 — wind assembly (`sesam_wind.py`).

The kernels are transcribed from Appendix A2 of Willeit et al. (2022)
(GMD 15, 5905-5948).  These tests guard the physics transcription: the
(A22) drag coefficient, the (A21) cross-isobar bisection solve and its
K-independence, spherical SLP gradients, the (A17)-(A18) surface geostrophic
wind and thermal-wind shear (sign and hand magnitude), the (A19)-(A20)
ageostrophic flow toward low pressure, the mass-conserving ageostrophic
vertical profile (paper text), the (A26)-(A27) katabatic gate and downslope
sign, the (A24)-(A25) Taylor rotation, and the (A16) assembly.

The supported climate path never calls this module
(``PlanetParams.enable_sesam_dynamics`` is False), so these tests exercise a
default-off diagnostic only.
"""
from __future__ import annotations

import numpy as np
import pytest

import sesam_dynamics as sd
import sesam_vertical as sv
import sesam_wind as sw
from planet_params import PlanetParams

H, W = 64, 128
LAT = (0.5 - (np.arange(H) + 0.5) / H) * np.pi

EARTH = dict(gravity=9.81, radius_m=6.371e6, omega=7.2921e-5)


# ---------------------------------------------------------------------------
# Spherical gradients
# ---------------------------------------------------------------------------


def test_horizontal_gradient_analytic_meridional():
    field = np.repeat(np.sin(LAT)[:, None], W, axis=1)
    dphi, dlam = sw.horizontal_gradient(field, LAT)
    # Interior matches the analytic derivative to O(dphi^2) grid accuracy.
    interior = slice(4, -4)
    expected = np.broadcast_to(np.cos(LAT)[interior, None], dphi[interior].shape)
    np.testing.assert_allclose(dphi[interior], expected, atol=2e-3)
    np.testing.assert_allclose(dlam, 0.0, atol=1e-12)


def test_horizontal_gradient_periodic_zonal_exact():
    lam = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False)
    field = np.repeat(np.cos(lam)[None, :], H, axis=0)
    _, dlam = sw.horizontal_gradient(field, LAT)
    expected = np.broadcast_to(-np.sin(lam)[None, :], field.shape)
    # Centred difference is grid-accurate (O(dlam^2), not exact).
    np.testing.assert_allclose(dlam, expected, atol=5e-4)


# ---------------------------------------------------------------------------
# (A22)-(A23) drag, (A21) cross-isobar angle
# ---------------------------------------------------------------------------


def test_drag_coefficient_hand_values():
    z0 = np.full((2, 2), 0.1)
    zoro = np.zeros((2, 2))
    kind = np.array([[0, 1], [2, 1]])
    cd = sw.drag_coefficient(z0, zoro, kind)
    assert cd[0, 0] == pytest.approx(1.3e-3)  # ocean constant
    expected_land = (0.4 / np.log(100.0 / 0.1)) ** 2
    assert cd[0, 1] == pytest.approx(expected_land, rel=1e-9)
    assert cd[1, 0] == pytest.approx(expected_land, rel=1e-9)  # ice sheet: land form
    # Orographic roughness increases the drag.
    cd_oro = sw.drag_coefficient(z0, np.full((2, 2), 2.0), kind)
    assert cd_oro[0, 1] > cd[0, 1]
    assert sw.oro_roughness_m(np.array([500.0]))[0] == pytest.approx(2.0)


def test_cross_isobar_angle_matches_independent_solve():
    cd = np.full((H, W), 3.0e-3)
    out = sw.cross_isobar_angle(cd, LAT, omega=EARTH["omega"])
    # Independent scalar solve at the actual row latitude nearest 45N.
    j = np.argmin(np.abs(LAT - np.pi / 4.0))
    f_abs = 2.0 * EARTH["omega"] * abs(np.sin(LAT[j]))
    rhs = 3.0e-3 / np.sqrt(f_abs)
    lo, hi = 0.0, np.pi / 4.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.sin(mid) / np.sqrt(1.0 - np.sin(2.0 * mid)) > rhs:
            hi = mid
        else:
            lo = mid
    assert abs(out["alpha_rad"][j, 0]) == pytest.approx(mid, abs=1e-3)
    # Hemisphere sign with symmetric magnitude.
    assert out["alpha_rad"][j, 0] > 0.0
    assert out["alpha_rad"][H - 1 - j, 0] < 0.0
    assert abs(out["alpha_rad"][j, 0]) == pytest.approx(abs(out["alpha_rad"][H - 1 - j, 0]), rel=1e-6)
    # Component consistency: epsilon from |alpha|, magnitude form of scab.
    a = np.abs(out["alpha_rad"])
    np.testing.assert_allclose(out["epsilon"], np.sqrt(1.0 - np.sin(2.0 * a)), rtol=1e-12)
    np.testing.assert_allclose(
        out["sin_cos_alpha"], np.sin(a) * np.cos(a), rtol=1e-12
    )
    np.testing.assert_allclose(out["sin_alpha"] ** 2 + out["cos_alpha"] ** 2, 1.0, atol=1e-12)


def test_cross_isobar_angle_clamps_and_monotonicity():
    # Absurdly large CD hits the upper clamp; tiny CD the lower clamp.
    big = sw.cross_isobar_angle(np.full((H, W), 1.0), LAT, omega=EARTH["omega"])
    small = sw.cross_isobar_angle(np.full((H, W), 1e-6), LAT, omega=EARTH["omega"])
    assert np.abs(big["alpha_rad"]).max() == pytest.approx(0.5)
    assert np.abs(small["alpha_rad"]).min() == pytest.approx(0.05)
    # Larger CD -> larger angle (midlatitudes).
    mid = sw.cross_isobar_angle(np.full((H, W), 5e-3), LAT, omega=EARTH["omega"])
    j45 = np.argmin(np.abs(LAT - np.pi / 4.0))
    assert abs(mid["alpha_rad"][j45, 0]) > abs(small["alpha_rad"][j45, 0])


# ---------------------------------------------------------------------------
# (A17)-(A18) surface geostrophic wind
# ---------------------------------------------------------------------------


def test_surface_geostrophic_wind_hand_value_and_signs():
    # Planted zonal subtropical high at 30N: easterlies equatorward of it,
    # westerlies poleward of it.
    slp = 101000.0 + 800.0 * np.exp(-((LAT - np.pi / 6.0) / 0.15) ** 2)[:, None]
    slp = np.repeat(slp, W, axis=1)
    dphi, dlam = sw.horizontal_gradient(slp, LAT)
    ug0, vg0 = sw.surface_geostrophic_wind(
        dphi, dlam, LAT, radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289
    )
    j_eq_side = np.argmin(np.abs(LAT - np.radians(15.0)))
    j_pol_side = np.argmin(np.abs(LAT - np.radians(45.0)))
    assert ug0[j_eq_side, 0] < 0.0  # trade easterlies
    assert ug0[j_pol_side, 0] > 0.0  # midlatitude westerlies
    assert np.abs(vg0).max() < 1e-9  # zonally uniform SLP -> no vg
    # Hand magnitude at 45N: ug0 = -(dp/dphi)/(rho*f*Re).
    f = 2.0 * EARTH["omega"] * np.sin(LAT[j_pol_side])
    expected = -dphi[j_pol_side, 0] / (1.289 * f * EARTH["radius_m"])
    assert ug0[j_pol_side, 0] == pytest.approx(expected, rel=1e-9)


def test_surface_geostrophic_wind_vg0_bounded_near_pole():
    # Module docstring note 9: an azonal SLP pattern that does NOT vanish at
    # the pole row (unlike a true continuous field) is the exact scenario
    # that produced thousands of m/s in the P2 exit-gate measurement.
    h = 512
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lam = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    slp = 101000.0 + 2000.0 * np.cos(lam)[None, :] * np.ones((h, 1))
    dphi, dlam = sw.horizontal_gradient(slp, lat)
    ug0, vg0 = sw.surface_geostrophic_wind(
        dphi, dlam, lat, radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289
    )
    assert np.isfinite(vg0).all()
    # Within the top/bottom row the undamped 1/cos(phi) would give ~1e3-1e4
    # m/s for this forcing; the damped result stays physically bounded.
    assert np.abs(vg0[0]).max() < 50.0
    assert np.abs(vg0[-1]).max() < 50.0
    # Disabling the safeguard reproduces the undamped blow-up.
    _, vg0_undamped = sw.surface_geostrophic_wind(
        dphi, dlam, lat, radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        rho0_kg_m3=1.289, damping=None,
    )
    assert np.abs(vg0_undamped[0]).max() > 500.0


def test_surface_geostrophic_wind_ug0_bounded_near_equator():
    # Module docstring note 11: even with the paper's own hard |f| floor
    # (3e-5), an ordinary meridional SLP gradient at low latitude (a
    # measured real-data pattern -- see docs/SESAM_GAP_ANALYSIS.md's
    # 2026-08-17 entry) drives ug0 to 100+ m/s once divided by the small
    # (but not floored) f near 15N/S.
    h = 512
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    w = 32
    # A smooth meridional SLP ramp of ordinary real-world magnitude
    # (~1.4 hPa per grid row, matching the measured Andes-region pattern).
    row_gradient_pa = 138.0
    slp = (101000.0 + row_gradient_pa * np.arange(h))[:, None] * np.ones((1, w))
    dphi, dlam = sw.horizontal_gradient(slp, lat)
    j_15 = np.argmin(np.abs(np.degrees(lat) - 15.0))
    ug0, _ = sw.surface_geostrophic_wind(
        dphi, dlam, lat, radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289
    )
    assert np.isfinite(ug0).all()
    assert np.abs(ug0[j_15]).max() < 50.0
    ug0_undamped, _ = sw.surface_geostrophic_wind(
        dphi, dlam, lat, radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        rho0_kg_m3=1.289, damping=None,
    )
    assert np.abs(ug0_undamped[j_15]).max() > 50.0


def test_surface_geostrophic_wind_damping_inert_away_from_frame_extremes():
    # The safeguard must not perturb the already-tested 45N physics: at
    # sin^2(45)=cos^2(45)=0.5, min(1, 5*0.5)=min(1, 3*0.5)=1 -- full damping
    # envelope evaluates to exactly 1 there.
    slp = 101000.0 + 800.0 * np.exp(-((LAT - np.pi / 6.0) / 0.15) ** 2)[:, None]
    slp = np.repeat(slp, W, axis=1)
    dphi, dlam = sw.horizontal_gradient(slp, LAT)
    ug0_a, vg0_a = sw.surface_geostrophic_wind(
        dphi, dlam, LAT, radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289
    )
    ug0_b, vg0_b = sw.surface_geostrophic_wind(
        dphi, dlam, LAT, radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        rho0_kg_m3=1.289, damping=None,
    )
    j_45 = np.argmin(np.abs(LAT - np.radians(45.0)))
    np.testing.assert_allclose(ug0_a[j_45], ug0_b[j_45], rtol=1e-9)
    np.testing.assert_allclose(vg0_a[j_45], vg0_b[j_45], rtol=1e-9)


# ---------------------------------------------------------------------------
# (A17)-(A18) thermal-wind shear
# ---------------------------------------------------------------------------


def test_thermal_wind_shear_sign_and_hand_magnitude():
    levels = np.array([0.0, 2000.0, 5000.0, 9000.0])
    # T decreases poleward everywhere and is zonally uniform: pure westerly shear.
    t_surface = 300.0 - 40.0 * np.sin(LAT) ** 2
    t_z = np.repeat(t_surface[None, :, None], levels.size, axis=0)
    t_z = np.repeat(t_z, W, axis=2)
    # Make levels progressively cooler but keep the same meridional gradient.
    for k in range(1, levels.size):
        t_z[k] = t_z[0] - 6.5e-3 * levels[k]
    shear_u, shear_v = sw.thermal_wind_shear(
        t_z, levels, LAT, radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        reference_temp_k=273.15, gravity=EARTH["gravity"], damping=None,
    )
    # Zero shear at the surface level.
    np.testing.assert_allclose(shear_u[0], 0.0, atol=1e-12)
    # NH: westerly shear aloft (ug increases with height).
    j45 = np.argmin(np.abs(LAT - np.pi / 4.0))
    assert shear_u[-1, j45, 0] > 0.0
    assert np.all(np.diff(shear_u[:, j45, 0]) > 0.0)
    # SH: also westerly shear aloft — poleward cooling strengthens the
    # westerlies in both hemispheres (the physical thermal-wind relation).
    assert shear_u[-1, H - 1 - j45, 0] > 0.0
    # Hand magnitude for one interval at 45N:
    # shear = -(g/(T0 f Re)) * dT/dphi * dz, with dT/dphi the exact analytic
    # gradient of the profile (the discrete gradient is used internally;
    # compare against the discrete value for exactness).
    dphi_exact, _ = sw.horizontal_gradient(t_z[0], LAT)
    f = 2.0 * EARTH["omega"] * np.sin(LAT[j45])
    coef = EARTH["gravity"] / (273.15 * f * EARTH["radius_m"])
    expected_step = -coef * dphi_exact[j45, 0] * (levels[1] - levels[0])
    assert shear_u[1, j45, 0] == pytest.approx(expected_step, rel=1e-9)
    assert np.abs(shear_v).max() < 1e-9  # zonally uniform -> no v shear


def test_thermal_wind_damping_suppresses_equator():
    levels = np.array([0.0, 5000.0])
    t_surface = 300.0 - 40.0 * np.sin(LAT) ** 2
    t_z = np.repeat(t_surface[None, :, None], levels.size, axis=0)
    t_z = np.repeat(t_z, W, axis=2)
    damped, _ = sw.thermal_wind_shear(
        t_z, levels, LAT, radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        reference_temp_k=273.15, gravity=EARTH["gravity"], damping=(5.0, 3.0),
    )
    undamped, _ = sw.thermal_wind_shear(
        t_z, levels, LAT, radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        reference_temp_k=273.15, gravity=EARTH["gravity"], damping=None,
    )
    j_eq = np.argmin(np.abs(LAT))
    # Equatorial damping factor min(1, 5 sin^2(lat)) * min(1, 3 cos^2(lat)).
    expected_factor = min(1.0, 5.0 * np.sin(LAT[j_eq]) ** 2) * min(1.0, 3.0 * np.cos(LAT[j_eq]) ** 2)
    assert damped[-1, j_eq, 0] == pytest.approx(undamped[-1, j_eq, 0] * expected_factor, rel=1e-9)


# ---------------------------------------------------------------------------
# (A19)-(A20) ageostrophic surface wind
# ---------------------------------------------------------------------------


def test_ageostrophic_surface_wind_crosses_toward_low_pressure():
    # Planted zonal subtropical high at 30N: surface flow must diverge away
    # from it toward low pressure on both sides.
    slp = 101000.0 + 800.0 * np.exp(-((LAT - np.pi / 6.0) / 0.15) ** 2)[:, None]
    slp = np.repeat(slp, W, axis=1)
    dphi, dlam = sw.horizontal_gradient(slp, LAT)
    scab = np.full((H, W), 0.4)
    ua, va = sw.ageostrophic_surface_wind(
        dphi, dlam, LAT, scab, radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289
    )
    j_eq_side = np.argmin(np.abs(LAT - np.radians(15.0)))
    j_pol_side = np.argmin(np.abs(LAT - np.radians(45.0)))
    assert va[j_eq_side, 0] < 0.0  # away from the high, equatorward
    assert va[j_pol_side, 0] > 0.0  # away from the high, poleward
    # Hand magnitude at 45N: va = -scab*(dp/dphi)/(rho*|f|*Re).
    f_abs = 2.0 * EARTH["omega"] * np.sin(LAT[j_pol_side])
    expected = -0.4 * dphi[j_pol_side, 0] / (1.289 * f_abs * EARTH["radius_m"])
    assert va[j_pol_side, 0] == pytest.approx(expected, rel=1e-9)
    # Zonally uniform SLP -> ua = 0.
    assert np.abs(ua).max() < 1e-12


def test_ageostrophic_surface_wind_ua_bounded_near_pole():
    h = 512
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lam = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    slp = 101000.0 + 2000.0 * np.cos(lam)[None, :] * np.ones((h, 1))
    dphi, dlam = sw.horizontal_gradient(slp, lat)
    scab = np.full((h, 64), 0.4)
    ua, va = sw.ageostrophic_surface_wind(
        dphi, dlam, lat, scab, radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289
    )
    assert np.isfinite(ua).all()
    assert np.abs(ua[0]).max() < 50.0
    assert np.abs(ua[-1]).max() < 50.0
    ua_undamped, _ = sw.ageostrophic_surface_wind(
        dphi, dlam, lat, scab, radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        rho0_kg_m3=1.289, damping=None,
    )
    assert np.abs(ua_undamped[0]).max() > 150.0


def test_ageostrophic_surface_wind_va_bounded_near_equator():
    # va shares ug0's equatorial breakdown with an even smaller |f| floor
    # (1e-5 vs 3e-5), so it is more exposed, not less (module docstring
    # note 11).
    h = 512
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    w = 32
    row_gradient_pa = 138.0
    slp = (101000.0 + row_gradient_pa * np.arange(h))[:, None] * np.ones((1, w))
    dphi, dlam = sw.horizontal_gradient(slp, lat)
    scab = np.full((h, w), 0.4)
    j_15 = np.argmin(np.abs(np.degrees(lat) - 15.0))
    _, va = sw.ageostrophic_surface_wind(
        dphi, dlam, lat, scab, radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289
    )
    assert np.isfinite(va).all()
    _, va_undamped = sw.ageostrophic_surface_wind(
        dphi, dlam, lat, scab, radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        rho0_kg_m3=1.289, damping=None,
    )
    assert np.abs(va[j_15]).max() < 0.5 * np.abs(va_undamped[j_15]).max()


# ---------------------------------------------------------------------------
# Ageostrophic vertical profile (mass conservation)
# ---------------------------------------------------------------------------


def test_ageostrophic_profile_column_mass_conservation():
    sigma = np.linspace(1.0, 0.05, 951)  # dense, surface first
    ua_sfc = np.full((H, W), 2.0)
    va_sfc = np.full((H, W), -1.0)
    ua_z, va_z = sw.ageostrophic_profile(
        ua_sfc, va_sfc, sigma, LAT, 0.2,
    )
    sig_pbl = sw.pbl_top_sigma(LAT)
    # Sigma-integrated ageostrophic flux vanishes per column (trapezoid).
    for arr, surf in ((ua_z, ua_sfc), (va_z, va_sfc)):
        integ = np.trapezoid(arr, sigma, axis=0) if hasattr(np, "trapezoid") else np.trapz(arr, sigma, axis=0)
        assert np.abs(integ).max() < 5e-3 * np.abs(surf).max() * 0.2
    # Profile structure: surface value through the PBL, zero outside the
    # PBL and compensation layer.
    k_pbl = np.argmin(np.abs(sigma - 0.95))
    j_mid = H // 2
    assert ua_z[k_pbl, j_mid, 0] == pytest.approx(2.0)
    k_free = np.argmin(np.abs(sigma - 0.5))
    assert ua_z[k_free, j_mid, 0] == pytest.approx(0.0, abs=1e-12)
    # Compensation magnitude: -(1 - sig_pbl)/depth.
    depth = 0.2
    k_comp = np.argmin(np.abs(sigma - 0.25))
    expected = -2.0 * (1.0 - sig_pbl[j_mid]) / depth
    assert ua_z[k_comp, j_mid, 0] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# (A26)-(A27) katabatic wind
# ---------------------------------------------------------------------------


def test_katabatic_wind_downslope_sign_and_gate():
    # Cone: elevation falls to the east and to the south uniformly.
    lam = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False)
    zs = 500.0 * np.cos(lam)[None, :] + 300.0 * np.sin(LAT)[:, None]
    zs = np.broadcast_to(zs, (H, W)).copy()
    dphi, dlam = sw.horizontal_gradient(zs, LAT)
    cd = np.full((H, W), 4.0e-3)
    t2m = np.full((H, W), 260.0)
    tskin = np.full((H, W), 255.0)  # cold surface under warmer air (inversion)
    uk, vk = sw.katabatic_wind(
        t2m, tskin, cd, dlam, dphi, LAT, radius_m=EARTH["radius_m"], gravity=EARTH["gravity"]
    )
    # Flow is downslope: uk opposes the zonal slope sign everywhere.
    assert np.all(uk * dlam <= 0.0)
    assert np.all(vk * dphi <= 0.0)
    # Hand magnitude at one cell.
    j, i = H // 2, W // 4
    slope_x = dlam[j, i] / (EARTH["radius_m"] * np.cos(LAT[j]))
    factor = EARTH["gravity"] * 100.0 / 4.0e-3 * (5.0 / 260.0)
    expected = np.sqrt(factor * abs(slope_x)) * np.sign(-slope_x)
    assert uk[j, i] == pytest.approx(expected, rel=1e-9)
    # Gated off when the surface is warmer than the air (no inversion).
    uk_off, vk_off = sw.katabatic_wind(
        t2m, tskin + 10.0, cd, dlam, dphi, LAT, radius_m=EARTH["radius_m"], gravity=EARTH["gravity"]
    )
    assert np.abs(uk_off).max() == 0.0 and np.abs(vk_off).max() == 0.0
    # Flat terrain -> no katabatic flow even under an inversion.
    uk_flat, vk_flat = sw.katabatic_wind(
        t2m, tskin, cd, np.zeros((H, W)), np.zeros((H, W)), LAT,
        radius_m=EARTH["radius_m"], gravity=EARTH["gravity"]
    )
    assert np.abs(uk_flat).max() == 0.0 and np.abs(vk_flat).max() == 0.0


# ---------------------------------------------------------------------------
# (A24)-(A25) Taylor surface wind
# ---------------------------------------------------------------------------


def test_taylor_surface_wind_rotation_and_zero_angle_limit():
    rng = np.random.default_rng(0)
    ug0 = rng.normal(0.0, 5.0, (H, W))
    vg0 = rng.normal(0.0, 5.0, (H, W))
    # Planted signed angle 0.3 NH / -0.3 SH.
    alpha = np.where(LAT[:, None] >= 0.0, 0.3, -0.3) * np.ones((H, W))
    eps = np.sqrt(1.0 - np.sin(2.0 * np.abs(alpha)))
    us, vs = sw.taylor_surface_wind(ug0, vg0, np.sin(alpha), np.cos(alpha), eps)
    expected_us = eps * (ug0 * np.cos(alpha) - vg0 * np.sin(alpha))
    expected_vs = eps * (vg0 * np.cos(alpha) + ug0 * np.sin(alpha))
    np.testing.assert_allclose(us, expected_us, rtol=1e-12)
    np.testing.assert_allclose(vs, expected_vs, rtol=1e-12)
    # Zero angle -> epsilon 1, us = ug0, vs = vg0.
    us0, vs0 = sw.taylor_surface_wind(
        ug0, vg0, np.zeros((H, W)), np.ones((H, W)), np.ones((H, W))
    )
    np.testing.assert_allclose(us0, ug0)
    np.testing.assert_allclose(vs0, vg0)


# ---------------------------------------------------------------------------
# u500 interpolation
# ---------------------------------------------------------------------------


def test_u_at_pressure_linear_interpolation():
    n = 4
    p = np.array([101000.0, 70000.0, 40000.0, 20000.0])
    u = np.array([0.0, 10.0, 30.0, 50.0])
    u_z = np.repeat(u[:, None, None], 2, axis=1)
    p_z = np.repeat(p[:, None, None], 2, axis=1)
    got = sw.u_at_pressure(u_z, p_z, 55000.0)
    assert got[0, 0] == pytest.approx(20.0, rel=1e-9)
    # Above the column top clamps to the top level; below surface to surface.
    assert sw.u_at_pressure(u_z, p_z, 5000.0)[0, 0] == pytest.approx(50.0)
    assert sw.u_at_pressure(u_z, p_z, 200000.0)[0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# (A16) assembly
# ---------------------------------------------------------------------------


def _assembled_wind():
    tsl_z = 300.0 - 60.0 * np.sin(LAT) ** 2
    skin = np.repeat(tsl_z[:, None], W, axis=1)
    zs = np.zeros((H, W))
    slp_res = sd.compute_slp(
        skin_temp_k=skin, surface_elevation_m=zs, sin_cos_alpha_bar=0.433,
        gravity=EARTH["gravity"], radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        p0_pa=101100.0, reference_temp_k=273.15,
    )
    levels = np.array([0.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0])
    vs = sv.compute_vertical_structure(
        levels, near_surface_air_temp_k=skin - 2.0, skin_temp_k=skin,
        surface_kind=np.zeros((H, W), dtype=np.int64),
        near_surface_specific_humidity_kgkg=np.full((H, W), 0.008),
        surface_elevation_m=zs, tropopause_height_m=np.full((H, W), 12000.0),
        p0_pa=101100.0, gravity=EARTH["gravity"], reference_temp_k=288.0,
    )
    wind = sw.compute_wind(
        slp_pa=slp_res.slp_pa, temperature_z=vs.temperature_k, levels_m=levels,
        pressure_z=vs.pressure_pa, skin_temp_k=skin, t2m_k=skin - 1.0,
        surface_elevation_m=zs, surface_kind=np.zeros((H, W), dtype=np.int64),
        roughness_m=np.full((H, W), 0.1),
        gravity=EARTH["gravity"], radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        rho0_kg_m3=1.289, reference_temp_k=273.15,
    )
    return wind


def test_compute_wind_circulation_signs_and_finiteness():
    wind = _assembled_wind()
    for arr in (
        wind.surface_u_m_s, wind.surface_v_m_s, wind.u_z_m_s, wind.v_z_m_s,
        wind.geostrophic_u_z_m_s, wind.ageostrophic_u_z_m_s,
    ):
        assert np.isfinite(arr).all()

    def at(deg: float) -> int:
        return int(np.argmin(np.abs(LAT - np.radians(deg))))

    # Trades (easterlies + equatorward), westerlies (+ poleward), polar easterlies.
    assert wind.surface_u_m_s[at(15.0), 0] < 0.0
    assert wind.surface_v_m_s[at(15.0), 0] < 0.0
    assert wind.surface_u_m_s[at(45.0), 0] > 0.0
    assert wind.surface_v_m_s[at(45.0), 0] > 0.0
    assert wind.surface_u_m_s[at(75.0), 0] < 0.0
    # SH mirror.
    assert wind.surface_u_m_s[at(-15.0), 0] < 0.0
    assert wind.surface_v_m_s[at(-15.0), 0] > 0.0
    assert wind.surface_u_m_s[at(-45.0), 0] > 0.0
    assert wind.surface_v_m_s[at(-45.0), 0] < 0.0
    # 500 hPa westerly jet aloft in midlatitudes, stronger than the surface wind.
    assert wind.u500_pa_zonal_m_s[at(45.0)] > wind.surface_u_m_s[at(45.0), 0]
    assert wind.u500_pa_zonal_m_s.shape == (H,)


def test_compute_wind_shapes_and_reproducibility():
    wind = _assembled_wind()
    n = wind.u_z_m_s.shape[0]
    assert wind.u_z_m_s.shape == (n, H, W)
    assert wind.ageostrophic_v_z_m_s.shape == (n, H, W)
    wind2 = _assembled_wind()
    np.testing.assert_array_equal(wind.surface_u_m_s, wind2.surface_u_m_s)


def test_compute_wind_surface_damping_passthrough():
    # Module docstring note 11: surface_damping is independent of
    # thermal_wind_damping and defaults to the same (c_eq, c_pol) envelope.
    wind_default = _assembled_wind()
    tsl_z = 300.0 - 60.0 * np.sin(LAT) ** 2
    skin = np.repeat(tsl_z[:, None], W, axis=1)
    zs = np.zeros((H, W))
    slp_res = sd.compute_slp(
        skin_temp_k=skin, surface_elevation_m=zs, sin_cos_alpha_bar=0.433,
        gravity=EARTH["gravity"], radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        p0_pa=101100.0, reference_temp_k=273.15,
    )
    levels = np.array([0.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0])
    vs = sv.compute_vertical_structure(
        levels, near_surface_air_temp_k=skin - 2.0, skin_temp_k=skin,
        surface_kind=np.zeros((H, W), dtype=np.int64),
        near_surface_specific_humidity_kgkg=np.full((H, W), 0.008),
        surface_elevation_m=zs, tropopause_height_m=np.full((H, W), 12000.0),
        p0_pa=101100.0, gravity=EARTH["gravity"], reference_temp_k=288.0,
    )
    wind_undamped = sw.compute_wind(
        slp_pa=slp_res.slp_pa, temperature_z=vs.temperature_k, levels_m=levels,
        pressure_z=vs.pressure_pa, skin_temp_k=skin, t2m_k=skin - 1.0,
        surface_elevation_m=zs, surface_kind=np.zeros((H, W), dtype=np.int64),
        roughness_m=np.full((H, W), 0.1),
        gravity=EARTH["gravity"], radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        rho0_kg_m3=1.289, reference_temp_k=273.15, surface_damping=None,
    )
    # Near the equatorial row the two must differ (damping suppresses ug0);
    # at 45N (damp == 1 exactly) they must agree.
    j_eq = np.argmin(np.abs(LAT - np.radians(5.0)))
    j_45 = np.argmin(np.abs(LAT - np.radians(45.0)))
    assert abs(wind_default.surface_u_m_s[j_eq, 0]) < abs(wind_undamped.surface_u_m_s[j_eq, 0])
    np.testing.assert_allclose(
        wind_default.surface_u_m_s[j_45], wind_undamped.surface_u_m_s[j_45], rtol=1e-9
    )


def test_sesam_dynamics_gate_defaults_off():
    assert PlanetParams().enable_sesam_dynamics is False
