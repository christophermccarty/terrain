"""Tests for SESAM stage P5 sub-deliverable 3 -- longwave (`sesam_longwave.py`).

Equations transcribed from Appendix A8 of Willeit et al. (2022) (GMD 15,
5905-5948), verified against the article PDF, and cross-checked against a
second independently-citable source (PIK Report No. 81, Petoukhov,
Ganopolski & Claussen 2003) for the absorber-mass-path closure the GMD paper
states only as an abstract integral. See the module docstring for the full
provenance and the two findings recorded there (the shared beta_co2
exponent, and the (A106)/(A107) discretization's boundary-condition proof).

The supported climate path never calls this module
(``PlanetParams.enable_sesam_radiation`` is False), so these tests exercise a
default-off diagnostic only.
"""
from __future__ import annotations

import numpy as np
import pytest

import sesam_longwave as slw
from planet_params import PlanetParams

A8 = slw._a8_defaults()
_SIGMA = 5.670374419e-8


def test_sesam_radiation_gate_defaults_off():
    assert PlanetParams().enable_sesam_radiation is False


# ---------------------------------------------------------------------------
# Blackbody emission
# ---------------------------------------------------------------------------


def test_blackbody_emission_hand_value():
    b = slw.blackbody_emission_w_m2(np.array([[288.0]]))
    assert b == pytest.approx(_SIGMA * 288.0 ** 4, rel=1e-9)


# ---------------------------------------------------------------------------
# (A110)-(A112) transmission functions
# ---------------------------------------------------------------------------


def test_water_vapor_transmission_zero_mass_path_is_full_transmission():
    assert slw.water_vapor_lw_transmission(np.array([[0.0]])) == pytest.approx(1.0)


def test_water_vapor_transmission_hand_value():
    m = np.array([[2.0]])
    t = slw.water_vapor_lw_transmission(m)
    x = A8["beta0"] * 2.0
    expected = 1.0 / (
        1.0
        + A8["a1wv_lw"] * x ** A8["beta1wv"]
        + A8["a2wv_lw"] * x ** A8["beta2wv"]
        + A8["a3wv_lw"] * x ** A8["beta3wv"]
    )
    assert t == pytest.approx(expected, rel=1e-9)


def test_water_vapor_transmission_decreases_with_mass_path():
    small = slw.water_vapor_lw_transmission(np.array([[0.5]]))
    large = slw.water_vapor_lw_transmission(np.array([[5.0]]))
    assert large < small


def test_co2_transmission_zero_mass_path_is_full_transmission():
    assert slw.co2_lw_transmission(np.array([[0.0]])) == pytest.approx(1.0)


def test_co2_transmission_hand_value():
    m = np.array([[300.0]])
    t = slw.co2_lw_transmission(m)
    x = A8["beta0"] * 300.0
    power = x ** A8["beta_co2"]
    forcing = 1.0 - min(0.2, 0.1 * (300.0 / 1000.0) ** 2)
    ratio = (1.0 + A8["a0co2"] * A8["a1co2"] * power) / (1.0 + A8["a0co2"] * power)
    assert t == pytest.approx(forcing * ratio, rel=1e-9)


def test_co2_transmission_forcing_cap_is_active_at_very_high_mass():
    """Planted-violation guard: the min(0.2, ...) cap must actually engage,
    otherwise transmission would keep dropping without bound."""
    m = np.array([[50000.0]])  # far beyond the cap threshold
    t = slw.co2_lw_transmission(m)
    x = A8["beta0"] * 50000.0
    power = x ** A8["beta_co2"]
    ratio = (1.0 + A8["a0co2"] * A8["a1co2"] * power) / (1.0 + A8["a0co2"] * power)
    uncapped_forcing = 1.0 - 0.1 * (50000.0 / 1000.0) ** 2  # would be very negative
    capped_expected = 0.8 * ratio  # forcing floors at 1 - 0.2 = 0.8
    assert uncapped_forcing < 0.0  # sanity: the cap really is needed here
    assert t == pytest.approx(capped_expected, rel=1e-9)


def test_ozone_transmission_zero_mass_path_is_full_transmission():
    assert slw.ozone_lw_transmission(np.array([[0.0]])) == pytest.approx(1.0)


def test_ozone_transmission_hand_value():
    m = np.array([[0.01]])
    t = slw.ozone_lw_transmission(m)
    expected = 1.0 - A8["a0o3"] * 0.01 ** A8["beta_o3"]
    assert t == pytest.approx(max(expected, 0.0), rel=1e-9)


def test_ozone_transmission_clipped_to_unit_interval():
    huge = slw.ozone_lw_transmission(np.array([[1.0]]))
    assert 0.0 <= float(huge[0, 0]) <= 1.0


# ---------------------------------------------------------------------------
# (A113) cloud transmission, (A108)/(A109) combination
# ---------------------------------------------------------------------------


def test_cloud_transmission_outside_cloud_is_full_transmission():
    t = slw.cloud_lw_transmission(
        np.array([[500.0]]), np.array([[1000.0]]), np.array([[5.0]]), np.array([[False]])
    )
    assert t == pytest.approx(1.0)


def test_cloud_transmission_inside_cloud_hand_value():
    t = slw.cloud_lw_transmission(
        np.array([[500.0]]), np.array([[1000.0]]), np.array([[5.0]]), np.array([[True]])
    )
    expected = np.exp(-500.0 / 1000.0 * 5.0)
    assert t == pytest.approx(expected, rel=1e-9)


def test_combined_transmission_hand_value():
    dwv = np.array([[0.8]])
    dco2 = np.array([[0.9]])
    do3 = np.array([[0.95]])
    d = slw.combined_transmission(dwv, dco2, do3)
    assert d == pytest.approx(0.8 * 0.9 * 0.95, rel=1e-9)
    d_cloudy = slw.combined_transmission(dwv, dco2, do3, cloud_transmission=0.5)
    assert d_cloudy == pytest.approx(0.8 * 0.9 * 0.95 * 0.5, rel=1e-9)


# ---------------------------------------------------------------------------
# (A114)-(A116) absorber mass paths
# ---------------------------------------------------------------------------


def test_water_vapor_mass_path_hand_value_two_levels():
    # Uniform q, rho, and p == p0 (ratio 1) across two levels 0 m and 1000 m apart.
    q = np.full((2, 1, 1), 0.01)
    rho = np.full((2, 1, 1), 1.2)
    p = np.full((2, 1, 1), 101325.0)
    levels = np.array([0.0, 1000.0])
    m = slw.water_vapor_mass_path_g_cm2(q, p, rho, levels, np.array([[101325.0]]), 0, 1)
    # weight = q*rho*(p/p0)^k = 0.01*1.2*1 = 0.012 kg/m^3 uniformly -> trapezoid = 0.012*1000
    expected_kg_m2 = 0.012 * 1000.0
    assert m == pytest.approx(expected_kg_m2 * 0.1, rel=1e-9)


def test_water_vapor_mass_path_zero_humidity_is_zero():
    q = np.zeros((2, 1, 1))
    rho = np.full((2, 1, 1), 1.2)
    p = np.full((2, 1, 1), 101325.0)
    levels = np.array([0.0, 1000.0])
    m = slw.water_vapor_mass_path_g_cm2(q, p, rho, levels, np.array([[101325.0]]), 0, 1)
    assert m == pytest.approx(0.0, abs=1e-12)


def test_co2_mass_path_hand_value():
    r1 = np.array([[1.0]])  # surface
    r2 = np.array([[np.exp(-1.0)]])  # p/p0 at some level
    p0 = np.array([[101325.0]])
    g = 9.81
    m = slw.co2_mass_path_g_cm2(400.0, r1, r2, p0, g, A8)

    chi = 400.0 * 1e-6 * (44.01 / 28.97)
    k1 = A8["k_co2"] + 1.0
    expected_kg_m2 = chi * (101325.0 / g) / k1 * (r1[0, 0] ** k1 - r2[0, 0] ** k1)
    assert m == pytest.approx(abs(expected_kg_m2) * 0.1, rel=1e-9)


def test_co2_mass_path_increases_with_concentration():
    r1 = np.array([[1.0]])
    r2 = np.array([[0.5]])
    p0 = np.array([[101325.0]])
    low = slw.co2_mass_path_g_cm2(280.0, r1, r2, p0, 9.81)
    high = slw.co2_mass_path_g_cm2(800.0, r1, r2, p0, 9.81)
    assert high > low


def test_ozone_mass_path_hand_value_matches_water_vapor_structure():
    o3 = np.full((2, 1, 1), 8e-6)
    rho = np.full((2, 1, 1), 0.5)
    p = np.full((2, 1, 1), 30000.0)
    levels = np.array([15000.0, 16000.0])
    m = slw.ozone_mass_path_g_cm2(o3, p, rho, levels, np.array([[30000.0]]), 0, 1, A8)
    expected_kg_m2 = 8e-6 * 0.5 * 1.0 * 1000.0  # p/p0 == 1 here
    assert m == pytest.approx(expected_kg_m2 * 0.1, rel=1e-9)


def test_mass_path_shape_mismatch_raises():
    q = np.zeros((2, 1, 1))
    rho = np.zeros((2, 1, 1))
    p = np.zeros((2, 1, 2))
    levels = np.array([0.0, 1000.0])
    with pytest.raises(ValueError):
        slw.water_vapor_mass_path_g_cm2(q, p, rho, levels, 101325.0, 0, 1)


# ---------------------------------------------------------------------------
# (A106)/(A107) flux profile
# ---------------------------------------------------------------------------


def _symmetric_matrix(n, shape, values):
    """values[i][j] for i<=j; mirrored and unit-diagonal."""
    d = np.zeros((n, n) + shape)
    for i in range(n):
        d[i, i] = 1.0
    for (i, j), v in values.items():
        d[i, j] = v
        d[j, i] = v
    return d


def test_longwave_flux_downward_vanishes_at_top_of_atmosphere():
    """Structural boundary condition from (A106) itself: D(top,top)=1 makes
    F_down(top) = B(top) - B(top)*1 + (empty sum) = 0, for any transmission
    matrix -- no downward LW originates above the atmosphere."""
    n = 4
    shape = (1, 1)
    b = np.random.default_rng(0).uniform(200.0, 300.0, size=(n,) + shape) ** 0  # placeholder
    b = np.random.default_rng(0).uniform(1e-8, 5e-7, size=(n,) + shape)
    bs = np.array([[3e-7]])
    d = _symmetric_matrix(n, shape, {(0, 1): 0.7, (0, 2): 0.5, (0, 3): 0.3, (1, 2): 0.8, (1, 3): 0.4, (2, 3): 0.9})
    result = slw.longwave_flux_profile(b, bs, d)
    assert result.downward_w_m2[n - 1] == pytest.approx(0.0, abs=1e-12)


def test_longwave_flux_upward_at_surface_equals_surface_emission():
    """Structural boundary condition: D(0,0)=1 makes F_up(surface) =
    B(0) + (Bs-B(0))*1 - (empty sum) = Bs exactly, for any transmission
    matrix -- upward flux at the surface level is the surface's own
    blackbody emission."""
    n = 4
    shape = (1, 1)
    b = np.random.default_rng(1).uniform(1e-8, 5e-7, size=(n,) + shape)
    bs = np.array([[4e-7]])
    d = _symmetric_matrix(n, shape, {(0, 1): 0.7, (0, 2): 0.5, (0, 3): 0.3, (1, 2): 0.8, (1, 3): 0.4, (2, 3): 0.9})
    result = slw.longwave_flux_profile(b, bs, d)
    assert result.upward_w_m2[0] == pytest.approx(bs, rel=1e-9)


def test_longwave_flux_hand_value_three_levels():
    b0, b1, b2 = 3.0e-7, 2.5e-7, 1.0e-7
    bs = 3.2e-7
    b = np.array([[[b0]], [[b1]], [[b2]]])
    bs_arr = np.array([[bs]])
    d01, d02, d12 = 0.7, 0.4, 0.6
    d = _symmetric_matrix(3, (1, 1), {(0, 1): d01, (0, 2): d02, (1, 2): d12})
    result = slw.longwave_flux_profile(b, bs_arr, d)

    expected_down_0 = b0 * (1.0 - d01) + b1 * (d01 - d02)
    assert result.downward_w_m2[0, 0, 0] == pytest.approx(expected_down_0, rel=1e-9)

    expected_up_2 = b2 * (1.0 - d12) + b1 * (d12 - d02) + bs * d02
    assert result.upward_w_m2[2, 0, 0] == pytest.approx(expected_up_2, rel=1e-9)


def test_longwave_flux_shape_mismatch_raises():
    b = np.zeros((3, 1, 1))
    bs = np.zeros((1, 1))
    d = np.zeros((3, 3, 1, 2))
    with pytest.raises(ValueError):
        slw.longwave_flux_profile(b, bs, d)


# ---------------------------------------------------------------------------
# sky_combine
# ---------------------------------------------------------------------------


def test_sky_combine_hand_value():
    cs = np.array([[100.0]])
    cld = np.array([[40.0]])
    f = np.array([[0.3]])
    assert slw.sky_combine(cs, cld, f) == pytest.approx(0.3 * 40.0 + 0.7 * 100.0)


def test_sky_combine_identity_at_zero_and_full_cloud_fraction():
    cs = np.array([[100.0]])
    cld = np.array([[40.0]])
    assert slw.sky_combine(cs, cld, np.zeros((1, 1))) == pytest.approx(100.0)
    assert slw.sky_combine(cs, cld, np.ones((1, 1))) == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Full-column assembly (longwave_radiation)
# ---------------------------------------------------------------------------


def _realistic_profile(n, h, w):
    """A monotonically-decreasing-with-height temperature/humidity/ozone
    profile on a surface-to-30km grid, built from stage P1's own kernels --
    not hand-picked numbers, so the full pipeline is exercised on physically
    self-consistent inputs."""
    import sesam_vertical as sv

    levels = np.linspace(0.0, 30000.0, n)
    p0 = 101325.0
    t2m = np.full((h, w), 288.0)
    skin = np.full((h, w), 288.0)
    surface_kind = np.zeros((h, w), dtype=np.int64)
    qa = np.full((h, w), 0.008)
    zs = np.zeros((h, w))
    ht = np.full((h, w), 12000.0)
    structure = sv.compute_vertical_structure(
        levels,
        near_surface_air_temp_k=t2m, skin_temp_k=skin, surface_kind=surface_kind,
        near_surface_specific_humidity_kgkg=qa, surface_elevation_m=zs,
        tropopause_height_m=ht, p0_pa=p0, gravity=9.81, reference_temp_k=288.0,
    )
    return levels, structure, skin


def test_longwave_radiation_toa_flux_is_physically_bounded():
    """End-to-end (A106)-(A117) sanity check: OLR for an Earth-like profile
    should land in the observed climatological ballpark (Table 1's
    CLIMBER-X/observed thermal-up TOA is ~237-240 W/m^2), not merely be
    finite -- this is the same discipline as the P4 exit gate's raw
    unclamped precipitation-range check."""
    n, h, w = 8, 2, 2
    levels, structure, skin = _realistic_profile(n, h, w)
    o3 = np.zeros((n, h, w))  # no ozone: isolates the LW water/CO2/cloud path
    result = slw.longwave_radiation(
        structure.temperature_k, structure.specific_humidity_kgkg, o3,
        structure.pressure_pa, structure.air_density_kg_m3, levels,
        np.full((h, w), 101325.0), skin, co2_ppm=415.0, gravity=9.81,
        cloud_fraction=np.full((h, w), 0.3),
        cloud_base_height_m=np.full((h, w), 5000.0),
        cloud_top_height_m=np.full((h, w), 6000.0),
        cloud_optical_thickness=np.full((h, w), 10.0),
    )
    assert np.all(np.isfinite(result.outgoing_longwave_w_m2))
    assert np.all(result.outgoing_longwave_w_m2 > 100.0)
    assert np.all(result.outgoing_longwave_w_m2 < 400.0)


def test_longwave_radiation_downward_vanishes_at_top_of_atmosphere():
    """The sky-combined downward flux inherits the (A106) boundary condition
    exactly: both the clear and cloudy passes are individually zero at the
    top level, so sky_combine (a convex combination) is too."""
    n, h, w = 6, 2, 2
    levels, structure, skin = _realistic_profile(n, h, w)
    o3 = np.zeros((n, h, w))
    result = slw.longwave_radiation(
        structure.temperature_k, structure.specific_humidity_kgkg, o3,
        structure.pressure_pa, structure.air_density_kg_m3, levels,
        np.full((h, w), 101325.0), skin, co2_ppm=415.0, gravity=9.81,
        cloud_fraction=np.full((h, w), 0.5),
        cloud_base_height_m=np.full((h, w), 5000.0),
        cloud_top_height_m=np.full((h, w), 6000.0),
        cloud_optical_thickness=np.full((h, w), 10.0),
    )
    assert result.downward_w_m2[n - 1] == pytest.approx(0.0, abs=1e-9)


def test_longwave_radiation_more_co2_reduces_olr():
    """Physical sanity: more CO2 -> higher opacity -> lower OLR, holding
    everything else fixed (the same monotonicity direction as the real
    greenhouse effect, not merely "the function runs")."""
    n, h, w = 6, 2, 2
    levels, structure, skin = _realistic_profile(n, h, w)
    o3 = np.zeros((n, h, w))
    kwargs = dict(
        specific_humidity_profile_kg_kg=structure.specific_humidity_kgkg,
        ozone_mixing_ratio_profile_kg_kg=o3,
        pressure_profile_pa=structure.pressure_pa,
        air_density_profile_kg_m3=structure.air_density_kg_m3,
        levels_m=levels, surface_pressure_pa=np.full((h, w), 101325.0),
        surface_skin_temp_k=skin, gravity=9.81,
        cloud_fraction=np.zeros((h, w)),
        cloud_base_height_m=np.full((h, w), 5000.0),
        cloud_top_height_m=np.full((h, w), 6000.0),
        cloud_optical_thickness=np.zeros((h, w)),
    )
    olr_low = slw.longwave_radiation(structure.temperature_k, co2_ppm=280.0, **kwargs).outgoing_longwave_w_m2
    olr_high = slw.longwave_radiation(structure.temperature_k, co2_ppm=1200.0, **kwargs).outgoing_longwave_w_m2
    assert np.all(olr_high < olr_low)
