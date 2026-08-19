"""Tests for SESAM stage P5 sub-deliverable 2 -- shortwave (`sesam_shortwave.py`).

Equations transcribed from Appendix A7 of Willeit et al. (2022) (GMD 15,
5905-5948), verified directly against the article PDF and cross-checked
against the reference Fortran implementation (2026-08-19; see the module
docstring for two corrected paper transcription errors: the (A87)/(A88)
visible/IR band swap, and the (A97) cloud-thickness exponent sign).

The supported climate path never calls this module
(``PlanetParams.enable_sesam_radiation`` is False), so these tests exercise a
default-off diagnostic only.
"""
from __future__ import annotations

import numpy as np
import pytest

import sesam_shortwave as ssw
from planet_params import PlanetParams

A7 = ssw._a7_defaults()


def test_sesam_radiation_gate_defaults_off():
    assert PlanetParams().enable_sesam_radiation is False


def test_mu0_matches_longwave_diffusivity_factor():
    """mu0 = 1/beta0, reusing Table A8's constant (module docstring note 5)."""
    assert A7["mu0"] == pytest.approx(1.0 / 1.66, rel=1e-9)


# ---------------------------------------------------------------------------
# Column water path
# ---------------------------------------------------------------------------


def test_column_water_path_hand_value():
    w = ssw.column_water_path_g_cm2(np.array([[25.0]]))
    assert w == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# (A79)-(A80) atmospheric scattering albedo
# ---------------------------------------------------------------------------


def test_scattering_albedo_zero_aerosol_gives_pure_rayleigh_vu_and_zero_ir():
    mu = np.array([[0.7]])
    tau = np.array([[0.0]])
    r_im = np.array([[0.003]])
    vu, ir = ssw.atmospheric_scattering_albedo(mu, tau, r_im)
    assert vu == pytest.approx(A7["r_sct"])
    assert ir == pytest.approx(0.0)


def test_scattering_albedo_hand_value_nonzero_aerosol():
    mu = np.array([[0.7]])
    tau = np.array([[0.2]])
    r_im = np.array([[0.003]])
    vu, ir = ssw.atmospheric_scattering_albedo(mu, tau, r_im)

    f1 = 0.7 ** A7["p1"]
    f2 = (0.55 * 0.2) ** A7["p2"]
    f3 = A7["alpha1"] - A7["alpha2"] * np.log(1.0 + A7["alpha3"] * 0.003)
    core = np.exp(-f1 * f2 * f3)
    expected_vu = 1.0 - (1.0 - A7["r_sct"]) * core
    expected_ir = 1.0 - core

    assert vu == pytest.approx(expected_vu, rel=1e-9)
    assert ir == pytest.approx(expected_ir, rel=1e-9)


def test_scattering_albedo_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ssw.atmospheric_scattering_albedo(np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# (A81)-(A82) cloud albedo
# ---------------------------------------------------------------------------


def test_cloud_albedo_zero_optical_thickness_equals_scattering_albedo():
    """tau_cld=0 -> exp(0)=1 -> alb_cld collapses to alb_sct exactly."""
    alb_sct_vu = np.array([[0.17]])
    alb_sct_ir = np.array([[0.05]])
    mu = np.array([[0.7]])
    tau_cld = np.array([[0.0]])
    vu, ir = ssw.cloud_albedo(alb_sct_vu, alb_sct_ir, mu, tau_cld)
    assert vu == pytest.approx(0.17)
    assert ir == pytest.approx(0.05)


def test_cloud_albedo_hand_value():
    alb_sct_vu = np.array([[0.17]])
    alb_sct_ir = np.array([[0.05]])
    mu = np.array([[0.6]])
    tau_cld = np.array([[10.0]])
    vu, ir = ssw.cloud_albedo(alb_sct_vu, alb_sct_ir, mu, tau_cld)

    b_c = A7["g_cld"] / (0.6 ** A7["p3"])
    decay = np.exp(-b_c * 10.0 ** A7["p4"])
    expected_vu = 1.0 - (1.0 - 0.17) * decay
    expected_ir = 1.0 - (1.0 - 0.05) * decay
    assert vu == pytest.approx(expected_vu, rel=1e-9)
    assert ir == pytest.approx(expected_ir, rel=1e-9)


# ---------------------------------------------------------------------------
# (A87)-(A89) transmission functions
# ---------------------------------------------------------------------------


def test_water_vapor_transmission_zero_mass_path_is_full_transmission():
    """a1wv_sw + a2wv_sw == 1.0 in Table A7 -- zero absorber gives I=1 exactly."""
    itf = ssw.water_vapor_sw_transmission_ir(np.array([[0.0]]))
    assert itf == pytest.approx(1.0, rel=1e-9)
    assert A7["a1wv_sw"] + A7["a2wv_sw"] == pytest.approx(1.0)


def test_water_vapor_transmission_decreases_with_mass_path():
    m_small = ssw.water_vapor_sw_transmission_ir(np.array([[1.0]]))
    m_large = ssw.water_vapor_sw_transmission_ir(np.array([[10.0]]))
    assert m_large < m_small


def test_aerosol_transmission_zero_mass_path_is_full_transmission():
    itf = ssw.aerosol_sw_transmission(np.array([[0.0]]), np.array([[0.003]]))
    assert itf == pytest.approx(1.0, rel=1e-9)


def test_aerosol_transmission_hand_value():
    itf = ssw.aerosol_sw_transmission(np.array([[0.5]]), np.array([[0.003]]))
    expected = np.exp(-A7["gamma1aer"] * 0.5 * 0.003 ** A7["gamma2aer"])
    assert itf == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# (A94)-(A105) absorber mass paths
# ---------------------------------------------------------------------------


def test_absorber_mass_paths_cs_family_hand_values():
    w = np.array([[4.0]])
    mu = np.array([[0.5]])
    hcld = np.array([[8000.0]])
    dcld = np.array([[500.0]])
    hq = np.array([[2000.0]])
    mu0 = A7["mu0"]

    mw = ssw.absorber_mass_paths(w, mu, hcld, dcld, hq)

    expected_cs = (1.0 / 0.5 + 1.0 / mu0) * 4.0
    expected_cs1 = (1.0 / 0.5) * 4.0
    expected_cs2 = expected_cs1 + (1.0 - np.exp(-0.25)) * (2.0 / mu0) * 4.0

    assert mw.cs == pytest.approx(expected_cs, rel=1e-9)
    assert mw.cs1 == pytest.approx(expected_cs1, rel=1e-9)
    assert mw.cs2 == pytest.approx(expected_cs2, rel=1e-9)


def test_absorber_mass_paths_cloud_family_uses_corrected_negative_exponent():
    """Planted-violation guard for the (A97) sign correction (module docstring
    note 2): the cloud-thickness contribution must stay bounded in [0, W] as
    Dcld grows, not diverge as the paper's literal +Dcld/Hq would."""
    w = np.array([[4.0]])
    mu = np.array([[0.5]])
    hcld = np.array([[8000.0]])
    hq = np.array([[2000.0]])
    mu0 = A7["mu0"]

    thin = ssw.absorber_mass_paths(w, mu, hcld, np.array([[1.0]]), hq)
    thick = ssw.absorber_mass_paths(w, mu, hcld, np.array([[1.0e6]]), hq)

    exp_hc_hq = np.exp(-8000.0 / 2000.0)
    icos = 1.0 / 0.5 + 1.0 / mu0
    bounded_limit = exp_hc_hq * (icos + 1.0) * 4.0  # Dcld -> infinity limit

    assert np.isfinite(thick.cld)
    assert thick.cld == pytest.approx(bounded_limit, rel=1e-6)
    assert thin.cld < thick.cld  # monotonically increasing and bounded, not diverging


def test_absorber_mass_paths_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ssw.absorber_mass_paths(
            np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 2))
        )


# ---------------------------------------------------------------------------
# (A75)-(A78) planetary albedo, (A83)-(A86) surface transmission
# ---------------------------------------------------------------------------


def test_planetary_albedo_isolates_layered_combinator_with_unit_transmissions():
    alb_layer = np.array([[0.2]])
    alb_sur = np.array([[0.1]])
    ones = np.array([[1.0]])
    vu, ir = ssw.planetary_albedo(alb_layer, alb_layer, alb_sur, alb_sur, ones, ones, 1.0, 1.0)
    expected = 0.2 + (1.0 - 0.2) ** 2 * 0.1 / (1.0 - 0.2 * 0.1)
    assert vu == pytest.approx(expected, rel=1e-9)
    assert ir == pytest.approx(expected, rel=1e-9)


def test_planetary_albedo_applies_ozone_and_cloud_transmission_multiplicatively():
    alb_layer = np.array([[0.0]])  # zero layer albedo isolates the transmission product
    alb_sur = np.array([[0.0]])
    ones = np.array([[1.0]])
    vu, ir = ssw.planetary_albedo(alb_layer, alb_layer, alb_sur, alb_sur, ones, ones, 0.5, 0.5, cloud_transmission=0.9)
    assert vu == pytest.approx(0.0 * 0.5 * 0.9)  # layer+surface term is 0 here
    assert ir == pytest.approx(0.0)


def test_surface_transmission_identity_at_zero_albedo_and_unit_itf():
    """Zero atmosphere/surface albedo and unit ITFs must pass all radiation
    straight through (term1 -> 1, term2 -> 0)."""
    zeros = np.array([[0.0]])
    ones = np.array([[1.0]])
    vu, ir = ssw.surface_transmission(
        zeros, zeros, zeros, zeros, zeros, zeros, ones, ones, ones, ones, 1.0, 1.0
    )
    assert vu == pytest.approx(1.0)
    assert ir == pytest.approx(1.0)


def test_surface_transmission_shape_mismatch_raises():
    ones = np.ones((2, 2))
    bad = np.ones((2, 3))
    with pytest.raises(ValueError):
        ssw.surface_transmission(ones, ones, ones, ones, ones, ones, ones, ones, ones, bad, 1.0, 1.0)


# ---------------------------------------------------------------------------
# (A69)-(A74) band/sky combination
# ---------------------------------------------------------------------------


def test_band_combine_hand_value():
    vu = np.array([[10.0]])
    ir = np.array([[20.0]])
    combined = ssw.band_combine(vu, ir)
    assert combined == pytest.approx(0.45 * 10.0 + 0.55 * 20.0)


def test_sky_combine_hand_value():
    cs = np.array([[100.0]])
    cld = np.array([[40.0]])
    f = np.array([[0.3]])
    combined = ssw.sky_combine(cs, cld, f)
    assert combined == pytest.approx(0.3 * 40.0 + 0.7 * 100.0)


def test_sky_combine_identity_at_zero_and_full_cloud_fraction():
    cs = np.array([[100.0]])
    cld = np.array([[40.0]])
    assert ssw.sky_combine(cs, cld, np.zeros((1, 1))) == pytest.approx(100.0)
    assert ssw.sky_combine(cs, cld, np.ones((1, 1))) == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Full-pipeline assembly
# ---------------------------------------------------------------------------


def _grid(value, shape=(4, 4)):
    return np.full(shape, value, dtype=np.float64)


def test_shortwave_radiation_stays_within_physical_bounds():
    result = ssw.shortwave_radiation(
        incoming_toa_w_m2=_grid(400.0),
        cos_zenith=_grid(0.6),
        cloud_fraction=_grid(0.4),
        cloud_top_height_m=_grid(6000.0),
        cloud_optical_thickness=_grid(5.0),
        cloud_geometric_thickness_m=_grid(1000.0),
        column_water_kg_m2=_grid(20.0),
        humidity_scale_height_m=_grid(2000.0),
        surface_albedo_vu=_grid(0.2),
        surface_albedo_ir=_grid(0.2),
        aerosol_optical_thickness=_grid(0.0),
        aerosol_imaginary_refractive_index=_grid(0.003),
    )
    assert np.all(np.isfinite(result.toa_upward_w_m2))
    assert np.all(np.isfinite(result.surface_downward_w_m2))
    assert np.all(result.toa_upward_w_m2 >= 0.0)
    assert np.all(result.toa_upward_w_m2 <= 400.0)
    assert np.all(result.surface_downward_w_m2 >= 0.0)
    assert np.all(result.surface_downward_w_m2 <= 400.0)


def test_shortwave_radiation_zero_cloud_fraction_matches_clear_sky_branch_only():
    result = ssw.shortwave_radiation(
        incoming_toa_w_m2=_grid(400.0),
        cos_zenith=_grid(0.6),
        cloud_fraction=_grid(0.0),
        cloud_top_height_m=_grid(6000.0),
        cloud_optical_thickness=_grid(5.0),
        cloud_geometric_thickness_m=_grid(1000.0),
        column_water_kg_m2=_grid(20.0),
        humidity_scale_height_m=_grid(2000.0),
        surface_albedo_vu=_grid(0.2),
        surface_albedo_ir=_grid(0.2),
        aerosol_optical_thickness=_grid(0.0),
        aerosol_imaginary_refractive_index=_grid(0.003),
    )
    expected_top = ssw.band_combine(
        _grid(400.0) * result.clear_sky_toa_albedo_vu,
        _grid(400.0) * result.clear_sky_toa_albedo_ir,
    )
    assert result.toa_upward_w_m2 == pytest.approx(expected_top)


def test_shortwave_radiation_higher_surface_albedo_increases_toa_upward():
    """Physical sign check: brighter surface must reflect more to space."""
    kwargs = dict(
        incoming_toa_w_m2=_grid(400.0),
        cos_zenith=_grid(0.6),
        cloud_fraction=_grid(0.2),
        cloud_top_height_m=_grid(6000.0),
        cloud_optical_thickness=_grid(5.0),
        cloud_geometric_thickness_m=_grid(1000.0),
        column_water_kg_m2=_grid(20.0),
        humidity_scale_height_m=_grid(2000.0),
        aerosol_optical_thickness=_grid(0.0),
        aerosol_imaginary_refractive_index=_grid(0.003),
    )
    dark = ssw.shortwave_radiation(surface_albedo_vu=_grid(0.1), surface_albedo_ir=_grid(0.1), **kwargs)
    bright = ssw.shortwave_radiation(surface_albedo_vu=_grid(0.8), surface_albedo_ir=_grid(0.8), **kwargs)
    assert np.all(bright.toa_upward_w_m2 >= dark.toa_upward_w_m2)


def test_shortwave_radiation_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ssw.shortwave_radiation(
            incoming_toa_w_m2=_grid(400.0, (4, 4)),
            cos_zenith=_grid(0.6, (4, 5)),
            cloud_fraction=_grid(0.2),
            cloud_top_height_m=_grid(6000.0),
            cloud_optical_thickness=_grid(5.0),
            cloud_geometric_thickness_m=_grid(1000.0),
            column_water_kg_m2=_grid(20.0),
            humidity_scale_height_m=_grid(2000.0),
            surface_albedo_vu=_grid(0.2),
            surface_albedo_ir=_grid(0.2),
            aerosol_optical_thickness=_grid(0.0),
            aerosol_imaginary_refractive_index=_grid(0.003),
        )
