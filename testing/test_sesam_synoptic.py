"""Tests for SESAM stage P3 — synoptic/EKE closure (`sesam_synoptic.py`).

The kernels are transcribed from Appendix A5 of Willeit et al. (2022)
(GMD 15, 5905-5948).  These tests guard the physics transcription: the
(A54) Brunt–Väisälä frequency, the (A53) Eady-baroclinicity production, the
(A55) dissipation, the (A50)/(A51) macroturbulent diffusion coefficients,
the (A56)/(A57) synoptic wind and vertical velocity, the (A58) total wind,
and the (A59)/(A60) wind stress — plus the diagnostic steady-state EKE where
production balances dissipation.

The supported climate path never calls this module
(``PlanetParams.enable_sesam_dynamics`` is False), so these tests exercise a
default-off diagnostic only.
"""
from __future__ import annotations

import numpy as np
import pytest

import sesam_synoptic as ss
import sesam_vertical as sv
from planet_params import PlanetParams

H, W = 64, 128
LAT = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
A5 = ss._a5_defaults()

CD = np.full((H, W), 1.3e-3)


# ---------------------------------------------------------------------------
# (A54) Brunt–Väisälä frequency
# ---------------------------------------------------------------------------


def test_brunt_vaisala_frequency_hand_value():
    levels = np.array([0.0, 10000.0])
    # Linear potential-temperature increase of 30 K over 10 km.
    theta = np.zeros((2, H, W))
    theta[0] = 273.0
    theta[1] = theta[0] + 30.0
    n = ss.brunt_vaisala_frequency(
        potential_temperature_k=theta, levels_m=levels, gravity=9.81
    )
    dtheta_dz = 30.0 / 10000.0
    theta_mid = 0.5 * (273.0 + 303.0)
    expected = np.sqrt(9.81 / theta_mid * dtheta_dz)
    assert n[0, 0] == pytest.approx(expected, rel=1e-9)
    # Stable stratification -> real positive frequency everywhere.
    assert np.isfinite(n).all() and (n > 0.0).all()


# ---------------------------------------------------------------------------
# (A53) production / Eady rate
# ---------------------------------------------------------------------------


def test_eke_production_hand_value_and_nonnegativity():
    rate = np.full((H, W), 2.0e-5)
    prod = ss.eke_production(rate, a5=A5)
    assert prod[0, 0] == pytest.approx(A5["c1syn"] + A5["c2syn"] * 2.0e-5, rel=1e-9)
    # Zero rate -> baseline production only.
    assert ss.eke_production(np.zeros((H, W)), a5=A5)[0, 0] == pytest.approx(A5["c1syn"])
    # Topography damping scales the production (only when enabled).
    zs = np.zeros((H, W)); zs[10] = 1500.0  # half of the 3000 m reference
    prod_damped = ss.eke_production(rate, surface_elevation_m=zs, topography_damping_coeff=1.0, a5=A5)
    assert prod_damped[10, 0] == pytest.approx(prod[10, 0] * 0.5, rel=1e-9)


def test_eady_growth_rate_zero_shear_and_poleward_growth():
    levels = np.array([0.0, 5500.0, 11000.0])
    th = np.zeros((levels.size, H, W))
    th[1] = th[0] + 30.0
    th[2] = th[0] + 60.0
    p = np.empty_like(th)
    for k, lev in enumerate(levels):
        p[k] = 101325.0 * np.exp(-lev / 8000.0)
    # No vertical wind shear -> zero Eady rate everywhere.
    u = np.zeros_like(th)
    v = np.zeros_like(th)
    eady = ss.eady_growth_rate(u, v, th, levels_m=levels, pressure_pa=p,
                               gravity=9.81, omega=7.2921e-5, latitude_rad=LAT)
    assert np.abs(eady).max() == pytest.approx(0.0, abs=1e-12)
    # Uniform westerly shear -> rate ∝ |f| and grows poleward.
    for k in range(1, levels.size):
        u[k] = u[k - 1] + 10.0
    eady2 = ss.eady_growth_rate(u, v, th, levels_m=levels, pressure_pa=p,
                                gravity=9.81, omega=7.2921e-5, latitude_rad=LAT)
    j_eq = np.argmin(np.abs(LAT))
    j_hi = np.argmin(np.abs(LAT - np.pi / 3.0))
    assert abs(eady2[j_eq, 0]) < abs(eady2[j_hi, 0])  # larger |f| poleward
    # Zero at the equator (f = 0).
    assert abs(eady2[j_eq, 0]) < 1e-6


# ---------------------------------------------------------------------------
# (A55) dissipation, (A50)/(A51) diffusion
# ---------------------------------------------------------------------------


def test_eke_dissipation_hand_value_and_floor():
    coeff = ss.eke_dissipation_coefficient(CD, a5=A5)
    assert coeff[0, 0] == pytest.approx(A5["c3syn"] + A5["c4syn"] * 1.3e-3, rel=1e-9)
    diss = ss.eke_dissipation(np.full((H, W), 100.0), CD, a5=A5)
    assert diss[0, 0] == pytest.approx(coeff[0, 0] * 100.0**1.5, rel=1e-9)
    # Sub-floor EKE is clamped to the (EKE rms = 1 m^2 s^-2) floor.
    diss_low = ss.eke_dissipation(np.full((H, W), 0.0), CD, a5=A5)
    assert diss_low[0, 0] == pytest.approx(coeff[0, 0] * 1.0**1.5, rel=1e-9)


def test_diffusion_coefficients_hand_values():
    at = ss.horizontal_diffusion_coefficient(np.full((H, W), 100.0), a5=A5)
    assert at[0, 0] == pytest.approx(A5["c5syn"] * np.sqrt(100.0), rel=1e-9)
    aq = ss.moisture_diffusion_coefficient(np.full((H, W), 100.0), a5=A5)
    assert aq[0, 0] == pytest.approx(A5["c6syn"] * 100.0, rel=1e-9)


def test_steady_state_production_equals_dissipation():
    rate = np.full((H, W), 5.0e-5)
    prod = ss.eke_production(rate, a5=A5)
    keq = ss.eke_steady_state(prod, CD, a5=A5)
    diss = ss.eke_dissipation(keq, CD, a5=A5)
    np.testing.assert_allclose(diss, prod, rtol=1e-9)
    tend = ss.eke_relaxation_tendency(keq, prod, CD, a5=A5)
    np.testing.assert_allclose(tend, 0.0, atol=1e-12)
    # Analytical: K = (P / (c3 + c4 CD))^(2/3).
    coeff = ss.eke_dissipation_coefficient(CD, a5=A5)
    expected = (prod[0, 0] / coeff[0, 0]) ** (2.0 / 3.0)
    assert keq[0, 0] == pytest.approx(expected, rel=1e-9)
    # Stronger baroclinicity -> larger equilibrium EKE.
    assert np.max(ss.eke_steady_state(prod * 4.0, CD, a5=A5)) > np.max(keq)


# ---------------------------------------------------------------------------
# (A56)/(A57) synoptic wind / vertical velocity
# ---------------------------------------------------------------------------


def test_synoptic_surface_wind_hand_value_and_floor():
    eke = np.full((H, W), 100.0)
    usyn = ss.synoptic_surface_wind(eke, np.full((H, W), 0.5), np.full((H, W), 0.9), a5=A5)
    assert usyn[0, 0] == pytest.approx(A5["c7syn"] * 0.5 * 0.9 * np.sqrt(100.0), rel=1e-9)
    # Tiny EKE hits the 1 m/s minimum.
    usyn_min = ss.synoptic_surface_wind(np.zeros((H, W)), np.full((H, W), 1.0), np.full((H, W), 1.0), a5=A5)
    assert usyn_min.min() == pytest.approx(1.0)
    assert ss.synoptic_vertical_velocity(np.full((H, W), 100.0), a5=A5)[0, 0] == pytest.approx(
        A5["c8syn"] * np.sqrt(100.0), rel=1e-9
    )


def test_total_wind_and_ocean_floor():
    u = np.full((H, W), 3.0)
    v = np.full((H, W), 4.0)
    usyn = np.full((H, W), 0.0)
    w = ss.total_wind_magnitude(u, v, usyn)
    assert w[0, 0] == pytest.approx(5.0, rel=1e-9)
    # With synoptic contribution: sqrt(3^2 + 4^2 + usyn^2) >= 5.
    w2 = ss.total_wind_magnitude(u, v, np.full((H, W), 12.0))
    assert w2[0, 0] == pytest.approx(13.0, rel=1e-9)
    # Ocean floor of 5 m/s lifts calm ocean cells.
    ocean = np.arange(W) < W // 2
    mask = np.broadcast_to(ocean[None, :], (H, W))
    w_floor = ss.total_wind_magnitude(np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W)), ocean_mask=mask)
    assert w_floor[0, 0] == 5.0 and w_floor[0, W - 1] == 0.0


def test_wind_stress_hand_value():
    u = np.full((H, W), 3.0)
    v = np.full((H, W), 4.0)
    total = np.full((H, W), 5.0)
    taux, tauy = ss.wind_stress(u, v, total, CD, 1.289)
    assert taux[0, 0] == pytest.approx(1.3e-3 * 1.289 * 3.0 * 5.0, rel=1e-9)
    assert tauy[0, 0] == pytest.approx(1.3e-3 * 1.289 * 4.0 * 5.0, rel=1e-9)


def test_scalar_at_pressure_linear_interpolation():
    n = 4
    p = np.array([101000.0, 70000.0, 40000.0, 20000.0])
    q = np.array([280.0, 270.0, 250.0, 230.0])
    prof = np.repeat(q[:, None, None], 2, axis=1)
    pz = np.repeat(p[:, None, None], 2, axis=1)
    assert ss.scalar_at_pressure(prof, pz, 55000.0)[0, 0] == pytest.approx(260.0, rel=1e-9)
    assert ss.scalar_at_pressure(prof, pz, 5000.0)[0, 0] == pytest.approx(230.0)
    assert ss.scalar_at_pressure(prof, pz, 200000.0)[0, 0] == pytest.approx(280.0)


# ---------------------------------------------------------------------------
# (A58)-(A60) assembly
# ---------------------------------------------------------------------------


def _synthetic_state():
    levels = np.array([0.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0])
    skin = np.repeat((300.0 - 60.0 * np.sin(LAT) ** 2)[:, None], W, axis=1)
    zs = np.zeros((H, W))
    vs = sv.compute_vertical_structure(
        levels, near_surface_air_temp_k=skin - 2.0, skin_temp_k=skin,
        surface_kind=np.zeros((H, W), dtype=np.int64),
        near_surface_specific_humidity_kgkg=np.full((H, W), 0.008),
        surface_elevation_m=zs, tropopause_height_m=np.full((H, W), 12000.0),
        p0_pa=101325.0, gravity=9.81, reference_temp_k=288.0,
    )
    u = np.zeros((levels.size, H, W))
    for k, lev in enumerate(levels):
        u[k] = 5.0 + lev / 12000.0 * 15.0
    return vs, u, zs, levels


def test_compute_synoptic_shapes_finite_and_positive():
    vs, u, zs, levels = _synthetic_state()
    v = np.zeros_like(u)
    res = ss.compute_synoptic(
        potential_temperature_k=vs.potential_temperature_k, u_wind_z=u, v_wind_z=v,
        pressure_z=vs.pressure_pa, levels_m=levels,
        surface_u_m_s=u[0], surface_v_m_s=v[0], surface_elevation_m=zs,
        surface_kind=np.zeros((H, W), dtype=np.int64),
        drag_coefficient=CD, epsilon=np.full((H, W), 0.5), cos_alpha=np.full((H, W), 0.9),
        gravity=9.81, omega=7.2921e-5, rho0_kg_m3=1.289,
    )
    for arr in (
        res.eddy_kinetic_energy_m2_s2, res.production_m2_s3, res.dissipation_m2_s3,
        res.eady_growth_rate, res.brunt_vaisala_frequency, res.diffusion_coefficient_heat_m2_s,
        res.synoptic_surface_wind_m_s, res.synoptic_vertical_velocity_m_s, res.total_wind_m_s,
        res.wind_stress_u_pa, res.wind_stress_v_pa,
    ):
        assert arr.shape == (H, W)
        assert np.isfinite(arr).all()
    assert (res.production_m2_s3 >= 0.0).all()
    assert (res.eddy_kinetic_energy_m2_s2 > 0.0).all()
    assert (res.synoptic_surface_wind_m_s >= 1.0).all()
    # Total wind is never smaller than the surface wind magnitude.
    usfc = np.sqrt(u[0] ** 2 + v[0] ** 2)
    assert (res.total_wind_m_s >= usfc - 1e-9).all()
    # Stronger shear -> larger EKE (monotonic baroclinicity response).
    u2 = u * 2.0
    res2 = ss.compute_synoptic(
        potential_temperature_k=vs.potential_temperature_k, u_wind_z=u2, v_wind_z=v,
        pressure_z=vs.pressure_pa, levels_m=levels,
        surface_u_m_s=u2[0], surface_v_m_s=v[0], surface_elevation_m=zs,
        surface_kind=np.zeros((H, W), dtype=np.int64),
        drag_coefficient=CD, epsilon=np.full((H, W), 0.5), cos_alpha=np.full((H, W), 0.9),
        gravity=9.81, omega=7.2921e-5, rho0_kg_m3=1.289,
    )
    assert np.mean(res2.eddy_kinetic_energy_m2_s2) > np.mean(res.eddy_kinetic_energy_m2_s2)


def test_sesam_synoptic_gate_defaults_off():
    assert PlanetParams().enable_sesam_dynamics is False
