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


# ---------------------------------------------------------------------------
# (A52) prognostic K: advection + diffusion transport
# ---------------------------------------------------------------------------

_RADIUS_M = 6.371e6


def _geometry(h, w):
    area, xlen, ylen = ss.spherical_transport_geometry(h, w, _RADIUS_M)
    dx = ss.zonal_center_spacing_m(h, w, _RADIUS_M)
    dy = _RADIUS_M * (np.pi / h)
    return area, xlen, ylen, dx, dy


def test_spherical_transport_geometry_matches_sphere_area_and_poles():
    h, w = 8, 16
    area, xlen, ylen = ss.spherical_transport_geometry(h, w, _RADIUS_M)
    assert area.shape == (h, w) and xlen.shape == (h, w) and ylen.shape == (h + 1, w)
    # Cell areas sum to the full sphere.
    assert np.sum(area) == pytest.approx(4.0 * np.pi * _RADIUS_M ** 2, rel=1e-9)
    # North/south faces shrink to exactly zero at the poles.
    assert np.allclose(ylen[0], 0.0) and np.allclose(ylen[-1], 0.0)


def test_eke_diffusion_step_hand_value_single_substep():
    """Hand-computable (A52) diffusion: one interior peak, one Euler substep.

    3x4 grid, uniform 1000 m spacing, c5syn overridden to 1.0 so
    AT = sqrt(K).  A single nonzero cell (K=4 -> AT=2) has zero-AT neighbours
    on every side (AT=0 there), so every face diffusivity is the average of
    (2, 0) = 1.0.  The exact per-face self-loss geometry for this uniform
    1000 m grid is ``4 * x_len/(dx*area) = 4 * 1000/(1000*1e6) = 4e-6``
    (four identical faces, each contributing ``x_len/(dx*area)``), so the
    diffusion number at the peak is ``AT * 4e-6 * dt_seconds``; dt_days=0.5
    keeps it at 0.3456 < 0.4, so the kernel takes exactly one substep and the
    result is hand-computable: each of the 4 face fluxes is
    ``1.0 * 4.0 / 1000 * 1000 = 4.0`` (m^2 s^-2 m^2 s^-1), and the peak loses
    ``4 * 4.0 * 43200 = 691,200`` (m^2 s^-2 * m^2) of its 4,000,000 initial
    mass while each neighbour gains 172,800 -- conserving the total exactly.
    """
    a5 = dict(A5)
    a5["c5syn"] = 1.0
    k0 = np.zeros((3, 4))
    k0[1, 1] = 4.0
    res = ss.eke_diffusion_step(
        k0, dx_m=1000.0, dy_m=1000.0, dt_days=0.5,
        cell_area_m2=1e6, x_face_length_m=1000.0, y_face_length_m=1000.0,
        eke_floor=0.0, a5=a5,
    )
    assert res.substeps == 1
    assert res.eke_m2_s2[1, 1] == pytest.approx(3.3088, rel=1e-9)
    for j, i in [(1, 0), (1, 2), (0, 1), (2, 1)]:
        assert res.eke_m2_s2[j, i] == pytest.approx(0.1728, rel=1e-9)
    # Untouched corners stay exactly zero.
    for j, i in [(0, 0), (0, 2), (2, 0), (2, 2), (0, 3), (1, 3), (2, 3)]:
        assert res.eke_m2_s2[j, i] == 0.0
    assert res.residual_m2_s2 == pytest.approx(0.0, abs=1e-6)


def test_eke_diffusion_omitting_face_averaging_breaks_symmetry_planted_violation():
    """Plant the (A50) face-averaging omission and show it desymmetrises a
    symmetric bump -- the same setup as the hand-value test above.

    The correct implementation face-averages AT before forming each flux, so
    a symmetric bump diffuses identically to all four neighbours (0.1728
    each, per the hand-value test). Using the *donor* cell's own AT instead
    (no averaging) still conserves the global total (the same flux value is
    subtracted from one side and added to the other either way), but breaks
    the physical symmetry: the neighbour on the AT=0 side of each face gets
    no flux at all (donor AT=0), while the neighbour on the peak's own side
    gets double the correct amount.  This is exactly the failure the module
    docstring's note 6 warns about, reproduced here without touching the
    module so a real regression of the averaging would be caught the same
    way.
    """
    a5 = dict(A5)
    a5["c5syn"] = 1.0
    k0 = np.zeros((3, 4))
    k0[1, 1] = 4.0
    correct = ss.eke_diffusion_step(
        k0, dx_m=1000.0, dy_m=1000.0, dt_days=0.5,
        cell_area_m2=1e6, x_face_length_m=1000.0, y_face_length_m=1000.0,
        eke_floor=0.0, a5=a5,
    )
    neighbours_correct = [correct.eke_m2_s2[j, i] for j, i in [(1, 0), (1, 2), (0, 1), (2, 1)]]
    assert len(set(np.round(neighbours_correct, 6))) == 1  # all four identical

    def buggy_one_sided_diffusion(k0, dt_days, c5):
        depth = k0.copy()
        dt_seconds = dt_days * 86400.0
        at = c5 * np.sqrt(np.maximum(depth, 0.0))
        east_flux = at * (depth - np.roll(depth, -1, axis=1)) / 1000.0 * 1000.0  # no averaging
        west_flux_in = np.roll(east_flux, 1, axis=1)
        h, w = depth.shape
        face_flux = np.zeros((h + 1, w))
        face_flux[1:-1] = at[:-1] * (depth[:-1] - depth[1:]) / 1000.0 * 1000.0  # no averaging
        mass = depth * 1e6 + dt_seconds * (west_flux_in - east_flux + face_flux[:-1] - face_flux[1:])
        return mass / 1e6

    buggy = buggy_one_sided_diffusion(k0, 0.5, a5["c5syn"])
    neighbours_buggy = [buggy[j, i] for j, i in [(1, 0), (1, 2), (0, 1), (2, 1)]]
    # The buggy, un-averaged scheme desymmetrises: two neighbours get nothing,
    # two get double -- unlike the real, face-averaged implementation.
    assert len(set(np.round(neighbours_buggy, 6))) > 1
    assert buggy[1, 0] == pytest.approx(0.0, abs=1e-9)
    assert buggy[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert buggy[1, 2] == pytest.approx(2.0 * 0.1728, rel=1e-6)
    assert buggy[2, 1] == pytest.approx(2.0 * 0.1728, rel=1e-6)


def test_eke_diffusion_cfl_substepping_prevents_maximum_principle_violation_planted_violation():
    """Plant the CFL-substepping omission and show it lets a single big Euler
    step wildly overshoot the physical bound.

    Pure diffusion (no source) can never raise a field's peak above its own
    initial peak (discrete maximum principle). With a large initial K
    (AT = 100) and a 1-day step, the real dissipation-free kernel needs 44
    substeps and stays within [K0.min(), K0.max()]; forcing a single
    unsubstepped Euler step over the same dt (dropping the module's own CFL
    substepping) blows past that bound by more than 4x and drives a
    neighbour cell negative -- exactly the failure `diffusion_r_limit`
    substepping exists to prevent.
    """
    a5 = dict(A5)
    a5["c5syn"] = 1.0
    k0 = np.zeros((3, 4))
    k0[1, 1] = 10000.0
    correct = ss.eke_diffusion_step(
        k0, dx_m=1000.0, dy_m=1000.0, dt_days=1.0,
        cell_area_m2=1e6, x_face_length_m=1000.0, y_face_length_m=1000.0,
        eke_floor=0.0, a5=a5,
    )
    assert correct.substeps > 1
    assert correct.eke_m2_s2.max() <= k0.max() + 1e-6
    assert correct.eke_m2_s2.min() >= 0.0

    def buggy_single_step_no_cfl(k0, dt_days, c5):
        depth = k0.copy()
        dt_seconds = dt_days * 86400.0  # the *entire* step in one go, no substepping
        at = c5 * np.sqrt(np.maximum(depth, 0.0))
        at_east = 0.5 * (at + np.roll(at, -1, axis=1))
        east_flux = at_east * (depth - np.roll(depth, -1, axis=1)) / 1000.0 * 1000.0
        west_flux_in = np.roll(east_flux, 1, axis=1)
        h, w = depth.shape
        at_face_y = np.zeros((h + 1, w))
        at_face_y[1:-1] = 0.5 * (at[:-1] + at[1:])
        face_flux = np.zeros((h + 1, w))
        face_flux[1:-1] = at_face_y[1:-1] * (depth[:-1] - depth[1:]) / 1000.0 * 1000.0
        mass = depth * 1e6 + dt_seconds * (west_flux_in - east_flux + face_flux[:-1] - face_flux[1:])
        return mass / 1e6

    buggy = buggy_single_step_no_cfl(k0, 1.0, a5["c5syn"])
    assert buggy.max() > 4.0 * k0.max()  # gross overshoot of the maximum principle
    assert buggy.min() < 0.0  # unphysical negative EKE


def test_eke_transport_step_conserves_total_energy_pure_transport():
    """Pure transport (no production/dissipation supplied) must conserve
    total K modulo the non-negativity floor, matching column_water.py's own
    conservation contract -- `eke_transport_step` literally reuses
    `evolve_column_water` for the advective term.
    """
    h, w = 16, 32
    area, xlen, ylen, dx, dy = _geometry(h, w)
    rng = np.random.default_rng(0)
    lat = ss._latitude_rad(h)
    k0 = np.full((h, w), 150.0) + 20.0 * rng.standard_normal((h, w)) ** 2
    u = 12.0 * np.cos(lat)[:, None] * np.ones((h, w))
    v = 3.0 * np.sin(2 * lat)[:, None] * np.ones((h, w))
    step = ss.eke_transport_step(
        k0, u, v, dx_m=dx, dy_m=dy, dt_days=0.5,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
    )
    total_before = float(np.sum(k0 * area))
    total_after = float(np.sum(step.eke_m2_s2 * area))
    # column_water.evolve_column_water stores its result as float32 (see its
    # own tests, e.g. test_prior_art_kernels.py's 1e-5-scale residual
    # tolerances), so the achievable precision here is float32-limited, not
    # float64 machine precision.
    assert total_after == pytest.approx(total_before, rel=1e-6)
    assert abs(step.relative_residual) < 1e-5
    assert np.isfinite(step.eke_m2_s2).all()
    assert (step.eke_m2_s2 >= 0.0).all()


def test_eke_diffusion_smooths_bump_and_preserves_centroid():
    """Physical-sanity check: diffusion alone smooths a bump (peak drops,
    the total conserves, and -- since diffusion has no directional bias on a
    periodic domain away from a boundary -- the bump's centroid does not
    drift).
    """
    h, w = 24, 48
    area, xlen, ylen, dx, dy = _geometry(h, w)
    lat = ss._latitude_rad(h)
    lon = np.linspace(-np.pi, np.pi, w, endpoint=False)
    j0 = int(np.argmin(np.abs(lat - np.radians(30.0))))
    i0 = w // 2
    k0 = np.full((h, w), 50.0)
    k0[j0, i0] = 4000.0

    res = ss.eke_diffusion_step(
        k0, dx_m=dx, dy_m=dy, dt_days=2.0,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
    )
    assert res.eke_m2_s2[j0, i0] < k0[j0, i0]
    assert res.eke_m2_s2[j0, i0 - 1] > k0[j0, i0 - 1]
    assert res.eke_m2_s2[j0, i0 + 1] > k0[j0, i0 + 1]
    assert res.residual_m2_s2 == pytest.approx(0.0, abs=1.0)

    def centroid_lon(field, row):
        weight = field[row] - float(field[row].min())
        ang = np.exp(1j * lon)
        return np.angle(np.sum(weight * ang) / np.sum(weight))

    dlon = centroid_lon(res.eke_m2_s2, j0) - centroid_lon(k0, j0)
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
    shift_m = abs(dlon) * _RADIUS_M * np.cos(lat[j0])
    assert shift_m < 0.5 * float(dx[j0, i0])  # sub-grid-cell centroid drift


def test_eke_advection_translates_bump_downstream_by_expected_distance():
    """Physical-sanity check: a uniform zonal wind should translate a K
    bump's centroid downstream by roughly ``u * dt`` -- diffusion (also
    active in `eke_transport_step`) smooths the bump but, being symmetric
    on a periodic domain away from a boundary, does not itself move the
    centroid, so the measured shift isolates the advective displacement.
    """
    h, w = 32, 64
    area, xlen, ylen, dx, dy = _geometry(h, w)
    lat = ss._latitude_rad(h)
    lon = np.linspace(-np.pi, np.pi, w, endpoint=False)
    j0 = int(np.argmin(np.abs(lat - np.radians(45.0))))
    i0 = w // 4

    k0 = np.full((h, w), 50.0)
    # Gaussian bump, wrapped across the periodic longitude boundary.
    offsets = (np.arange(w) - i0 + w // 2) % w - w // 2
    bump = 5000.0 * np.exp(-0.5 * (offsets / 1.2) ** 2)
    k0[j0, :] += bump

    u_speed = 20.0
    u = np.full((h, w), u_speed)
    v = np.zeros((h, w))
    dt_days = 0.25

    step = ss.eke_transport_step(
        k0, u, v, dx_m=dx, dy_m=dy, dt_days=dt_days,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
    )

    def centroid_lon(field, row):
        weight = field[row] - float(field[row].min())
        ang = np.exp(1j * lon)
        return np.angle(np.sum(weight * ang) / np.sum(weight))

    dlon = centroid_lon(step.eke_m2_s2, j0) - centroid_lon(k0, j0)
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
    actual_shift_m = dlon * _RADIUS_M * np.cos(lat[j0])
    expected_shift_m = u_speed * dt_days * 86400.0
    assert actual_shift_m == pytest.approx(expected_shift_m, rel=0.05)


def test_evolve_eke_full_step_shapes_finite_and_gate_off():
    h, w = 12, 24
    area, xlen, ylen, dx, dy = _geometry(h, w)
    k0 = np.full((h, w), 200.0)
    production = np.full((h, w), 2.0e-4)
    cd = np.full((h, w), 1.3e-3)
    u = np.full((h, w), 5.0)
    v = np.full((h, w), 1.0)
    step = ss.evolve_eke(
        k0, production, cd, u, v,
        dx_m=dx, dy_m=dy, dt_days=1.0,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
    )
    assert step.eke_m2_s2.shape == (h, w)
    assert np.isfinite(step.eke_m2_s2).all()
    assert (step.eke_m2_s2 >= 1.0).all()  # reference-implementation floor
    assert step.advection_substeps >= 1
    assert step.diffusion_substeps >= 1
    assert step.reaction_substeps >= 1
    # Stronger production -> higher equilibrium-ward EKE after the step.
    step_more = ss.evolve_eke(
        k0, production * 5.0, cd, u, v,
        dx_m=dx, dy_m=dy, dt_days=1.0,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
    )
    assert np.mean(step_more.eke_m2_s2) > np.mean(step.eke_m2_s2)
    assert PlanetParams().enable_sesam_dynamics is False


# ---------------------------------------------------------------------------
# (A52) implicit-zonal diffusion -- the 512x1024 polar-stiffness remedy
# (docs/SESAM_GAP_ANALYSIS.md P3, 2026-08-18 follow-up entry)
# ---------------------------------------------------------------------------


def test_cyclic_thomas_batch_matches_dense_solve():
    """`_cyclic_thomas_batch` against `numpy.linalg.solve` on the equivalent
    dense periodic matrix, arbitrary (non-constant-coefficient) rows -- the
    linear-algebra primitive `eke_diffusion_step_implicit_zonal`'s zonal
    half-step depends on, verified independently of the diffusion physics.
    """
    rng = np.random.default_rng(0)
    for h, w in [(1, 4), (3, 5), (5, 8), (2, 16)]:
        sub = rng.uniform(0.1, 2.0, (h, w))
        diag = rng.uniform(5.0, 10.0, (h, w))
        sup = rng.uniform(0.1, 2.0, (h, w))
        rhs = rng.standard_normal((h, w))
        x_batch = ss._cyclic_thomas_batch(sub, diag, sup, rhs)
        for r in range(h):
            m = np.zeros((w, w))
            for i in range(w):
                m[i, i] = diag[r, i]
                m[i, (i - 1) % w] += sub[r, i]
                m[i, (i + 1) % w] += sup[r, i]
            x_dense = np.linalg.solve(m, rhs[r])
            np.testing.assert_allclose(x_batch[r], x_dense, atol=1e-8, rtol=1e-8)


def test_eke_diffusion_implicit_zonal_matches_explicit_on_tractable_grid():
    """Small grid where the explicit scheme is itself tractable: the
    implicit-zonal path (a different time-discretisation of the identical
    continuous PDE) should land close to the explicit result after the same
    simulated time, and both must independently conserve total energy.
    """
    h, w = 24, 48
    area, xlen, ylen, dx, dy = _geometry(h, w)
    lat = ss._latitude_rad(h)
    j0 = int(np.argmin(np.abs(lat - np.radians(30.0))))
    i0 = w // 2
    k0 = np.full((h, w), 50.0)
    k0[j0, i0] = 4000.0

    explicit = ss.eke_diffusion_step(
        k0, dx_m=dx, dy_m=dy, dt_days=2.0, cell_area_m2=area,
        x_face_length_m=xlen, y_face_length_m=ylen,
    )
    implicit = ss.eke_diffusion_step_implicit_zonal(
        k0, dx_m=dx, dy_m=dy, dt_days=2.0, cell_area_m2=area,
        x_face_length_m=xlen, y_face_length_m=ylen,
    )
    assert explicit.residual_m2_s2 == pytest.approx(0.0, abs=1e-6)
    assert implicit.residual_m2_s2 == pytest.approx(0.0, abs=1e-6)
    # Same qualitative smoothing response (peak drops from the same start).
    assert implicit.eke_m2_s2[j0, i0] < k0[j0, i0]
    assert explicit.eke_m2_s2[j0, i0] < k0[j0, i0]
    # Within ~5% of each other -- close agreement expected since both solve
    # the same PDE, not identical since backward- vs forward-Euler differ.
    assert implicit.eke_m2_s2[j0, i0] == pytest.approx(explicit.eke_m2_s2[j0, i0], rel=0.05)
    for j, i in [(j0, i0 - 1), (j0, i0 + 1), (j0 - 1, i0), (j0 + 1, i0)]:
        assert implicit.eke_m2_s2[j, i] == pytest.approx(explicit.eke_m2_s2[j, i], rel=0.15)


def _pole_like_geometry(h: int, w: int, shrink: float = 1.0e-6):
    """Synthetic small grid reproducing the *exact mechanism* measured on
    the real 512x1024 grid (`docs/SESAM_GAP_ANALYSIS.md` P3, 2026-08-18): row
    0's cell area shrinks sharply (mimicking the true spherical polar-cap
    area) while the east/west face length stays the *same constant* as every
    other row (mimicking `spherical_transport_geometry`'s ``x_len =
    radius*dlat``, which does not shrink with latitude) -- this is what makes
    the zonal self-loss term ``x_len/(dx*area)`` diverge specifically at the
    pole. The north-facing face bordering row 0 is shrunk *proportionally*
    with area (matching the real geometry, where the interior polar face
    length and the cap area shrink together with dlat), which keeps the
    meridional term the same order at every row -- reproducing the real
    finding that only the zonal term is pole-stiff, not the meridional one.
    """
    area_normal = 1.0e10
    area = np.full((h, w), area_normal)
    area[0, :] = area_normal * shrink
    x_len = np.full((h, w), 50000.0)
    dx = np.full((h, w), 50000.0)
    y_len = np.full((h + 1, w), 50000.0)
    y_len[0, :] = 0.0
    y_len[1, :] = 50000.0 * shrink
    y_len[-1, :] = 0.0
    dy = 50000.0
    return area, x_len, dx, y_len, dy


def test_eke_diffusion_implicit_zonal_stable_where_explicit_is_infeasible():
    """The actual motivating case: reproduce the real grid's pole mechanism
    (see `_pole_like_geometry`) with a realistic near-pole EKE magnitude.
    The plain explicit `eke_diffusion_step` must fail to converge within its
    own substep cap (the same failure mode that made a real 512x1024 run
    impractical -- see the P3 2026-08-18 gap-analysis entry for the measured
    substep counts); `eke_diffusion_step_implicit_zonal` on the identical
    geometry/state must complete in a handful of substeps, stay finite, and
    respect the discrete maximum principle (pure diffusion cannot raise a
    field's peak above its own initial peak).
    """
    h, w = 6, 8
    area, x_len, dx, y_len, dy = _pole_like_geometry(h, w)
    k0 = np.full((h, w), 500.0)
    k0[0, :] = 6202.0  # matches the diagnosed real near-pole EKE magnitude

    with pytest.raises(RuntimeError):
        ss.eke_diffusion_step(
            k0, dx_m=dx, dy_m=dy, dt_days=0.25, cell_area_m2=area,
            x_face_length_m=x_len, y_face_length_m=y_len,
        )

    implicit = ss.eke_diffusion_step_implicit_zonal(
        k0, dx_m=dx, dy_m=dy, dt_days=0.25, cell_area_m2=area,
        x_face_length_m=x_len, y_face_length_m=y_len,
    )
    assert implicit.substeps < 500  # explicit would need > 200,000 on the same case
    assert np.isfinite(implicit.eke_m2_s2).all()
    assert implicit.eke_m2_s2.min() >= 0.0
    assert implicit.eke_m2_s2.max() <= k0.max() + 1e-6  # discrete maximum principle
    assert implicit.relative_residual == pytest.approx(0.0, abs=1e-9)


def test_eke_transport_step_implicit_zonal_flag_wired_and_conserves():
    """`implicit_zonal_diffusion=True` on `eke_transport_step`/`evolve_eke`
    actually routes through the new kernel (not silently ignored) and keeps
    the same conservation contract as the default explicit path.
    """
    h, w = 16, 32
    area, xlen, ylen, dx, dy = _geometry(h, w)
    rng = np.random.default_rng(1)
    lat = ss._latitude_rad(h)
    k0 = np.full((h, w), 150.0) + 20.0 * rng.standard_normal((h, w)) ** 2
    u = 12.0 * np.cos(lat)[:, None] * np.ones((h, w))
    v = 3.0 * np.sin(2 * lat)[:, None] * np.ones((h, w))
    step = ss.eke_transport_step(
        k0, u, v, dx_m=dx, dy_m=dy, dt_days=0.5,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
        implicit_zonal_diffusion=True,
    )
    total_before = float(np.sum(k0 * area))
    total_after = float(np.sum(step.eke_m2_s2 * area))
    assert total_after == pytest.approx(total_before, rel=1e-6)
    assert np.isfinite(step.eke_m2_s2).all()
    assert (step.eke_m2_s2 >= 0.0).all()

    production = np.full((h, w), 2.0e-4)
    cd = np.full((h, w), 1.3e-3)
    full = ss.evolve_eke(
        k0, production, cd, u, v,
        dx_m=dx, dy_m=dy, dt_days=0.5,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
        implicit_zonal_diffusion=True,
    )
    assert np.isfinite(full.eke_m2_s2).all()
    assert (full.eke_m2_s2 >= 1.0).all()
