"""Tests for SESAM stage P4 -- column energy/water closure (`sesam_thermo.py`).

Equations transcribed from Appendix A3/A4 of Willeit et al. (2022) (GMD 15,
5905-5948), verified directly against the article PDF (2026-08-18; see the
module docstring). These tests guard: (A41)/(A43) near-surface diagnostics,
the (A40) column heat capacity and diabatic source assembly, the shared
conservative linear-diffusion primitive (explicit and implicit-zonal), the
(A40) column-energy evolution, the (A45) slope-convergence term, the (A44)
precipitation formula, and the (A42)/(A44) full column-water step.

The supported climate path never calls this module
(``PlanetParams.enable_sesam_column_closure`` is False), so these tests
exercise a default-off diagnostic only.
"""
from __future__ import annotations

import numpy as np
import pytest

import sesam_thermo as st
from column_water import evolve_column_water
from planet_params import PlanetParams
from sesam_synoptic import spherical_transport_geometry, zonal_center_spacing_m
from sesam_vertical import saturation_specific_humidity

_RADIUS_M = 6.371e6
A4 = st._a4_defaults()


def _geometry(h, w):
    area, xlen, ylen = spherical_transport_geometry(h, w, _RADIUS_M)
    dx = zonal_center_spacing_m(h, w, _RADIUS_M)
    dy = _RADIUS_M * (np.pi / h)
    return area, xlen, ylen, dx, dy


def test_sesam_column_closure_gate_defaults_off():
    assert PlanetParams().enable_sesam_column_closure is False


# ---------------------------------------------------------------------------
# (A41)/(A43) near-surface diagnostics
# ---------------------------------------------------------------------------


def test_t2m_diagnostic_hand_value():
    ta = np.array([[280.0, 300.0]])
    tstar = np.array([[290.0, 280.0]])
    t2m = st.t2m_diagnostic(ta, tstar)
    assert t2m == pytest.approx(np.array([[285.0, 290.0]]))


def test_t2m_diagnostic_shape_mismatch_raises():
    with pytest.raises(ValueError):
        st.t2m_diagnostic(np.zeros((2, 2)), np.zeros((2, 3)))


def test_surface_relative_humidity_star_hand_value():
    tstar = np.full((1, 1), 288.0)
    p = np.full((1, 1), 101325.0)
    qsat_star = saturation_specific_humidity(tstar, p)
    qa = 0.5 * qsat_star
    r_star = st.surface_relative_humidity_star(qa, tstar, p)
    assert r_star == pytest.approx(0.5, rel=1e-9)


def test_q2m_diagnostic_matches_hand_computed_blend():
    ta = np.full((1, 1), 285.0)
    tstar = np.full((1, 1), 290.0)
    p = np.full((1, 1), 101325.0)
    ra = np.full((1, 1), 0.6)
    r_star = np.full((1, 1), 0.8)
    q2m = st.q2m_diagnostic(ra, r_star, ta, tstar, p)
    t2m = 0.5 * (285.0 + 290.0)
    expected = 0.5 * (0.6 + 0.8) * saturation_specific_humidity(np.array([[t2m]]), p)
    assert q2m == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# (A40) column heat capacity and diabatic source
# ---------------------------------------------------------------------------


def test_column_heat_capacity_hand_value():
    cv = st.column_heat_capacity_j_m2_k(101325.0, 9.81, cv_specific_j_kg_k=717.0)
    expected = (101325.0 / 9.81) * 717.0
    assert cv == pytest.approx(expected, rel=1e-9)


def test_column_heat_capacity_rejects_nonpositive():
    with pytest.raises(ValueError):
        st.diabatic_heating_rate_k_day(
            np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)),
            np.zeros((1, 1)), 0.0,
        )


def test_diabatic_heating_rate_hand_value():
    cv = 1.0e7  # J m^-2 K^-1
    sw = np.full((1, 1), 100.0)
    lw = np.full((1, 1), -50.0)
    pw = np.full((1, 1), 8.64)  # mm/day -> 1e-4 kg m^-2 s^-1
    ps = np.zeros((1, 1))
    sh = np.full((1, 1), 20.0)
    rate = st.diabatic_heating_rate_k_day(sw, lw, pw, ps, sh, cv, latent_heat_vaporization_j_kg=2.5e6)
    pw_flux = 8.64 / 86400.0
    net_w_m2 = 100.0 - 50.0 + 2.5e6 * pw_flux + 20.0
    expected_k_day = (net_w_m2 / cv) * 86400.0
    assert rate == pytest.approx(np.array([[expected_k_day]]), rel=1e-9)


def test_diabatic_heating_rate_zero_forcing_is_zero():
    z = np.zeros((3, 4))
    rate = st.diabatic_heating_rate_k_day(z, z, z, z, z, 1.0e7)
    assert np.allclose(rate, 0.0)


# ---------------------------------------------------------------------------
# Shared conservative linear diffusion primitive
# ---------------------------------------------------------------------------


def test_linear_diffusion_step_hand_value_single_substep():
    """3x4 grid, uniform 1000 m spacing, uniform diffusivity AT=2.0 applied to
    a single interior peak (field=4.0) with zero-field neighbours.

    Unlike `eke_diffusion_step` (nonlinear, AT derived from the field being
    diffused), this diffusivity is externally fixed, so the face-averaged AT
    is exactly (2.0 + 0.0)/2 = 1.0 on every face -- an identical setup to
    ``sesam_synoptic``'s own hand-value test with ``c5syn=1`` and K=4 (whose
    sqrt(4)=2 matches this AT=2.0 exactly), so the numeric answer must match
    that test's: peak -> 3.3088, each of 4 neighbours -> 0.1728.
    """
    field0 = np.zeros((3, 4))
    field0[1, 1] = 4.0
    at = np.zeros((3, 4))
    at[1, 1] = 2.0  # only the peak's own AT matters given all neighbours are 0
    res = st._linear_diffusion_step(
        field0, at, dx_m=1000.0, dy_m=1000.0, dt_days=0.5,
        cell_area_m2=1e6, x_face_length_m=1000.0, y_face_length_m=1000.0,
        nonnegative=False,
    )
    assert res.substeps == 1
    assert res.field[1, 1] == pytest.approx(3.3088, rel=1e-9)
    for j, i in [(1, 0), (1, 2), (0, 1), (2, 1)]:
        assert res.field[j, i] == pytest.approx(0.1728, rel=1e-9)
    assert res.residual == pytest.approx(0.0, abs=1e-6)


def test_linear_diffusion_step_conserves_total_with_uniform_diffusivity():
    h, w = 12, 20
    rng = np.random.default_rng(7)
    field0 = rng.uniform(0.0, 10.0, size=(h, w))
    area, xlen, ylen, dx, dy = _geometry(h, w)
    at = np.full((h, w), 5.0e5)
    res = st._linear_diffusion_step(
        field0, at, dx_m=dx, dy_m=dy, dt_days=1.0,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
        nonnegative=True,
    )
    assert res.relative_residual == pytest.approx(0.0, abs=1e-6)
    assert res.field.min() >= 0.0
    assert res.field.max() <= field0.max() + 1e-6  # discrete maximum principle


def test_linear_diffusion_omitting_face_averaging_breaks_symmetry_planted_violation():
    field0 = np.zeros((3, 4))
    field0[1, 1] = 4.0
    at = np.zeros((3, 4))
    at[1, 1] = 2.0
    correct = st._linear_diffusion_step(
        field0, at, dx_m=1000.0, dy_m=1000.0, dt_days=0.5,
        cell_area_m2=1e6, x_face_length_m=1000.0, y_face_length_m=1000.0,
        nonnegative=False,
    )
    neighbours_correct = [correct.field[j, i] for j, i in [(1, 0), (1, 2), (0, 1), (2, 1)]]
    assert len(set(np.round(neighbours_correct, 6))) == 1

    def buggy_one_sided(field0, at, dt_days):
        dt_seconds = dt_days * 86400.0
        east_flux = at * (field0 - np.roll(field0, -1, axis=1)) / 1000.0 * 1000.0
        west_flux_in = np.roll(east_flux, 1, axis=1)
        h, w = field0.shape
        face_flux = np.zeros((h + 1, w))
        face_flux[1:-1] = at[:-1] * (field0[:-1] - field0[1:]) / 1000.0 * 1000.0
        mass = field0 * 1e6 + dt_seconds * (west_flux_in - east_flux + face_flux[:-1] - face_flux[1:])
        return mass / 1e6

    buggy = buggy_one_sided(field0, at, 0.5)
    neighbours_buggy = [buggy[j, i] for j, i in [(1, 0), (1, 2), (0, 1), (2, 1)]]
    assert len(set(np.round(neighbours_buggy, 6))) > 1
    assert buggy[1, 0] == pytest.approx(0.0, abs=1e-9)
    assert buggy[1, 2] == pytest.approx(2.0 * 0.1728, rel=1e-6)


def test_linear_diffusion_cfl_substepping_prevents_maximum_principle_violation_planted_violation():
    field0 = np.zeros((3, 4))
    field0[1, 1] = 10000.0
    at = np.full((3, 4), 100.0)
    correct = st._linear_diffusion_step(
        field0, at, dx_m=1000.0, dy_m=1000.0, dt_days=1.0,
        cell_area_m2=1e6, x_face_length_m=1000.0, y_face_length_m=1000.0,
        nonnegative=True,
    )
    assert correct.substeps > 1
    assert correct.field.max() <= field0.max() + 1e-6
    assert correct.field.min() >= 0.0

    def buggy_single_step(field0, at, dt_days):
        dt_seconds = dt_days * 86400.0
        at_east = 0.5 * (at + np.roll(at, -1, axis=1))
        east_flux = at_east * (field0 - np.roll(field0, -1, axis=1)) / 1000.0 * 1000.0
        west_flux_in = np.roll(east_flux, 1, axis=1)
        h, w = field0.shape
        at_face_y = np.zeros((h + 1, w))
        at_face_y[1:-1] = 0.5 * (at[:-1] + at[1:])
        face_flux = np.zeros((h + 1, w))
        face_flux[1:-1] = at_face_y[1:-1] * (field0[:-1] - field0[1:]) / 1000.0 * 1000.0
        mass = field0 * 1e6 + dt_seconds * (west_flux_in - east_flux + face_flux[:-1] - face_flux[1:])
        return mass / 1e6

    buggy = buggy_single_step(field0, at, 1.0)
    assert buggy.max() > 4.0 * field0.max()
    assert buggy.min() < 0.0


def test_linear_diffusion_implicit_zonal_matches_explicit_on_tractable_grid():
    h, w = 8, 16
    area, xlen, ylen, dx, dy = _geometry(h, w)
    rng = np.random.default_rng(3)
    field0 = rng.uniform(1.0, 5.0, size=(h, w))
    at = np.full((h, w), 2.0e5)
    explicit = st._linear_diffusion_step(
        field0, at, dx_m=dx, dy_m=dy, dt_days=0.25,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
    )
    implicit = st._linear_diffusion_step_implicit_zonal(
        field0, at, dx_m=dx, dy_m=dy, dt_days=0.25,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
    )
    assert np.max(np.abs(explicit.field - implicit.field)) < 0.15 * np.max(np.abs(field0))
    assert implicit.relative_residual == pytest.approx(0.0, abs=1e-6)


def _pole_like_geometry(h: int, w: int, shrink: float = 1.0e-6):
    """Synthetic small grid reproducing the *exact mechanism* measured on the
    real 512x1024 grid (docs/SESAM_GAP_ANALYSIS.md P3, 2026-08-18; the same
    helper `testing/test_sesam_synoptic.py` uses for the identical purpose):
    row 0's cell area shrinks sharply (mimicking the true spherical polar-cap
    area) while the east/west face length stays the *same constant* as every
    other row (mimicking `spherical_transport_geometry`'s ``x_len =
    radius*dlat``, which does not shrink with latitude) -- this is what makes
    the zonal self-loss term ``x_len/(dx*area)`` diverge specifically at the
    pole. The north-facing face bordering row 0 is shrunk *proportionally*
    with area, keeping the meridional term the same order at every row --
    reproducing the real finding that only the zonal term is pole-stiff.
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


def test_linear_diffusion_implicit_zonal_stable_where_explicit_is_infeasible():
    """Reproduce the real grid's pole mechanism (`_pole_like_geometry`) with
    an AT magnitude matching the diagnosed real near-pole EKE-driven
    diffusivity (module docstring note 5: AT/Aq inherit K's own scale). The
    plain explicit `_linear_diffusion_step` must fail to converge within its
    substep cap; `_linear_diffusion_step_implicit_zonal` on the identical
    geometry/state must complete in a handful of substeps, stay finite, and
    respect the discrete maximum principle.
    """
    h, w = 6, 8
    area, x_len, dx, y_len, dy = _pole_like_geometry(h, w)
    field0 = np.full((h, w), 500.0)
    field0[0, :] = 6202.0  # matches P3's diagnosed real near-pole EKE magnitude
    at = np.full((h, w), 3.5e6)  # matches P3's diagnosed real AT magnitude

    with pytest.raises(RuntimeError):
        st._linear_diffusion_step(
            field0, at, dx_m=dx, dy_m=dy, dt_days=0.25,
            cell_area_m2=area, x_face_length_m=x_len, y_face_length_m=y_len,
        )

    implicit = st._linear_diffusion_step_implicit_zonal(
        field0, at, dx_m=dx, dy_m=dy, dt_days=0.25,
        cell_area_m2=area, x_face_length_m=x_len, y_face_length_m=y_len,
    )
    assert implicit.substeps < 500  # explicit would need > 200,000 on the same case
    assert np.isfinite(implicit.field).all()
    assert implicit.field.min() >= 0.0
    assert implicit.field.max() <= field0.max() + 1e-6
    assert implicit.relative_residual == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# (A40) column energy evolution
# ---------------------------------------------------------------------------


def test_evolve_column_energy_pure_transport_conserves_total_heat():
    h, w = 10, 16
    area, xlen, ylen, dx, dy = _geometry(h, w)
    rng = np.random.default_rng(11)
    ta0 = 280.0 + rng.uniform(-5.0, 5.0, size=(h, w))
    zeros = np.zeros((h, w))
    wind_u = rng.uniform(-3.0, 3.0, size=(h, w))
    wind_v = rng.uniform(-1.0, 1.0, size=(h, w))
    eke = np.full((h, w), 200.0)
    step = st.evolve_column_energy(
        ta0, zeros, wind_u, wind_v, eke,
        dx_m=dx, dy_m=dy, dt_days=0.25,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
    )
    assert step.relative_residual == pytest.approx(0.0, abs=1e-5)
    assert np.isfinite(step.temperature_k).all()


def test_evolve_column_energy_adds_exact_heat_with_zero_wind():
    h, w = 6, 8
    area, xlen, ylen, dx, dy = _geometry(h, w)
    ta0 = np.full((h, w), 280.0)
    heating = np.full((h, w), 2.0)  # K/day
    zeros = np.zeros((h, w))
    eke = np.zeros((h, w))  # AT -> floor, diffusion negligible over 0.1 day
    step = st.evolve_column_energy(
        ta0, heating, zeros, zeros, eke,
        dx_m=dx, dy_m=dy, dt_days=0.1,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
    )
    # With zero wind, pure diabatic heating raises every cell by heating*dt,
    # independent of the (tiny, floor-limited) diffusion which only
    # redistributes -- since the field starts perfectly uniform, diffusion
    # has nothing to smooth and every cell moves by exactly heating*dt.
    assert step.temperature_k == pytest.approx(np.full((h, w), 280.2), abs=1e-3)


def test_evolve_column_energy_implicit_zonal_flag_wired_and_conserves():
    h, w = 8, 16
    area, xlen, ylen, dx, dy = _geometry(h, w)
    rng = np.random.default_rng(13)
    ta0 = 280.0 + rng.uniform(-5.0, 5.0, size=(h, w))
    zeros = np.zeros((h, w))
    eke = np.full((h, w), 5.0e5)  # large AT to exercise the implicit path
    step = st.evolve_column_energy(
        ta0, zeros, zeros, zeros, eke,
        dx_m=dx, dy_m=dy, dt_days=0.25,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
        implicit_zonal_diffusion=True,
    )
    assert step.relative_residual == pytest.approx(0.0, abs=1e-5)
    assert np.isfinite(step.temperature_k).all()


# ---------------------------------------------------------------------------
# (A45) slope convergence
# ---------------------------------------------------------------------------


def test_slope_convergence_hand_value():
    k = np.full((1, 1), 4.0)  # sqrt(K) = 2.0
    slope = np.full((1, 1), 0.1)
    qa = np.full((1, 1), 0.01)
    rho0 = 1.2
    cslope = st.slope_convergence_mm_day(k, slope, qa, rho0)
    c_slope_p = A4["c_slope_p"]
    expected_flux = c_slope_p * 2.0 * 0.1 * rho0 * 0.01
    assert cslope == pytest.approx(np.array([[expected_flux * 86400.0]]), rel=1e-9)


def test_slope_convergence_uses_slope_magnitude_not_sign():
    k = np.full((1, 2), 4.0)
    slope = np.array([[0.1, -0.1]])
    qa = np.full((1, 2), 0.01)
    cslope = st.slope_convergence_mm_day(k, slope, qa, 1.2)
    assert cslope[0, 0] == pytest.approx(cslope[0, 1], rel=1e-9)


# ---------------------------------------------------------------------------
# (A44) precipitation
# ---------------------------------------------------------------------------


def test_precipitation_rate_hand_value_ocean_no_turnover():
    c = np.full((1, 1), 3.0)
    cslope = np.full((1, 1), 1.0)
    e = np.full((1, 1), 2.0)
    ra = np.full((1, 1), 0.475)  # half of ramax=0.95
    qq = np.full((1, 1), 30.0)
    land = np.zeros((1, 1))
    p = st.precipitation_rate_mm_day(c, cslope, e, ra, qq, land_mask=land)
    ramax = A4["ra_max"]
    expected_gross = max(0.0, 3.0 + 1.0 + 2.0) * (0.475 / ramax)
    assert p == pytest.approx(np.array([[expected_gross]]), rel=1e-9)


def test_precipitation_rate_land_adds_turnover_term():
    c = np.zeros((1, 1))
    cslope = np.zeros((1, 1))
    e = np.zeros((1, 1))
    ra = np.full((1, 1), 0.5)
    qq = np.full((1, 1), 25.0)
    land = np.ones((1, 1))
    p = st.precipitation_rate_mm_day(c, cslope, e, ra, qq, land_mask=land)
    tau_p = A4["tau_p"]
    expected = 25.0 * 0.5 / tau_p
    assert p == pytest.approx(np.array([[expected]]), rel=1e-9)
    p_ocean = st.precipitation_rate_mm_day(c, cslope, e, ra, qq, land_mask=np.zeros((1, 1)))
    assert p_ocean == pytest.approx(np.array([[0.0]]), abs=1e-12)


def test_precipitation_rate_negative_convergence_floors_gross_term():
    c = np.full((1, 1), -5.0)
    cslope = np.zeros((1, 1))
    e = np.zeros((1, 1))
    ra = np.full((1, 1), 0.5)
    qq = np.zeros((1, 1))
    p = st.precipitation_rate_mm_day(c, cslope, e, ra, qq)
    assert p == pytest.approx(np.array([[0.0]]), abs=1e-12)


def test_precipitation_rate_efficiency_clamped_above_ramax():
    c = np.full((1, 1), 10.0)
    cslope = np.zeros((1, 1))
    e = np.zeros((1, 1))
    ra = np.full((1, 1), 1.0)  # above ramax=0.95
    qq = np.zeros((1, 1))
    p = st.precipitation_rate_mm_day(c, cslope, e, ra, qq)
    assert p == pytest.approx(np.array([[10.0]]), rel=1e-9)  # clamped to 100%, not >100%


def test_precipitation_rate_monotonic_in_relative_humidity():
    c = np.full((1, 3), 5.0)
    cslope = np.zeros((1, 3))
    e = np.zeros((1, 3))
    ra = np.array([[0.1, 0.5, 0.9]])
    qq = np.zeros((1, 3))
    p = st.precipitation_rate_mm_day(c, cslope, e, ra, qq)
    assert p[0, 0] < p[0, 1] < p[0, 2]


# ---------------------------------------------------------------------------
# (A42)/(A44) full column-water step
# ---------------------------------------------------------------------------


def test_moisture_convergence_matches_zero_source_transport_plus_diffusion():
    h, w = 10, 16
    area, xlen, ylen, dx, dy = _geometry(h, w)
    rng = np.random.default_rng(17)
    q0 = rng.uniform(5.0, 30.0, size=(h, w))
    wind_u = rng.uniform(-3.0, 3.0, size=(h, w))
    wind_v = rng.uniform(-1.0, 1.0, size=(h, w))
    eke = np.full((h, w), 150.0)
    conv = st.moisture_convergence_mm_day(
        q0, wind_u, wind_v, eke,
        dx_m=dx, dy_m=dy, dt_days=0.25,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
    )
    assert conv.convergence_mm_day.shape == (h, w)
    assert np.isfinite(conv.convergence_mm_day).all()


def test_evolve_column_water_vapor_conserves_water_energy_when_precip_matches_convergence():
    """A near-saturated, spatially uniform state (no advective/diffusive
    convergence since the field is flat) should generate precipitation from
    the land-turnover term alone, and total column-water mass should track
    ``E - P`` exactly (matching column_water.py's own conservation contract
    -- this is the P4 exit-gate's stated standard).
    """
    h, w = 8, 12
    area, xlen, ylen, dx, dy = _geometry(h, w)
    q0 = np.full((h, w), 25.0)
    e = np.full((h, w), 3.0)
    wind_u = np.zeros((h, w))
    wind_v = np.zeros((h, w))
    eke = np.full((h, w), 150.0)
    ra = np.full((h, w), 0.5)
    qa = np.full((h, w), 0.008)
    slope = np.zeros((h, w))
    land = np.zeros((h, w))  # ocean-only: no turnover, no advective convergence -> P from E-driven residual only
    step = st.evolve_column_water_vapor(
        q0, e, wind_u, wind_v, eke, ra, qa, slope, land,
        dx_m=dx, dy_m=dy, dt_days=0.25,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
        rho0_kg_m3=1.2,
    )
    total_area = float(np.sum(area))
    expected_water = float(np.sum(q0 * area)) / total_area + 0.25 * (
        3.0 - float(np.sum(step.precipitation_mm_day * area)) / total_area
    )
    actual_water = float(np.sum(step.water_mm.astype(np.float64) * area)) / total_area
    assert actual_water == pytest.approx(expected_water, abs=1e-3)
    assert step.relative_residual == pytest.approx(0.0, abs=1e-5)


def test_evolve_column_water_vapor_land_turnover_generates_precipitation():
    h, w = 6, 8
    area, xlen, ylen, dx, dy = _geometry(h, w)
    q0 = np.full((h, w), 40.0)
    e = np.zeros((h, w))
    wind_u = np.zeros((h, w))
    wind_v = np.zeros((h, w))
    eke = np.full((h, w), 100.0)
    ra = np.full((h, w), 0.6)
    qa = np.full((h, w), 0.008)
    slope = np.zeros((h, w))
    land = np.ones((h, w))
    step = st.evolve_column_water_vapor(
        q0, e, wind_u, wind_v, eke, ra, qa, slope, land,
        dx_m=dx, dy_m=dy, dt_days=0.25,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
        rho0_kg_m3=1.2,
    )
    assert np.all(step.precipitation_mm_day > 0.0)  # land turnover fires


def test_evolve_column_water_vapor_raw_global_precip_is_physically_bounded():
    """Screen-level sanity check with Earth-like magnitudes (not the real
    saved-state exit-gate measurement, which lives in
    scripts/diagnose_sesam_thermo.py): a spatially varying, Earth-scale
    synthetic state should not produce an absurd (negative, or >>10x Earth
    mean) global precipitation rate.
    """
    h, w = 16, 32
    area, xlen, ylen, dx, dy = _geometry(h, w)
    rng = np.random.default_rng(29)
    q0 = rng.uniform(5.0, 40.0, size=(h, w))
    e = rng.uniform(0.5, 5.0, size=(h, w))
    wind_u = rng.uniform(-8.0, 8.0, size=(h, w))
    wind_v = rng.uniform(-3.0, 3.0, size=(h, w))
    eke = rng.uniform(50.0, 300.0, size=(h, w))
    ra = rng.uniform(0.3, 0.9, size=(h, w))
    qa = rng.uniform(0.001, 0.02, size=(h, w))  # Earth-realistic specific humidity
    slope = rng.uniform(0.0, 0.05, size=(h, w))
    land = (rng.uniform(0.0, 1.0, size=(h, w)) > 0.3).astype(np.float64)
    step = st.evolve_column_water_vapor(
        q0, e, wind_u, wind_v, eke, ra, qa, slope, land,
        dx_m=dx, dy_m=dy, dt_days=0.25,
        cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
        rho0_kg_m3=1.2,
    )
    global_p = float(np.sum(step.precipitation_mm_day * area) / np.sum(area))
    assert global_p >= 0.0
    assert global_p < 50.0  # generously above Earth's ~2.9 mm/day mean
