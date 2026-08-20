"""Tests for SESAM stage P1 — diagnostic vertical structure (`sesam_vertical.py`).

The kernels are transcribed from Appendix A1 of Willeit et al. (2022)
(GMD 15, 5905-5948).  These tests guard the physics transcription the way the
reference-pack tests guard the constants: branch selection, the piecewise-lapse
integral, the ice/water saturation partition, the RH profile shape,
isothermal stratosphere, and the tropopause kernels.

The supported climate path never calls this module
(``PlanetParams.enable_sesam_vertical_structure`` is False), so these tests
exercise a default-off diagnostic only.
"""
from __future__ import annotations

import numpy as np
import pytest

import sesam_vertical as sv
from planet_params import PlanetParams

H, W = 3, 4
LEVELS = np.array([0.0, 500.0, 1500.0, 4000.0, 9000.0, 12000.0, 18000.0], dtype=float)
A1 = sv._a1_defaults()


def _baseline():
    ta = np.full((H, W), 290.0)
    ts = np.full((H, W), 283.0)
    qa = np.full((H, W), 0.010)
    kind = np.zeros((H, W), dtype=np.int64)
    zs = np.zeros((H, W))
    ht = np.full((H, W), 11000.0)
    return ta, ts, qa, kind, zs, ht


# ---------------------------------------------------------------------------
# (A7)-(A9) lapse-rate kernels
# ---------------------------------------------------------------------------


def test_free_troposphere_lapse_matches_hand_equations():
    qa = np.array([0.005, 0.010, 0.020])
    gb, gt = sv.free_troposphere_lapse(qa, a1=A1)
    c1, c2, c3 = A1["c1_Gamma"], A1["c2_Gamma"], A1["c3_Gamma"]
    expected_b = c1 - c2 * qa
    expected_t = expected_b - c2 * qa + c3
    np.testing.assert_allclose(gb, expected_b)
    np.testing.assert_allclose(gt, expected_t)


def test_near_surface_lapse_branches_and_caps():
    ta = np.array([[290.0, 290.0, 290.0, 280.0]])
    ts = np.array([[283.0, 283.0, 283.0, 285.0]])
    delta = ta - ts  # warm, warm, warm, cold (inversion)
    kind = np.array([[0, 1, 2, 1]])  # ocean, land, ice, cold land
    g = sv.near_surface_lapse(ta, ts, kind, a1=A1)
    # Un-capped values, then the published caps (7.5e-3 ocean, 10e-3 land/ice).
    assert g[0, 0] == pytest.approx(min(A1["c4_Gamma"] * 7.0, 7.5e-3))
    assert g[0, 1] == pytest.approx(min(A1["c5_Gamma"] * 7.0, 10.0e-3))
    assert g[0, 2] == pytest.approx(min(A1["c5_Gamma"] * 7.0, 10.0e-3))
    # Cold land (inversion) is negative and not lower-capped.
    assert g[0, 3] == pytest.approx(A1["c6_Gamma"] * (280.0 - 285.0))
    assert g[0, 3] < 0.0
    assert g.max() <= 10.0e-3 + 1e-9


def test_near_surface_lapse_large_cold_land_contrast_produces_unphysical_profile():
    """Documents a real finding from SESAM stage P6d (docs/SESAM_GAP_ANALYSIS.md
    Sec7, 2026-08-19): the (A9) cold-land branch's deliberately unbounded
    inversion term (see `test_near_surface_lapse_branches_and_caps` above --
    "not lower-capped" is intentional, paper-faithful behaviour, not a
    transcription gap) means a large, sustained Ta-T* contrast produces a
    near-surface lapse rate whose magnitude, integrated over even the ~1.5 km
    near-surface layer (`H_Gamma_s`), swings the profile hundreds of Kelvin
    away from the surface value. This was never exercised before P6d because
    every prior SESAM stage either used smooth synthetic test fields or a
    single diagnostic measurement against a saved state already near
    radiative equilibrium (small Ta-T* gap); P6d's live day-by-day coupling
    is the first caller that can let Ta drift far from the legacy T* between
    steps, and does so when the diabatic-source bridge/radiation feedback
    loop pushes a cell strongly out of balance (observed on a real 16x32
    smoke run: a 24 K Ta-T* gap alone produces a >600 K profile point).
    This is not asserted as a bug to fix here -- it is a live coupling
    constraint on how far Ta may safely diverge from T* before feeding this
    formula, deliberately left as an open finding for the P6 calibration
    window rather than silently patched (this project's own stop-condition
    discipline: a paper-faithful formula's own edge case is a finding about
    the *coupling*, not license to alter already-tested P1 physics without
    verifying the change against the source paper)."""
    ta = np.array([[261.75]])
    ts = np.array([[285.75]])  # a real 24 K gap observed during live P6d coupling
    kind = np.array([[1]])  # land
    qa = np.array([[0.0005]])
    zs = np.array([[0.0]])
    ht = np.array([[12000.0]])
    t_z, gamma_z, _ = sv.temperature_profile(LEVELS, ta, ts, kind, qa, zs, ht, a1=A1)
    assert np.any(t_z > 400.0), (
        "expected the documented unbounded-inversion blowup to reproduce; if this "
        "now fails, near_surface_lapse's cold-land branch behaviour has changed -- "
        "update docs/SESAM_GAP_ANALYSIS.md's P6d entry accordingly"
    )


# ---------------------------------------------------------------------------
# (A5) temperature profile
# ---------------------------------------------------------------------------


def test_temperature_profile_surface_and_isothermal_stratosphere():
    ta, ts, qa, kind, zs, ht = _baseline()
    t_z, g_z, t_surf = sv.temperature_profile(LEVELS, ta, ts, kind, qa, zs, ht, a1=A1)
    assert t_z.shape == (LEVELS.size, H, W)
    # Surface level (0 m) equals Ta.
    np.testing.assert_allclose(t_z[0], ta)
    # Monotonic non-increasing in the troposphere.
    for i in range(H):
        for j in range(W):
            assert np.all(np.diff(t_z[:, i, j]) <= 1e-6), (i, j)
    # Isothermal above the 11 km tropopause.
    above = t_z[LEVELS > 11000.0]
    for idx in range(1, above.shape[0]):
        np.testing.assert_allclose(above[idx], above[0], atol=1e-6)


def test_temperature_profile_analytic_piecewise_form():
    """One warm-ocean column: compare to the hand-expanded (A6) integral."""
    ta_v = np.array([[288.0]])
    ts_v = np.array([[283.0]])
    kind_v = np.array([[0]])
    qa_v = np.array([[0.010]])
    zs_v = np.zeros((1, 1))
    ht_v = np.array([[12000.0]])
    levels = np.array([0.0, 500.0, 1500.0, 6000.0, 12000.0, 15000.0])

    t_z, _, _ = sv.temperature_profile(
        levels, ta_v, ts_v, kind_v, qa_v, zs_v, ht_v, a1=A1
    )

    hs, ht_r = A1["H_Gamma_s"], A1["H_Gamma_t"]
    gs = min(A1["c4_Gamma"] * max(0.0, 288.0 - 283.0), 7.5e-3)  # ocean cap (A9)
    gb = A1["c1_Gamma"] - A1["c2_Gamma"] * 0.010
    gt = gb - A1["c2_Gamma"] * 0.010 + A1["c3_Gamma"]
    slope = (gt - gb) / ht_r
    b1 = hs

    def drop(z):
        if z <= b1:
            return gs * z
        if z <= 12000.0:
            return gs * b1 + gb * (z - b1) + 0.5 * slope * (z ** 2 - b1 ** 2)
        return gs * b1 + gb * (12000.0 - b1) + 0.5 * slope * (12000.0 ** 2 - b1 ** 2)

    expected = np.array([ta_v[0, 0] - drop(z) for z in levels])
    np.testing.assert_allclose(t_z[:, 0, 0], expected, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# (A13)/(A14)/(A15) humidity profile and saturation partition
# ---------------------------------------------------------------------------


def test_relative_humidity_profile_piecewise_boundaries():
    zs = np.zeros((H, W))
    ht = np.full((H, W), 11000.0)
    hr = np.full((H, W), A1["c1r"] * A1["c3r"])  # extratropical Hr = 6000 m
    ra = np.full((H, W), 0.7)
    levels = np.array([0.0, 3000.0, 6000.0, 12000.0])
    gr = sv._level_grid(levels, H, W)
    r, _ = sv.relative_humidity_profile(gr, ra, zs, ht, hr, a1=A1)

    zpbl = A1["c5r"]
    z_c4 = A1["c4r"]
    # PBL level below 1000 m -> ra
    assert r[0, 0, 0] == pytest.approx(0.7)
    # 3000 m = z_c4 top -> exponential decay from the PBL top
    expected = 0.7 * np.exp(-(z_c4 - zpbl) / hr[0, 0])
    assert r[1, 0, 0] == pytest.approx(expected, rel=1e-9)
    # 6000 m in the constant "low troposphere" tail (still < HT) -> same value
    assert r[2, 0, 0] == pytest.approx(expected, rel=1e-9)
    # 12000 m stratosphere -> r_st
    assert r[3, 0, 0] == pytest.approx(A1["r_st"])


def test_rh_scale_height_tropical_vs_extratropical():
    f_eq = sv.tropical_weight(np.array([0.0]))
    h_eq = sv.rh_scale_height(f_eq, 0.0, a1=A1)
    assert h_eq[0] == pytest.approx(A1["c1r"])
    f_pol = sv.tropical_weight(np.array([np.pi / 2.0]))
    h_pol = sv.rh_scale_height(f_pol, 0.0, a1=A1)
    assert h_pol[0] == pytest.approx(A1["c1r"] * A1["c3r"])


def test_saturation_partition_ice_water_and_mid_range():
    p = np.array([100000.0, 100000.0, 100000.0])
    t = np.array([-30.0, -5.0, 20.0]) + 273.15
    q = sv.saturation_specific_humidity(t, p)
    eps = 0.622

    def qsat_from_es(es, pval):
        return eps * es / (pval - (1.0 - eps) * es)

    # Warm case: exact Magnus water curve.
    es_w = 611.2 * np.exp(17.67 * 20.0 / (20.0 + 243.5))
    assert q[2] == pytest.approx(qsat_from_es(es_w, p[2]), rel=1e-9)

    # Cold case: over-ice curve (lower es).
    es_i = 611.2 * np.exp(22.46 * (-30.0) / (-30.0 + 272.62))
    assert q[0] == pytest.approx(qsat_from_es(es_i, p[0]), rel=1e-9)
    es_w_c = 611.2 * np.exp(17.67 * (-30.0) / (-30.0 + 243.5))
    assert es_i < es_w_c

    # Mid case is a linear blend of the two saturation values.
    es_w_m = 611.2 * np.exp(17.67 * (-5.0) / (-5.0 + 243.5))
    es_i_m = 611.2 * np.exp(22.46 * (-5.0) / (-5.0 + 272.62))
    w = (-5.0 + 15.0) / 15.0  # (T - T_ice) / (T_0 - T_ice) at -5 C
    exp_mid = w * qsat_from_es(es_w_m, p[1]) + (1.0 - w) * qsat_from_es(es_i_m, p[1])
    assert q[1] == pytest.approx(exp_mid, rel=1e-9)


# ---------------------------------------------------------------------------
# Tropopause kernels
# ---------------------------------------------------------------------------


def test_tropopause_shape_peaks_in_hadley_region():
    lat = np.linspace(-np.pi / 2, np.pi / 2, 180)
    s = sv.tropopause_shape_s(lat, 0.1, 0.5)
    assert np.isfinite(s).all()
    peak_lat = lat[int(np.argmax(s))]
    assert -0.7 < peak_lat < 0.7


def test_tropopause_tendency_negative_for_net_downward_shape():
    s = np.full((H, W), 10.0)
    r = np.zeros((H, W))
    dt = sv.tropopause_tendency(r, s)
    assert dt.shape == (H, W)
    assert (dt < 0).all()


# ---------------------------------------------------------------------------
# (A12) potential temperature
# ---------------------------------------------------------------------------


def test_potential_temperature_linear_definition():
    t = np.full((2, H, W), 250.0)
    th = sv.potential_temperature_profile(t, [0.0, 100.0], gravity=9.81, specific_heat=1005.0)
    assert th[1, 0, 0] == pytest.approx(th[0, 0, 0] + 9.81 / 1005.0 * 100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Assembly and gating
# ---------------------------------------------------------------------------


def test_compute_vertical_structure_shapes_and_finite():
    ta, ts, qa, kind, zs, ht = _baseline()
    res = sv.compute_vertical_structure(
        LEVELS,
        near_surface_air_temp_k=ta,
        skin_temp_k=ts,
        surface_kind=kind,
        near_surface_specific_humidity_kgkg=qa,
        surface_elevation_m=zs,
        tropopause_height_m=ht,
        p0_pa=101325.0,
        gravity=9.81,
        reference_temp_k=288.0,
        itcz_latitude_rad=0.1,
        hadley_width_rad=0.5,
    )
    assert res.temperature_k.shape == (LEVELS.size, H, W)
    assert res.relative_humidity.shape == (LEVELS.size, H, W)
    assert res.specific_humidity_kgkg.shape == (LEVELS.size, H, W)
    assert res.pressure_pa.shape == (LEVELS.size, H, W)
    assert res.tropopause_shape_s is not None
    assert res.tropopause_shape_s.shape == (H, W)
    assert res.tropopause_tendency is None  # R_strat_net not supplied
    for arr in (
        res.temperature_k,
        res.lapse_rate_k_per_m,
        res.relative_humidity,
        res.specific_humidity_kgkg,
        res.potential_temperature_k,
        res.pressure_pa,
    ):
        assert np.isfinite(arr).all()
    assert res.relative_humidity.min() >= 0.0
    assert res.relative_humidity.max() <= 1.0 + 1e-9
    assert res.pressure_pa.min() > 0.0


def test_compute_with_tropopause_tendency_when_r_strat_given():
    ta, ts, qa, kind, zs, ht = _baseline()
    res = sv.compute_vertical_structure(
        LEVELS,
        near_surface_air_temp_k=ta,
        skin_temp_k=ts,
        surface_kind=kind,
        near_surface_specific_humidity_kgkg=qa,
        surface_elevation_m=zs,
        tropopause_height_m=ht,
        p0_pa=101325.0,
        gravity=9.81,
        reference_temp_k=288.0,
        itcz_latitude_rad=0.1,
        hadley_width_rad=0.5,
        r_strat_net_w_m2=np.zeros((H, W)),
    )
    assert res.tropopause_tendency is not None
    assert res.tropopause_tendency.shape == (H, W)


def test_surface_kind_validation_rejects_unknown_code():
    with pytest.raises(ValueError):
        sv.near_surface_lapse(np.zeros((H, W)), np.zeros((H, W)), np.full((H, W), 5))


def test_continuous_fields_require_matching_shapes():
    with pytest.raises(ValueError):
        sv.temperature_profile(
            LEVELS,
            np.zeros((H, W)),
            np.zeros((H + 1, W)),
            np.zeros((H, W), dtype=np.int64),
            np.zeros((H, W)),
            np.zeros((H, W)),
            np.full((H, W), 11000.0),
        )


def test_sesam_vertical_gate_defaults_off():
    assert PlanetParams().enable_sesam_vertical_structure is False