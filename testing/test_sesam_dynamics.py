"""Tests for SESAM stage P2 — SLP reconstruction (`sesam_dynamics.py`).

The kernels are transcribed from Appendix A2 of Willeit et al. (2022)
(GMD 15, 5905-5948).  These tests guard the physics transcription the way the
reference-pack tests guard the constants: the (A31) cell coordinate, (A32)
ITCZ position, (A33) Hadley width scale, the (A34) cell gradient ordering
(including the planted sign violation that pins the documented correction),
(A35) topography factor, (A30) circulation direction in all six cell
branches, (A29) zonal-SLP pattern and mass conservation, (A37) thermal
azonal SLP, and the (A38)-(A39) Charney–Eliassen response against a
hand-computed Fourier solution.

The supported climate path never calls this module
(``PlanetParams.enable_sesam_dynamics`` is False), so these tests exercise a
default-off diagnostic only.
"""
from __future__ import annotations

import numpy as np
import pytest

import sesam_dynamics as sd
from planet_params import PlanetParams

H, W = 64, 128
LAT = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
A2 = sd._a2_defaults()

EARTH = dict(gravity=9.81, radius_m=6.371e6, omega=7.2921e-5, p0_pa=101100.0)


def _earth_like_skin(lat: np.ndarray, w: int) -> np.ndarray:
    """Zonally uniform Earth-like skin temperature: 300 - 60 sin^2(lat)."""
    tsl_z = 300.0 - 60.0 * np.sin(lat) ** 2
    return np.repeat(tsl_z[:, None], w, axis=1)


# ---------------------------------------------------------------------------
# (A32) ITCZ, (A33) Hadley width scale, (A31) cell coordinate
# ---------------------------------------------------------------------------


def test_itcz_latitude_hand_value_and_sign():
    itcz = sd.itcz_latitude(289.0, 287.0, a2=A2)
    assert itcz == pytest.approx(A2["c2mmc"] * 2.0)
    assert itcz > 0.0  # warmer NH pulls the ITCZ north
    assert sd.itcz_latitude(287.0, 289.0, a2=A2) < 0.0


def test_hadley_width_scale_reciprocal_form_and_clamps():
    # Verified fraction form: Dhad = c3/(T_trp - c4) (module docstring note 2).
    t_trp = 288.0
    scale = sd.hadley_width_scale(t_trp, a2=A2)
    assert scale == pytest.approx(A2["c3mmc"] / (t_trp - A2["c4mmc"]))
    # Warming shrinks the scale, which widens the Hadley cells (paper text).
    assert sd.hadley_width_scale(t_trp + 10.0, a2=A2) < scale
    # Very hot tropics hits the lower clamp; cold tropics the upper clamp.
    assert sd.hadley_width_scale(400.0, a2=A2) == pytest.approx(0.5)
    assert sd.hadley_width_scale(210.0, a2=A2) == pytest.approx(1.5)
    # T_trp at/below c4 is floored at c4 + 50 K before dividing.
    floored = sd.hadley_width_scale(A2["c4mmc"] - 20.0, a2=A2)
    assert floored == pytest.approx(
        np.clip(A2["c3mmc"] / 50.0, 0.5, 1.5)
    )


def test_cell_coordinate_zero_at_itcz_and_hand_value():
    itcz = 0.05
    dhad = 1.0
    phi = sd.cell_coordinate(np.array([itcz]), itcz, dhad, a2=A2)
    assert phi[0] == pytest.approx(0.0, abs=1e-12)
    # Hand check at a chosen latitude: 6*Dhad*(lat - itcz/(c1*(lat-itcz)^2+1)).
    lat = np.array([0.3, -0.2])
    delta = lat - itcz
    expected = 6.0 * dhad * (lat - itcz / (A2["c1mmc"] * delta**2 + 1.0))
    np.testing.assert_allclose(sd.cell_coordinate(lat, itcz, dhad, a2=A2), expected)
    # Poles asymptote to ±6*Dhad*pi/2 when the ITCZ is at the equator.
    poles = sd.cell_coordinate(np.array([-np.pi / 2, np.pi / 2]), 0.0, dhad, a2=A2)
    assert poles[1] == pytest.approx(3.0 * np.pi, rel=1e-3)
    assert poles[0] == pytest.approx(-3.0 * np.pi, rel=1e-3)


def test_hadley_geometry_recovers_edges_and_width():
    dhad = 1.0
    phi_mmc = sd.cell_coordinate(LAT, 0.0, dhad, a2=A2)
    geom = sd.hadley_geometry(LAT, phi_mmc)
    # With Dhad = 1 the ±π crossings sit near ±32°.
    assert geom["hadley_edge_nh_rad"] == pytest.approx(np.radians(31.6), abs=0.05)
    assert geom["hadley_edge_sh_rad"] == pytest.approx(np.radians(-31.6), abs=0.05)
    assert geom["hadley_centre_rad"] == pytest.approx(0.0, abs=0.02)
    assert geom["hadley_width_rad"] == pytest.approx(
        geom["hadley_edge_nh_rad"] - geom["hadley_edge_sh_rad"]
    )
    # Unbracketed coordinate raises rather than fabricating an edge.
    with pytest.raises(ValueError):
        sd.hadley_geometry(LAT, 0.3 * phi_mmc)


def test_tropical_weight_from_hadley_shape():
    centre = 0.1
    width = np.radians(60.0)
    ft = sd.tropical_weight_from_hadley(LAT, centre, width)
    # 1 at the Hadley centre, 0 toward the poles, bounded [0, 1].
    assert np.max(ft) == pytest.approx(1.0, abs=1e-3)
    assert ft[0] < 1e-6 and ft[-1] < 1e-6
    assert ft.min() >= 0.0 and ft.max() <= 1.0
    # Hand form at one interior latitude.
    lat = 0.4
    fi = np.clip(0.7 * (lat - centre) / (0.5 * width), -np.pi / 2, np.pi / 2)
    expected = 1.0 - np.sin(fi) ** 8
    got = sd.tropical_weight_from_hadley(np.array([lat]), centre, width)
    assert got[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# (A34) gradients, (A35) topography factor
# ---------------------------------------------------------------------------


def test_cell_temperature_gradients_hand_values():
    # Piecewise-linear in |lat| so interpolation to the fixed latitudes is exact.
    tsl_z = 300.0 - 60.0 * np.abs(LAT) / (0.5 * np.pi)
    grad = sd.cell_temperature_gradients(LAT, tsl_z)
    t30, t60 = 280.0, 260.0
    # The tropical max and the pole value are sampled on the grid: the max
    # sits at the equator-adjacent rows, the pole interpolation clamps to the
    # northernmost/southernmost row.
    t_max_g = 300.0 - 60.0 * np.min(np.abs(LAT)) / (0.5 * np.pi)
    t90_g = 300.0 - 60.0 * np.max(np.abs(LAT)) / (0.5 * np.pi)
    np.testing.assert_allclose(
        grad["nh"], [t_max_g - t30, t30 - t60, t60 - t90_g], rtol=1e-9
    )
    np.testing.assert_allclose(
        grad["sh"], [t_max_g - t30, t30 - t60, t60 - t90_g], rtol=1e-9
    )
    assert grad["t_max_k"] == pytest.approx(t_max_g)


def test_cell_temperature_gradients_hadley_gradient_nonnegative_on_inverted_profile():
    # Plant a profile with a cool tropics and warm ±30° ridges: the Hadley
    # gradient stays non-negative by construction (the paper's global max
    # makes the max(0, .) floor vacuous but keeps the property).
    control_lat = np.array([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0.0, np.pi / 6, np.pi / 3, np.pi / 2])
    control_t = np.array([250.0, 280.0, 300.0, 285.0, 300.0, 280.0, 250.0])
    tsl_z = np.interp(LAT, control_lat, control_t)
    grad = sd.cell_temperature_gradients(LAT, tsl_z)
    assert grad["nh"][0] >= 0.0
    assert grad["sh"][0] >= 0.0
    # And the inverted gradient is smaller than on the Earth-like profile
    # (the cool tropics drives a weaker Hadley cell).
    earth = sd.cell_temperature_gradients(LAT, 300.0 - 60.0 * np.sin(LAT) ** 2)
    assert grad["nh"][0] < earth["nh"][0]


def test_topography_factor_hand_value_and_clamps():
    fz = sd.topography_factor(np.array([0.0, 375.0, 1500.0, -100.0]), a2=A2)
    np.testing.assert_allclose(fz, [1.0, 0.5, 0.0, 1.0])


# ---------------------------------------------------------------------------
# (A30) overturning wind — circulation direction in all six branches
# ---------------------------------------------------------------------------


def _earth_cell_inputs():
    skin = _earth_like_skin(LAT, W)
    tsl_z = np.mean(skin, axis=1)
    phi_mmc = sd.cell_coordinate(LAT, 0.0, 1.0, a2=A2)
    grad = sd.cell_temperature_gradients(LAT, tsl_z)
    fz = np.ones(H)
    return phi_mmc, grad, fz


def test_overturning_wind_directions_all_six_branches():
    phi_mmc, grad, fz = _earth_cell_inputs()
    va = sd.mean_overturning_wind(phi_mmc, grad["nh"], grad["sh"], fz, a2=A2)

    def at(lat_deg: float) -> float:
        return float(va[np.argmin(np.abs(LAT - np.radians(lat_deg)))])

    # NH: equatorward Hadley and polar surface flow, poleward Ferrel flow.
    assert at(15.0) < 0.0
    assert at(45.0) > 0.0
    assert at(75.0) < 0.0
    # SH mirrors with opposite signs.
    assert at(-15.0) > 0.0
    assert at(-45.0) < 0.0
    assert at(-75.0) > 0.0


def test_overturning_wind_hand_value():
    phi_mmc, grad, fz = _earth_cell_inputs()
    va = sd.mean_overturning_wind(phi_mmc, grad["nh"], grad["sh"], fz, a2=A2)
    j = np.argmin(np.abs(LAT - np.radians(15.0)))
    expected = -A2["C1_cell"] * grad["nh"][0] * np.sin(phi_mmc[j])
    assert va[j] == pytest.approx(expected, rel=1e-9)
    # Hadley is the strongest cell, polar the weakest (C1 >> C3).
    assert abs(va[np.argmin(np.abs(LAT - np.radians(15.0)))]) > abs(
        va[np.argmin(np.abs(LAT - np.radians(75.0)))]
    )


def test_overturning_wind_zero_beyond_3pi():
    # Dhad at the 1.5 clamp pushes high-latitude rows past |phi| = 3*pi.
    phi_mmc = sd.cell_coordinate(LAT, 0.0, 1.5, a2=A2)
    assert np.max(np.abs(phi_mmc)) > 3.0 * np.pi
    dt = np.array([10.0, 10.0, 10.0])
    va = sd.mean_overturning_wind(phi_mmc, dt, dt, np.ones(H), a2=A2)
    beyond = np.abs(phi_mmc) >= 3.0 * np.pi
    assert np.all(va[beyond] == 0.0)
    assert np.any(va[~beyond] != 0.0)


def test_printed_a34_ordering_reverses_circulation_planted_violation():
    """Pin the documented (A34) correction (module docstring note 3).

    Feeding the gradients in the paper's printed operand order
    (T(30) - T_max, T(60) - T(30), T(pole) - T(60)) through (A30) must
    reverse every cell's surface flow relative to the implemented ordering.
    """
    phi_mmc, grad, fz = _earth_cell_inputs()
    va = sd.mean_overturning_wind(phi_mmc, grad["nh"], grad["sh"], fz, a2=A2)

    # Printed ordering (negative gradients on Earth) fed through the same
    # (A30) kernel; (A30) is linear in ΔT, so the flow field flips exactly.
    va_printed = sd.mean_overturning_wind(phi_mmc, -grad["nh"], -grad["sh"], fz, a2=A2)

    interior = np.abs(phi_mmc) < 3.0 * np.pi
    np.testing.assert_allclose(va_printed[interior], -va[interior], rtol=1e-9)
    # And the flipped field is poleward in the NH Hadley cell — the
    # physically impossible circulation the correction avoids.
    j15 = np.argmin(np.abs(LAT - np.radians(15.0)))
    assert va_printed[j15] > 0.0
    assert va[j15] < 0.0


# ---------------------------------------------------------------------------
# (A29) zonal SLP
# ---------------------------------------------------------------------------


def test_zonal_slp_pattern_and_mass_conservation():
    phi_mmc, grad, fz = _earth_cell_inputs()
    va = sd.mean_overturning_wind(phi_mmc, grad["nh"], grad["sh"], fz, a2=A2)
    p = sd.zonal_slp_anomaly(
        LAT, va, 0.433, radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289
    )
    assert np.isfinite(p).all()
    # Zero cosine-weighted global mean (mass conservation).
    w = np.cos(LAT)
    assert np.sum(p * w) / np.sum(w) == pytest.approx(0.0, abs=1e-9)
    extrema = sd.zonal_slp_extrema(LAT, p, 0.0)
    # Subtropical highs in both hemispheres, between 20° and 45°.
    for hemi in ("nh", "sh"):
        high = extrema[hemi]["subtropical_high"]
        low = extrema[hemi]["subpolar_low"]
        assert 15.0 < abs(high["latitude_deg"]) < 45.0
        assert 45.0 < abs(low["latitude_deg"]) < 80.0
        assert high["anomaly_pa"] > 0.0
        assert low["anomaly_pa"] < 0.0
        # Earth-like magnitude: highs a few hPa above the global mean.
        assert 100.0 < high["anomaly_pa"] < 3000.0
    # ITCZ trough below the subtropical highs.
    assert extrema["itcz_trough"]["anomaly_pa"] < extrema["nh"]["subtropical_high"]["anomaly_pa"]
    # Pressure rises poleward of each subpolar low (relative polar highs;
    # the weak C3 polar cell need not lift the anomaly above zero).
    assert p[0] > extrema["nh"]["subpolar_low"]["anomaly_pa"]
    assert p[-1] > extrema["sh"]["subpolar_low"]["anomaly_pa"]


def test_zonal_slp_equator_stable_and_amplitude_scales_with_alpha():
    phi_mmc, grad, fz = _earth_cell_inputs()
    va = sd.mean_overturning_wind(phi_mmc, grad["nh"], grad["sh"], fz, a2=A2)
    p1 = sd.zonal_slp_anomaly(
        LAT, va, 0.433, radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289
    )
    p2 = sd.zonal_slp_anomaly(
        LAT, va, 0.866, radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289
    )
    assert np.isfinite(p1).all() and np.isfinite(p2).all()
    # Doubling |sin a cos a| halves the anomaly (pattern unchanged).
    np.testing.assert_allclose(p2, 0.5 * p1, rtol=1e-9)
    # Scalar and uniform unsigned-magnitude (H,) input are identical (the
    # |f|/magnitude convention of module docstring note 5).
    p3 = sd.zonal_slp_anomaly(
        LAT, va, np.full(H, 0.433), radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289
    )
    np.testing.assert_allclose(p3, p1, rtol=1e-12)
    # For a north/south-symmetric climate the anomaly is symmetric: v̄a is
    # antisymmetric while |f| and the magnitude factor are symmetric, so
    # ∂p̄/∂ϕ is antisymmetric and its pole-to-pole integral symmetric.
    np.testing.assert_allclose(p1[::-1], p1, atol=1e-6)


# ---------------------------------------------------------------------------
# (A37) thermal azonal SLP
# ---------------------------------------------------------------------------


def test_sea_level_temperature_hand_value():
    tsl = sd.sea_level_temperature(np.full((2, 2), 280.0), np.array([[0.0, 1000.0], [2000.0, -500.0]]))
    np.testing.assert_allclose(tsl, [[280.0, 286.5], [293.0, 276.75]])


def test_thermal_azonal_slp_sign_magnitude_and_zero_mean():
    tsl = np.full((H, W), 288.0)
    tsl[:, : W // 2] += 5.0  # warm "ocean" half, cold "land" half
    p = sd.thermal_azonal_slp(
        tsl, gravity=EARTH["gravity"], p0_pa=EARTH["p0_pa"], reference_temp_k=273.15, a2=A2
    )
    coefficient = (
        EARTH["gravity"] * EARTH["p0_pa"] * A2["H0_slp"]
        / (2.0 * 287.0 * 273.15**2)
    )
    # Warm half -> low pressure of exactly the (A37) magnitude.
    assert p[0, 0] == pytest.approx(-coefficient * 2.5, rel=1e-9)
    assert p[0, W // 2] == pytest.approx(+coefficient * 2.5, rel=1e-9)
    # Zero zonal mean on every row.
    np.testing.assert_allclose(np.mean(p, axis=1), 0.0, atol=1e-9)
    # Earth-constant sanity: ~230 Pa/K, so a 10 K contrast is ~23 hPa.
    assert coefficient == pytest.approx(231.0, rel=0.02)


# ---------------------------------------------------------------------------
# (A38)-(A39) Charney–Eliassen topographic waves
# ---------------------------------------------------------------------------


def test_charney_eliassen_matches_hand_fourier_solution():
    h, w = 32, 128
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lam = np.linspace(0.0, 2.0 * np.pi, w, endpoint=False)
    amplitude = 800.0
    zs = np.repeat((amplitude * np.cos(3.0 * lam))[None, :], h, axis=0)
    out = sd.charney_eliassen_slp(
        zs,
        lat,
        np.full(h, 20.0),
        np.full(h, 12000.0),
        radius_m=EARTH["radius_m"],
        omega=EARTH["omega"],
        rho0_kg_m3=1.289,
        p0_pa=EARTH["p0_pa"],
        a2=A2,
    )
    j = np.argmin(np.abs(lat - np.pi / 4.0))
    phi = lat[j]
    k0 = 1.0 / (EARTH["radius_m"] * np.cos(phi))
    m = np.pi / (EARTH["radius_m"] * np.radians(35.0))
    kzn = 3.0 * k0
    kn2 = kzn**2 + m**2
    beta = 2.0 * EARTH["omega"] * np.cos(phi) / EARTH["radius_m"]
    f = 2.0 * EARTH["omega"] * np.sin(phi)
    tau_s = A2["tau_e"] * 86400.0
    denom = complex(20.0 * kn2 - beta, -kn2 / (tau_s * kzn * 20.0))
    response = (f / 12000.0) * 0.4 * 20.0 / denom
    psi_expected = np.real(response * amplitude * np.exp(1j * 3.0 * lam))
    rho500 = 1.289 * 50000.0 / EARTH["p0_pa"]
    p_expected = rho500 * abs(f) * psi_expected
    np.testing.assert_allclose(out[j], p_expected, rtol=1e-9)
    np.testing.assert_allclose(np.mean(out, axis=1), 0.0, atol=1e-9)


def test_charney_eliassen_zero_topography_and_equator_stability():
    h, w = 16, 64
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    zero = sd.charney_eliassen_slp(
        np.zeros((h, w)),
        lat,
        np.full(h, 15.0),
        np.full(h, 12000.0),
        radius_m=EARTH["radius_m"],
        omega=EARTH["omega"],
        rho0_kg_m3=1.289,
        p0_pa=EARTH["p0_pa"],
        a2=A2,
    )
    assert np.abs(zero).max() == 0.0
    rng = np.random.default_rng(0)
    out = sd.charney_eliassen_slp(
        rng.normal(0.0, 500.0, (h, w)),
        lat,
        np.full(h, 15.0),
        np.full(h, 12000.0),
        radius_m=EARTH["radius_m"],
        omega=EARTH["omega"],
        rho0_kg_m3=1.289,
        p0_pa=EARTH["p0_pa"],
        a2=A2,
    )
    assert np.isfinite(out).all()
    # Response vanishes toward the equator (f -> 0 in the (A38) forcing).
    j_eq = np.argmin(np.abs(lat))
    j_mid = np.argmin(np.abs(lat - np.pi / 4.0))
    assert np.abs(out[j_eq]).max() < np.abs(out[j_mid]).max()


def test_charney_eliassen_u500_floor_applies():
    h, w = 8, 32
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lam = np.linspace(0.0, 2.0 * np.pi, w, endpoint=False)
    zs = np.repeat((500.0 * np.cos(2.0 * lam))[None, :], h, axis=0)
    base = dict(
        radius_m=EARTH["radius_m"], omega=EARTH["omega"], rho0_kg_m3=1.289,
        p0_pa=EARTH["p0_pa"], a2=A2,
    )
    easterly = sd.charney_eliassen_slp(zs, lat, np.full(h, -5.0), np.full(h, 12000.0), **base)
    floored = sd.charney_eliassen_slp(zs, lat, np.full(h, 0.1), np.full(h, 12000.0), **base)
    # Easterlies are clamped to the +0.1 m/s westerly floor, not NaN/Inf.
    np.testing.assert_allclose(easterly, floored)
    assert np.isfinite(easterly).all()


# ---------------------------------------------------------------------------
# (A28) assembly
# ---------------------------------------------------------------------------


def test_compute_slp_assembly_conservation_and_components():
    skin = _earth_like_skin(LAT, W)
    zs = np.zeros((H, W))
    res = sd.compute_slp(
        skin_temp_k=skin,
        surface_elevation_m=zs,
        sin_cos_alpha_bar=0.433,
        gravity=EARTH["gravity"],
        radius_m=EARTH["radius_m"],
        omega=EARTH["omega"],
        p0_pa=EARTH["p0_pa"],
        reference_temp_k=273.15,
        a2=A2,
    )
    assert res.slp_pa.shape == (H, W)
    # Global cosine-weighted mean restored to p0 exactly.
    w2d = np.broadcast_to(np.cos(LAT)[:, None], (H, W))
    assert np.sum(res.slp_pa * w2d) / np.sum(w2d) == pytest.approx(EARTH["p0_pa"], abs=1e-6)
    # Full field = zonal anomaly + azonal parts + the p0 offset.
    rebuilt = (
        res.zonal_slp_anomaly_pa[:, None]
        + res.thermal_azonal_slp_pa
        + res.orographic_azonal_slp_pa
    )
    offset = EARTH["p0_pa"] - np.sum(rebuilt * w2d) / np.sum(w2d)
    np.testing.assert_allclose(res.slp_pa, rebuilt + offset, atol=1e-6)
    # Zonally uniform input -> zero azonal parts.
    assert np.abs(res.thermal_azonal_slp_pa).max() < 1e-6
    assert np.abs(res.orographic_azonal_slp_pa).max() == 0.0
    # Sane diagnostics from the Earth-like input.
    assert abs(np.degrees(res.itcz_latitude_rad)) < 2.0
    assert 25.0 < np.degrees(res.hadley_edge_nh_rad) < 40.0
    assert np.isfinite(res.slp_pa).all()


def test_compute_slp_reproducible_and_shape_validated():
    skin = _earth_like_skin(LAT, W)
    zs = np.zeros((H, W))
    kwargs = dict(
        skin_temp_k=skin, surface_elevation_m=zs, sin_cos_alpha_bar=0.433,
        gravity=EARTH["gravity"], radius_m=EARTH["radius_m"], omega=EARTH["omega"],
        p0_pa=EARTH["p0_pa"], reference_temp_k=273.15, a2=A2,
    )
    a = sd.compute_slp(**kwargs)
    b = sd.compute_slp(**kwargs)
    np.testing.assert_array_equal(a.slp_pa, b.slp_pa)
    with pytest.raises(ValueError):
        sd.compute_slp(**{**kwargs, "surface_elevation_m": np.zeros((H + 1, W))})
    with pytest.raises(ValueError):
        sd.compute_slp(**{**kwargs, "u500_m_s": np.full(H, 15.0)})  # needs tropopause


def test_compute_slp_with_charney_eliassen_term():
    skin = _earth_like_skin(LAT, W)
    lam = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False)
    zs = np.repeat((600.0 * np.cos(2.0 * lam))[None, :], H, axis=0)
    res = sd.compute_slp(
        skin_temp_k=skin,
        surface_elevation_m=zs,
        sin_cos_alpha_bar=0.433,
        gravity=EARTH["gravity"],
        radius_m=EARTH["radius_m"],
        omega=EARTH["omega"],
        p0_pa=EARTH["p0_pa"],
        reference_temp_k=273.15,
        u500_m_s=np.full(H, 20.0),
        tropopause_height_m=np.full(H, 12000.0),
        a2=A2,
    )
    assert np.abs(res.orographic_azonal_slp_pa).max() > 0.0
    assert np.isfinite(res.slp_pa).all()
    w2d = np.broadcast_to(np.cos(LAT)[:, None], (H, W))
    assert np.sum(res.slp_pa * w2d) / np.sum(w2d) == pytest.approx(EARTH["p0_pa"], abs=1e-6)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def test_sesam_dynamics_gate_defaults_off():
    assert PlanetParams().enable_sesam_dynamics is False
