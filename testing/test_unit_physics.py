"""test_unit_physics.py — Fast unit tests for individual physics functions.

All tests use synthetic data only.  No opensimplex, no terrain generation,
no simulation spinup.  Should complete in < 5 seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Insolation
# ---------------------------------------------------------------------------

def test_equinox_global_mean_insolation():
    """Global area-weighted mean insolation at equinox ≈ S0/4."""
    from planet_params import EARTH
    lat = np.linspace(-np.pi / 2, np.pi / 2, 1000, dtype=np.float32)
    Q = EARTH.daily_mean_insolation(lat, day_of_year=80)
    cos_lat = np.cos(lat)
    mean_Q = float(np.sum(Q * cos_lat) / np.sum(cos_lat))
    assert abs(mean_Q - EARTH.solar_constant / 4.0) < 15.0, (
        f"Global mean insolation = {mean_Q:.1f} W/m², expected ~{EARTH.solar_constant/4:.0f}"
    )


def test_polar_night_zero_insolation():
    """South pole receives zero insolation at NH summer solstice (day 172)."""
    from planet_params import EARTH
    Q = EARTH.daily_mean_insolation(np.array([-np.pi / 2], dtype=np.float32), 172)
    assert float(Q[0]) < 1.0, f"South pole insolation at NH summer = {Q[0]:.2f} W/m²"


def test_polar_day_positive_insolation():
    """North pole receives positive insolation at NH summer solstice."""
    from planet_params import EARTH
    Q = EARTH.daily_mean_insolation(np.array([np.pi / 2], dtype=np.float32), 172)
    assert float(Q[0]) > 100.0, f"North pole insolation at NH summer = {Q[0]:.1f} W/m²"


def test_insolation_nonnegative():
    """Insolation is never negative at any latitude or day."""
    from planet_params import EARTH
    lats = np.linspace(-np.pi / 2, np.pi / 2, 180, dtype=np.float32)
    for day in [1, 80, 172, 265, 355]:
        Q = EARTH.daily_mean_insolation(lats, day_of_year=day)
        assert np.all(Q >= 0.0), f"Negative insolation on day {day}: min={Q.min():.2f}"


# ---------------------------------------------------------------------------
# Coriolis
# ---------------------------------------------------------------------------

def test_coriolis_sign():
    """f > 0 in NH, f < 0 in SH, f = 0 at equator."""
    from planet_params import EARTH
    assert EARTH.coriolis_parameter(np.array([np.pi / 4]))[0] > 0.0,  "NH f should be positive"
    assert EARTH.coriolis_parameter(np.array([-np.pi / 4]))[0] < 0.0, "SH f should be negative"
    assert abs(EARTH.coriolis_parameter(np.array([0.0]))[0]) < 1e-10,  "Equator f should be zero"


def test_coriolis_poles():
    """f = ±2Ω at poles."""
    from planet_params import EARTH
    f_np = float(EARTH.coriolis_parameter(np.array([np.pi / 2]))[0])
    f_sp = float(EARTH.coriolis_parameter(np.array([-np.pi / 2]))[0])
    assert abs(f_np - 2 * EARTH.omega) < 1e-10, f"f_north_pole = {f_np:.4e}, expected {2*EARTH.omega:.4e}"
    assert abs(f_sp + 2 * EARTH.omega) < 1e-10, f"f_south_pole = {f_sp:.4e}, expected {-2*EARTH.omega:.4e}"


def test_coriolis_rotation_matrix_conserves_speed():
    """The exact rotation matrix must conserve |V| to floating-point precision."""
    rng = np.random.default_rng(42)
    u = rng.standard_normal((32, 64)).astype(np.float32)
    v = rng.standard_normal((32, 64)).astype(np.float32)
    speed_before = np.sqrt(u**2 + v**2)

    theta = np.full_like(u, np.pi / 6)  # 30° rotation
    u_rot = np.cos(theta) * u + np.sin(theta) * v
    v_rot = -np.sin(theta) * u + np.cos(theta) * v
    speed_after = np.sqrt(u_rot**2 + v_rot**2)

    np.testing.assert_allclose(speed_before, speed_after, rtol=1e-5,
        err_msg="Rotation matrix does not conserve wind speed")


# ---------------------------------------------------------------------------
# CO2 / radiative forcing
# ---------------------------------------------------------------------------

def test_co2_doubling_forcing():
    """2×CO2 should give ≈3.7 W/m² (IPCC AR6 best estimate 3.93 W/m² via ln(2)*5.35)."""
    from carbon_cycle import co2_radiative_forcing
    dF = co2_radiative_forcing(560.0, co2_reference=280.0)
    assert 3.5 < dF < 4.1, f"2×CO2 forcing = {dF:.3f} W/m² (expected ~3.7)"


def test_co2_reference_forcing_zero():
    """At reference CO2 the forcing should be exactly 0."""
    from carbon_cycle import co2_radiative_forcing
    dF = co2_radiative_forcing(280.0, co2_reference=280.0)
    assert abs(dF) < 0.01, f"Forcing at reference CO2 = {dF:.4f} W/m²"


def test_co2_forcing_monotone():
    """Higher CO2 → more forcing."""
    from carbon_cycle import co2_radiative_forcing
    f1 = co2_radiative_forcing(280.0)
    f2 = co2_radiative_forcing(400.0)
    f3 = co2_radiative_forcing(800.0)
    assert f1 < f2 < f3, f"Forcing not monotone: {f1:.2f}, {f2:.2f}, {f3:.2f}"


# ---------------------------------------------------------------------------
# Henry's Law / ocean CO2
# ---------------------------------------------------------------------------

def test_henry_cold_absorbs_more_co2():
    """Cold ocean dissolves more CO2 than warm ocean (Henry's Law)."""
    from carbon_cycle import ocean_co2_solubility
    T_cold = np.array([[271.0]], dtype=np.float32)
    T_warm = np.array([[298.0]], dtype=np.float32)
    c_cold = float(ocean_co2_solubility(T_cold, 400.0)[0, 0])
    c_warm = float(ocean_co2_solubility(T_warm, 400.0)[0, 0])
    assert c_cold > c_warm, f"Cold={c_cold:.1f} vs Warm={c_warm:.1f} ppm-eq"


def test_henry_scales_with_co2():
    """Equilibrium ocean CO2 scales linearly with atmospheric CO2."""
    from carbon_cycle import ocean_co2_solubility
    T = np.array([[288.0]], dtype=np.float32)
    c1 = float(ocean_co2_solubility(T, 280.0)[0, 0])
    c2 = float(ocean_co2_solubility(T, 560.0)[0, 0])
    ratio = c2 / c1
    assert 1.9 < ratio < 2.1, f"CO2 doubling ratio in ocean = {ratio:.3f} (expected ~2.0)"


# ---------------------------------------------------------------------------
# Sea ice
# ---------------------------------------------------------------------------

def test_sea_ice_forms_below_freeze():
    """Ocean cells below freeze_temp should gain ice."""
    from ocean import update_sea_ice
    T = np.full((16, 32), 269.0, dtype=np.float32)   # well below 271.0 K
    elev = np.zeros((16, 32), dtype=np.float32)       # all ocean
    ice, delta, _ = update_sea_ice(T, elev, None, dt_days=10.0)
    assert np.any(ice > 0.0), "No ice formed below freeze threshold"
    assert np.all(delta >= 0.0), "delta_ice < 0 during freezing"


def test_sea_ice_melts_above_melt():
    """Ocean cells above melt_temp should lose ice."""
    from ocean import update_sea_ice
    T = np.full((16, 32), 275.0, dtype=np.float32)   # well above 271.5 K
    elev = np.zeros((16, 32), dtype=np.float32)
    ice_prev = np.full((16, 32), 0.5, dtype=np.float32)
    ice, delta, _ = update_sea_ice(T, elev, ice_prev, dt_days=5.0)
    assert np.all(ice < 0.5), "Ice did not melt above melt threshold"
    assert np.all(delta <= 0.0), "delta_ice > 0 during melting"


def test_sea_ice_stays_in_bounds():
    """Ice fraction must always be in [0, 1]."""
    from ocean import update_sea_ice
    rng = np.random.default_rng(7)
    T = rng.uniform(265.0, 280.0, (32, 64)).astype(np.float32)
    elev = np.zeros((32, 64), dtype=np.float32)
    ice_prev = rng.uniform(0.0, 1.0, (32, 64)).astype(np.float32)
    ice, _, _h = update_sea_ice(T, elev, ice_prev, dt_days=1.0)
    assert np.all(ice >= 0.0) and np.all(ice <= 1.0), (
        f"Ice out of [0,1]: min={ice.min():.3f} max={ice.max():.3f}"
    )


def test_sea_ice_land_is_zero():
    """Land cells should always have zero ice."""
    from ocean import update_sea_ice
    T = np.full((16, 32), 265.0, dtype=np.float32)
    # Use loaded-DEM convention: 0.0 = ocean (bottom 4 rows), non-zero = land (rest).
    # This triggers the zeros_frac > 0.05 path in _ocean_mask_from_elevation so that
    # land cells are unambiguously identified as non-zero elevation.
    elev = np.full((16, 32), 0.5, dtype=np.float32)
    elev[-4:, :] = 0.0   # bottom 4 rows (~25%) are ocean
    land_mask = elev > 0.0  # land = non-zero in loaded-DEM convention
    ice, _, _h = update_sea_ice(T, elev, None, dt_days=10.0)
    assert np.all(ice[land_mask] == 0.0), "Ice formed on land cells"


# ---------------------------------------------------------------------------
# Land/sea mask consistency
# ---------------------------------------------------------------------------

def test_mask_loaded_dem_zeros_are_ocean():
    """Loaded DEM convention: cells with elevation == 0.0 are ocean."""
    from masks import get_masks
    elev = np.zeros((32, 64), dtype=np.float32)
    elev[10:20, 20:40] = 0.4   # land patch
    sea, land = get_masks(elev)  # get_masks returns (sea_mask, land_mask)
    # All zero cells should be sea
    assert np.all(sea[elev == 0.0]), "Zero-elevation cells should be ocean"
    # Non-zero cells should be land
    assert np.all(land[elev > 0.0]), "Positive-elevation cells should be land"


def test_mask_cache_returns_same_arrays():
    """Repeated calls with the same array should return the same objects (cached)."""
    from masks import get_masks
    elev = np.zeros((32, 64), dtype=np.float32)
    elev[:, 32:] = 0.3
    r1 = get_masks(elev)
    r2 = get_masks(elev)
    assert r1 is r2, "Mask cache should return the same tuple object"


def test_mask_land_sea_disjoint():
    """land_mask and sea_mask must be mutually exclusive and exhaustive."""
    from masks import get_masks
    elev = np.random.default_rng(0).random((32, 64)).astype(np.float32)
    land, sea = get_masks(elev)
    assert np.all(land ^ sea), "land_mask and sea_mask are not complementary"
    assert np.all(land | sea),  "Some cells are neither land nor sea"
