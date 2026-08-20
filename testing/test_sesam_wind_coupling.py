"""Tests for SESAM stage P6c -- live coupling of the P2 (SLP/wind) and P3
(EKE) closures into the prognostic simulation loop (`sesam_wind_coupling.py`).

docs/SESAM_GAP_ANALYSIS.md Sec7 P6. These tests guard the glue layer only
(shape/finiteness/sanity); the underlying P2/P3 equations are already
covered by `testing/test_sesam_dynamics.py`, `testing/test_sesam_wind.py`,
and `testing/test_sesam_synoptic.py`.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import sesam_wind_coupling as swc
from planet_params import EARTH, PlanetParams


def _synthetic_fields(h=8, w=16, seed=1):
    rng = np.random.default_rng(seed)
    skin = 273.0 + 25.0 * rng.uniform(-1, 1, size=(h, w))
    ta = skin + 3.0 * rng.uniform(-1, 1, size=(h, w))
    ra = np.clip(0.5 + 0.2 * rng.uniform(-1, 1, size=(h, w)), 0.0, 1.0)
    elev = np.clip(0.2 + 0.1 * rng.uniform(-1, 1, size=(h, w)), 0.0, 1.0) * 3000.0
    land = rng.uniform(0, 1, size=(h, w)) > 0.5
    return skin, ta, ra, elev, land


def test_dynamics_gate_defaults_off():
    assert PlanetParams().enable_sesam_dynamics is False


def test_wind_and_eke_step_finite_and_shaped():
    skin, ta, ra, elev, land = _synthetic_fields()
    h, w = skin.shape
    result = swc.sesam_wind_and_eke_step(
        air_temperature_k=ta, skin_temperature_k=skin, relative_humidity=ra,
        elevation_m=elev, land_mask=land, ice_mask=None,
        surface_pressure_pa=float(EARTH.surface_pressure_pa),
        radius_m=float(EARTH.radius_m), gravity=float(EARTH.surface_gravity),
        omega=float(EARTH.omega),
    )
    for field in (result.wind_u_m_s, result.wind_v_m_s, result.eke_m2_s2, result.total_wind_m_s):
        assert field.shape == (h, w)
        assert np.all(np.isfinite(field))
    assert np.all(result.eke_m2_s2 >= 0.0)
    # Sanity band, not a tight physical assertion: SESAM's own P2/P3 exit
    # gates already validated realistic magnitudes on real terrain
    # (docs/SESAM_GAP_ANALYSIS.md Sec7 P2/P3); this just catches a
    # catastrophically wrong wiring (e.g. an accidental unit error).
    speed = np.sqrt(result.wind_u_m_s ** 2 + result.wind_v_m_s ** 2)
    assert speed.mean() < 100.0
    assert result.eke_m2_s2.mean() < 1.0e5
    # (A58) Us = sqrt(us^2+vs^2+Usyn^2) is the zonal magnitude plus a
    # nonnegative synoptic-gustiness term added in quadrature -- it can
    # never be smaller than the bare zonal speed it's built from.
    assert np.all(result.total_wind_m_s >= speed - 1e-9)


def test_wind_and_eke_step_with_ice_mask_does_not_crash():
    skin, ta, ra, elev, land = _synthetic_fields()
    ice = np.zeros_like(land, dtype=bool)
    ice[0, :] = True
    result = swc.sesam_wind_and_eke_step(
        air_temperature_k=ta, skin_temperature_k=skin, relative_humidity=ra,
        elevation_m=elev, land_mask=land, ice_mask=ice,
        surface_pressure_pa=float(EARTH.surface_pressure_pa),
        radius_m=float(EARTH.radius_m), gravity=float(EARTH.surface_gravity),
        omega=float(EARTH.omega),
    )
    assert np.all(np.isfinite(result.wind_u_m_s))
    assert np.all(np.isfinite(result.eke_m2_s2))


def test_sesam_column_closure_step_accepts_real_eke_field():
    """sesam_coupling.py's eke_m2_s2 parameter (added for P6c) must actually
    be used, not silently ignored in favour of the placeholder."""
    import sesam_coupling as sc

    h, w = 6, 12
    rng = np.random.default_rng(2)
    ta = 280.0 + 10.0 * rng.uniform(-1, 1, size=(h, w))
    tstar = ta + 2.0 * rng.uniform(-1, 1, size=(h, w))
    ra = np.clip(0.6 + 0.15 * rng.uniform(-1, 1, size=(h, w)), 0.0, 1.0)
    elev = 500.0 + 200.0 * rng.uniform(-1, 1, size=(h, w))
    land = np.zeros((h, w), dtype=bool)
    u = 2.0 + rng.uniform(-1, 1, size=(h, w))
    v = rng.uniform(-1, 1, size=(h, w))

    low_eke = np.full((h, w), 1.0)
    high_eke = np.full((h, w), 5000.0)
    result_low = sc.sesam_column_closure_step(
        air_temperature_k=ta, skin_temperature_k=tstar, relative_humidity=ra,
        column_water_mm=None, wind_u_m_s=u, wind_v_m_s=v,
        elevation_m=elev, land_mask=land,
        surface_pressure_pa=float(EARTH.surface_pressure_pa),
        radius_m=float(EARTH.radius_m), dt_days=1.0, eke_m2_s2=low_eke,
    )
    result_high = sc.sesam_column_closure_step(
        air_temperature_k=ta, skin_temperature_k=tstar, relative_humidity=ra,
        column_water_mm=None, wind_u_m_s=u, wind_v_m_s=v,
        elevation_m=elev, land_mask=land,
        surface_pressure_pa=float(EARTH.surface_pressure_pa),
        radius_m=float(EARTH.radius_m), dt_days=1.0, eke_m2_s2=high_eke,
    )
    # Different EKE -> different diffusivity -> different precipitation
    # convergence/slope terms, even from an otherwise-identical, spatially
    # uniform starting state.
    assert not np.allclose(
        result_low.precipitation_mm_day, result_high.precipitation_mm_day
    )


def test_simulate_step_p6c_gated_on_runs_without_nan():
    from real_terrain_validation import load_bundled_earth_dem
    from simulate import create_initial_state, simulate_step, TimeScaleMode

    h, w = 16, 32
    elevation = load_bundled_earth_dem(h, w)
    pp = replace(EARTH, enable_sesam_column_closure=True, enable_sesam_dynamics=True)

    state = create_initial_state(elevation, day_of_year=80.0, planet_params=pp, block_size=1)
    for _ in range(3):
        state, _ = simulate_step(state, days=1.0, planet_params=pp, block_size=1,
                                  time_scale=TimeScaleMode.DAILY)
    assert state.sesam_column_water_mm is not None
    for name in ("air_temperature", "precipitation", "humidity", "sesam_column_water_mm"):
        arr = np.asarray(getattr(state, name))
        assert np.all(np.isfinite(arr)), f"{name} produced non-finite values"


def test_simulate_step_dynamics_gate_alone_is_inert_without_column_closure():
    """enable_sesam_dynamics=True with enable_sesam_column_closure=False must
    not call this module at all -- the P6c wiring lives entirely inside the
    P6b gate's own branch in simulate_step."""
    from real_terrain_validation import load_bundled_earth_dem
    from simulate import create_initial_state, simulate_step, TimeScaleMode

    h, w = 16, 32
    elevation = load_bundled_earth_dem(h, w)
    pp = replace(EARTH, enable_sesam_dynamics=True)
    assert pp.enable_sesam_column_closure is False

    state = create_initial_state(elevation, day_of_year=80.0, planet_params=pp, block_size=1)
    state, _ = simulate_step(state, days=1.0, planet_params=pp, block_size=1,
                              time_scale=TimeScaleMode.DAILY)
    assert state.sesam_column_water_mm is None
