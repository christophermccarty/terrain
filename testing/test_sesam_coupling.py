"""Tests for SESAM stage P6b -- live coupling of the column-energy/water
closure into the prognostic simulation loop (`sesam_coupling.py`).

docs/SESAM_GAP_ANALYSIS.md Sec7 P6. These tests guard the glue layer only
(shape/finiteness/gate-off-inertness); `sesam_thermo.py`'s own equations are
already covered by `testing/test_sesam_thermo.py`.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import sesam_coupling as sc
from planet_params import EARTH, PlanetParams


def _synthetic_fields(h=8, w=16, seed=0):
    rng = np.random.default_rng(seed)
    ta = 273.0 + 20.0 * rng.uniform(-1, 1, size=(h, w))
    tstar = ta + 3.0 * rng.uniform(-1, 1, size=(h, w))
    ra = np.clip(0.5 + 0.2 * rng.uniform(-1, 1, size=(h, w)), 0.0, 1.0)
    elev = np.clip(0.2 + 0.1 * rng.uniform(-1, 1, size=(h, w)), 0.0, 1.0)
    land = rng.uniform(0, 1, size=(h, w)) > 0.5
    u = 3.0 * rng.uniform(-1, 1, size=(h, w))
    v = 2.0 * rng.uniform(-1, 1, size=(h, w))
    return ta, tstar, ra, elev, land, u, v


def test_gate_defaults_off():
    assert PlanetParams().enable_sesam_column_closure is False


def test_column_closure_step_finite_and_shaped():
    ta, tstar, ra, elev, land, u, v = _synthetic_fields()
    h, w = ta.shape
    result = sc.sesam_column_closure_step(
        air_temperature_k=ta, skin_temperature_k=tstar, relative_humidity=ra,
        column_water_mm=None, wind_u_m_s=u, wind_v_m_s=v,
        elevation_m=elev * 3000.0, land_mask=land,
        surface_pressure_pa=float(EARTH.surface_pressure_pa),
        radius_m=float(EARTH.radius_m), dt_days=1.0,
    )
    for field in (
        result.air_temperature_k, result.relative_humidity,
        result.precipitation_mm_day, result.column_water_mm,
    ):
        assert field.shape == (h, w)
        assert np.all(np.isfinite(field))
    assert np.all(result.relative_humidity >= 0.0) and np.all(result.relative_humidity <= 1.0)
    assert np.all(result.precipitation_mm_day >= 0.0)
    assert np.all(result.column_water_mm >= 0.0)
    assert np.all(result.air_temperature_k >= 150.0) and np.all(result.air_temperature_k <= 350.0)


def test_column_closure_step_shape_mismatch_raises():
    ta = np.zeros((4, 4))
    with pytest.raises(ValueError):
        sc.sesam_column_closure_step(
            air_temperature_k=np.zeros((4,)), skin_temperature_k=ta,
            relative_humidity=ta, column_water_mm=None,
            wind_u_m_s=ta, wind_v_m_s=ta, elevation_m=ta,
            land_mask=np.zeros((4, 4), dtype=bool),
            surface_pressure_pa=101325.0, radius_m=6.371e6, dt_days=1.0,
        )


def test_column_water_lazy_init_matches_documented_scale():
    ta, tstar, ra, elev, land, u, v = _synthetic_fields()
    p0 = float(EARTH.surface_pressure_pa)
    from sesam_vertical import saturation_specific_humidity
    qa0 = ra * saturation_specific_humidity(ta, np.full(ta.shape, p0))
    expected_qq0 = qa0 * sc._QQ_SCALE_KG_M2_PER_KGKG

    # Passing the expected lazy-init value explicitly should not change the
    # zero-diffusion/zero-advection-independent water budget path relative to
    # column_water_mm=None (same starting point, just supplied explicitly).
    result_none = sc.sesam_column_closure_step(
        air_temperature_k=ta, skin_temperature_k=tstar, relative_humidity=ra,
        column_water_mm=None, wind_u_m_s=u, wind_v_m_s=v,
        elevation_m=elev * 3000.0, land_mask=land,
        surface_pressure_pa=p0, radius_m=float(EARTH.radius_m), dt_days=1.0,
    )
    result_explicit = sc.sesam_column_closure_step(
        air_temperature_k=ta, skin_temperature_k=tstar, relative_humidity=ra,
        column_water_mm=expected_qq0, wind_u_m_s=u, wind_v_m_s=v,
        elevation_m=elev * 3000.0, land_mask=land,
        surface_pressure_pa=p0, radius_m=float(EARTH.radius_m), dt_days=1.0,
    )
    assert result_none.column_water_mm == pytest.approx(result_explicit.column_water_mm, rel=1e-9)


def test_zero_wind_zero_diabatic_still_diffuses_but_stays_bounded():
    """No advection, no skin-air contrast: Ta should only respond to
    macroturbulent diffusion (placeholder-EKE-driven), so a spatially uniform
    input field is left exactly unchanged (nothing to diffuse toward)."""
    h, w = 6, 12
    ta = np.full((h, w), 280.0)
    ra = np.full((h, w), 0.6)
    elev = np.full((h, w), 500.0)
    land = np.zeros((h, w), dtype=bool)
    zeros = np.zeros((h, w))
    result = sc.sesam_column_closure_step(
        air_temperature_k=ta, skin_temperature_k=ta, relative_humidity=ra,
        column_water_mm=None, wind_u_m_s=zeros, wind_v_m_s=zeros,
        elevation_m=elev, land_mask=land,
        surface_pressure_pa=float(EARTH.surface_pressure_pa),
        radius_m=float(EARTH.radius_m), dt_days=1.0,
    )
    assert result.air_temperature_k == pytest.approx(280.0, abs=1e-6)


def test_simulate_step_default_path_unaffected_by_sesam_gate():
    """Gate off must be a byte-identical no-op: `sesam_column_water_mm` stays
    None and legacy air_temperature/precipitation are untouched by this
    module (docs/SESAM_GAP_ANALYSIS.md Sec7 P1-P5's own standing reservation,
    now load-bearing for P6's live wiring too)."""
    from real_terrain_validation import load_bundled_earth_dem
    from simulate import create_initial_state, simulate_step, TimeScaleMode

    h, w = 16, 32
    elevation = load_bundled_earth_dem(h, w)
    pp = EARTH
    assert pp.enable_sesam_column_closure is False

    state = create_initial_state(elevation, day_of_year=80.0, planet_params=pp, block_size=1)
    state, _ = simulate_step(state, days=1.0, planet_params=pp, block_size=1,
                              time_scale=TimeScaleMode.DAILY)
    assert state.sesam_column_water_mm is None
    assert np.all(np.isfinite(np.asarray(state.air_temperature)))


def test_simulate_step_gated_on_runs_without_nan():
    from real_terrain_validation import load_bundled_earth_dem
    from simulate import create_initial_state, simulate_step, TimeScaleMode

    h, w = 16, 32
    elevation = load_bundled_earth_dem(h, w)
    pp = replace(EARTH, enable_sesam_column_closure=True)

    state = create_initial_state(elevation, day_of_year=80.0, planet_params=pp, block_size=1)
    for _ in range(3):
        state, _ = simulate_step(state, days=1.0, planet_params=pp, block_size=1,
                                  time_scale=TimeScaleMode.DAILY)
    assert state.sesam_column_water_mm is not None
    for name in ("air_temperature", "precipitation", "humidity", "sesam_column_water_mm"):
        arr = np.asarray(getattr(state, name))
        assert np.all(np.isfinite(arr)), f"{name} produced non-finite values"


def test_real_diabatic_source_path_finite_and_shaped():
    """P6d: when sw_absorbed_w_m2/lw_net_w_m2 are supplied, the real (A40)
    assembly (sesam_thermo.diabatic_heating_rate_k_day) replaces bridge 3
    instead of the (T*-Ta)/1-day fallback."""
    ta, tstar, ra, elev, land, u, v = _synthetic_fields()
    h, w = ta.shape
    rng = np.random.default_rng(1)
    swa = 60.0 + 20.0 * rng.uniform(-1, 1, size=(h, w))
    lwa = -150.0 + 30.0 * rng.uniform(-1, 1, size=(h, w))
    result = sc.sesam_column_closure_step(
        air_temperature_k=ta, skin_temperature_k=tstar, relative_humidity=ra,
        column_water_mm=None, wind_u_m_s=u, wind_v_m_s=v,
        elevation_m=elev * 3000.0, land_mask=land,
        surface_pressure_pa=float(EARTH.surface_pressure_pa),
        radius_m=float(EARTH.radius_m), dt_days=1.0,
        sw_absorbed_w_m2=swa, lw_net_w_m2=lwa, gravity_m_s2=float(EARTH.surface_gravity),
    )
    for field in (
        result.air_temperature_k, result.relative_humidity,
        result.precipitation_mm_day, result.column_water_mm,
    ):
        assert field.shape == (h, w)
        assert np.all(np.isfinite(field))
    assert np.all(result.air_temperature_k >= 150.0) and np.all(result.air_temperature_k <= 350.0)


def test_real_diabatic_source_requires_both_fields():
    """Supplying only one of sw_absorbed_w_m2/lw_net_w_m2 falls back to
    bridge 3 (both-or-neither), not a partial/undefined source."""
    ta, tstar, ra, elev, land, u, v = _synthetic_fields()
    h, w = ta.shape
    swa_only = sc.sesam_column_closure_step(
        air_temperature_k=ta, skin_temperature_k=tstar, relative_humidity=ra,
        column_water_mm=None, wind_u_m_s=u, wind_v_m_s=v,
        elevation_m=elev * 3000.0, land_mask=land,
        surface_pressure_pa=float(EARTH.surface_pressure_pa),
        radius_m=float(EARTH.radius_m), dt_days=1.0,
        sw_absorbed_w_m2=np.full((h, w), 60.0), lw_net_w_m2=None,
    )
    bridge_only = sc.sesam_column_closure_step(
        air_temperature_k=ta, skin_temperature_k=tstar, relative_humidity=ra,
        column_water_mm=None, wind_u_m_s=u, wind_v_m_s=v,
        elevation_m=elev * 3000.0, land_mask=land,
        surface_pressure_pa=float(EARTH.surface_pressure_pa),
        radius_m=float(EARTH.radius_m), dt_days=1.0,
    )
    assert swa_only.air_temperature_k == pytest.approx(bridge_only.air_temperature_k, rel=1e-9)


def test_simulate_step_gated_on_with_radiation_runs_without_nan_near_equilibrium():
    """P6d gated on alongside P6b, single DAILY step from a fresh (near-
    equilibrium) initial state -- see testing/test_sesam_vertical.py's
    `test_near_surface_lapse_large_cold_land_contrast_produces_unphysical_profile`
    for the documented open finding that a *sustained, large* Ta-T* contrast
    (not exercised by a single step from a fresh state) currently makes P1's
    cold-land lapse branch blow up; this test intentionally stays inside the
    near-equilibrium regime that finding does not cover."""
    from real_terrain_validation import load_bundled_earth_dem
    from simulate import create_initial_state, simulate_step, TimeScaleMode

    h, w = 16, 32
    elevation = load_bundled_earth_dem(h, w)
    pp = replace(EARTH, enable_sesam_column_closure=True, enable_sesam_radiation=True)

    state = create_initial_state(elevation, day_of_year=80.0, planet_params=pp, block_size=1)
    state, _ = simulate_step(state, days=1.0, planet_params=pp, block_size=1,
                              time_scale=TimeScaleMode.DAILY)
    for name in ("air_temperature", "precipitation", "humidity", "sesam_column_water_mm"):
        arr = np.asarray(getattr(state, name))
        assert np.all(np.isfinite(arr)), f"{name} produced non-finite values"
    assert state.sesam_tropopause_height_m is not None
    assert np.all(np.isfinite(state.sesam_tropopause_height_m))


def test_simulate_step_multiday_call_does_not_saturate_clip_bounds():
    """Regression test for a real bug found and fixed 2026-08-19: the
    diabatic bridge in `sesam_coupling.py` is a 1-day relaxation *rate*
    (K/day). Calling `sesam_column_closure_step` once per multi-day outer
    step (MONTHLY mode's `days=30`) applied that rate for the whole span in
    a single Euler step, overshooting so badly that `air_temperature` went
    globally uniform at the 350 K clip bound one call, then the 150 K bound
    the next, oscillating every subsequent call -- confirmed on a real
    64x128 real-terrain run before the fix (both `state.temperature`, the
    untouched legacy skin field, and `state.air_temperature` collapsed to an
    unphysical uniform ~250 K global mean over a 1-year spinup). The fix
    substeps the closure at ~1-day cadence inside `simulate_step`
    (`_SESAM_COLUMN_CLOSURE_SUBSTEP_DAYS`), mirroring the pre-existing
    `_generate_precipitation_substepped`/`_evolve_wind_substepped` idiom.
    This test calls `simulate_step` with a single 30-day span (MONTHLY mode)
    and checks the legacy skin temperature -- never touched by this SESAM
    branch -- stays in a physically sane range, which the pre-fix bug's
    downstream ice-albedo/radiation feedback broke even though this module
    never writes that field directly."""
    from real_terrain_validation import load_bundled_earth_dem
    from simulate import create_initial_state, simulate_step, TimeScaleMode

    h, w = 16, 32
    elevation = load_bundled_earth_dem(h, w)
    pp = replace(EARTH, enable_sesam_column_closure=True)

    state = create_initial_state(elevation, day_of_year=80.0, planet_params=pp, block_size=1)
    for _ in range(3):
        state, _ = simulate_step(state, days=30.0, planet_params=pp, block_size=1,
                                  time_scale=TimeScaleMode.MONTHLY)
    skin = np.asarray(state.temperature)
    assert skin.std() > 1.0, (
        "legacy skin temperature collapsed to a near-uniform field -- the "
        "multi-day SESAM diabatic-bridge overshoot bug is back"
    )
    assert 250.0 < skin.mean() < 310.0
