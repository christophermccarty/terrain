"""Tests for SESAM stage P6d -- live coupling of the (A69)-(A117)/(A10)
radiation chain into the prognostic simulation loop
(`sesam_radiation_coupling.py`).

docs/SESAM_GAP_ANALYSIS.md Sec7 P6. These tests cover the glue layer under
near-equilibrium conditions (a realistic Ta/T* gap); they deliberately do
NOT assert multi-day stability under a large, sustained Ta/T* contrast --
see `testing/test_sesam_vertical.py`'s
`test_near_surface_lapse_large_cold_land_contrast_is_unbounded` for the real,
open finding that a sustained large contrast currently produces an
unphysical vertical temperature profile (P1's `near_surface_lapse`, not a
P6d wiring bug -- see this module's own docstring point 5).
"""
from __future__ import annotations

import numpy as np
import pytest

from planet_params import EARTH
from sesam_radiation_coupling import sesam_radiation_step


def _synthetic_fields(h=8, w=16, seed=0):
    rng = np.random.default_rng(seed)
    ta = 273.0 + 15.0 * rng.uniform(-1, 1, size=(h, w))
    tstar = ta + 2.0 * rng.uniform(-1, 1, size=(h, w))  # small, realistic gap
    ra = np.clip(0.5 + 0.2 * rng.uniform(-1, 1, size=(h, w)), 0.0, 1.0)
    fcld = np.clip(0.5 + 0.2 * rng.uniform(-1, 1, size=(h, w)), 0.0, 1.0)
    elev = np.clip(0.2 + 0.1 * rng.uniform(-1, 1, size=(h, w)), 0.0, 1.0) * 3000.0
    land = rng.uniform(0, 1, size=(h, w)) > 0.5
    return ta, tstar, ra, fcld, elev, land


def _call(ta, tstar, ra, fcld, elev, land, **overrides):
    h, w = ta.shape
    lat_deg = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lat_rad = np.deg2rad(lat_deg)[:, None] * np.ones((1, w))
    day = 80.0
    decl = EARTH.solar_declination(day)
    insolation = EARTH.daily_mean_insolation(lat_rad, day)
    kwargs = dict(
        air_temperature_k=ta, skin_temperature_k=tstar, relative_humidity=ra,
        cloud_fraction=fcld, column_water_mm=None, elevation_m=elev,
        land_mask=land, ice_mask=None, snow_depth=None, tropopause_height_m=None,
        surface_pressure_pa=float(EARTH.surface_pressure_pa),
        gravity=float(EARTH.surface_gravity), co2_ppm=420.0, day_of_year=day,
        itcz_seasonal_response=float(EARTH.itcz_seasonal_response),
        solar_declination_rad=float(decl), daily_mean_insolation_w_m2=insolation,
        dt_days=1.0,
    )
    kwargs.update(overrides)
    return sesam_radiation_step(**kwargs)


def test_gate_defaults_off():
    assert EARTH.enable_sesam_radiation is False


def test_radiation_step_finite_and_shaped():
    ta, tstar, ra, fcld, elev, land = _synthetic_fields()
    h, w = ta.shape
    result = _call(ta, tstar, ra, fcld, elev, land)
    for field in (result.sw_absorbed_w_m2, result.lw_net_w_m2, result.tropopause_height_m):
        assert field.shape == (h, w)
        assert np.all(np.isfinite(field))


def test_radiation_step_sign_conventions_sane_under_near_equilibrium():
    """Global-mean SWa should be positive (the atmosphere absorbs some
    shortwave) and global-mean LWa negative (net radiative cooling of the
    atmosphere column) -- the same qualitative regime real climate and
    stage P5's own TOA exit-gate measurement both show."""
    ta, tstar, ra, fcld, elev, land = _synthetic_fields()
    result = _call(ta, tstar, ra, fcld, elev, land)
    assert float(np.mean(result.sw_absorbed_w_m2)) > 0.0
    assert float(np.mean(result.lw_net_w_m2)) < 0.0


def test_tropopause_height_stays_within_clip_bounds():
    ta, tstar, ra, fcld, elev, land = _synthetic_fields()
    result = _call(ta, tstar, ra, fcld, elev, land)
    assert np.all(result.tropopause_height_m >= elev + 3000.0 - 1.0)
    assert np.all(result.tropopause_height_m <= 30_000.0 + 1.0)


def test_column_water_lazy_fallback_matches_qa_scale():
    ta, tstar, ra, fcld, elev, land = _synthetic_fields()
    result_none = _call(ta, tstar, ra, fcld, elev, land, column_water_mm=None)
    from sesam_vertical import saturation_specific_humidity
    p0_field = np.full(ta.shape, float(EARTH.surface_pressure_pa))
    qa = ra * saturation_specific_humidity(ta, p0_field)
    explicit_qq = qa * 2000.0
    result_explicit = _call(ta, tstar, ra, fcld, elev, land, column_water_mm=explicit_qq)
    assert result_none.sw_absorbed_w_m2 == pytest.approx(result_explicit.sw_absorbed_w_m2, rel=1e-9)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _call(
            np.zeros((4,)), np.zeros((4, 4)), np.zeros((4, 4)),
            np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4), dtype=bool),
        )
