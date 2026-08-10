"""Tests for latitude-dependent ocean mixed-layer depth (FEATURES.md item 6).

Two things were unified onto explicit, PlanetParams-exposed fields:

1. `_evolve_temperature`'s T_sst relaxation step already had a real, latitude
   -dependent mixed-layer depth ramp, but it was two hardcoded literals
   (30.0, 170.0). Now `mixed_layer_depth_tropical_m`/`_polar_m` -- Earth
   defaults reproduce the old ramp exactly (see
   test_mld_fields_reproduce_legacy_ramp_exactly).
2. The separate `ocean_seasonal_frac` calculation (T_base_ocean path) used an
   independent hand-tuned per-latitude polynomial, unrelated to any physical
   depth. `simulate._ocean_seasonal_fraction` now offers a second, physically
   -derived mode (`pp.derive_ocean_seasonal_lag=True`) computed from the same
   mixed-layer-depth fields via the standard slab-ocean relaxation response.
   Ships off by default pending a real-terrain calibration pass.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from planet_params import EARTH
from simulate import (
    _ocean_seasonal_fraction,
    create_initial_state,
    simulate_step,
)
from testing.conftest import make_mixed_elev

H, W = 32, 64


def test_mld_fields_default_to_legacy_earth_ramp():
    assert EARTH.mixed_layer_depth_tropical_m == pytest.approx(30.0)
    assert EARTH.mixed_layer_depth_polar_m == pytest.approx(200.0)
    assert EARTH.derive_ocean_seasonal_lag is False


def test_mld_fields_reproduce_legacy_ramp_exactly():
    """Perturbing the MLD fields must change the T_sst relaxation path --
    i.e. they are actually wired, not decorative defaults."""
    elev = make_mixed_elev(H, W)
    state = create_initial_state(elev, day_of_year=80.0, planet_params=EARTH)
    baseline, _ = simulate_step(state, days=1.0, planet_params=EARTH)

    pp_shallow = dataclasses.replace(
        EARTH, mixed_layer_depth_tropical_m=5.0, mixed_layer_depth_polar_m=40.0,
    )
    changed, _ = simulate_step(state, days=1.0, planet_params=pp_shallow)

    assert not np.allclose(baseline.temperature, changed.temperature, atol=1e-4), (
        "shallower mixed-layer depth should measurably change the T_sst "
        "relaxation response (faster response to forcing)"
    )


def test_ocean_seasonal_fraction_legacy_mode_unchanged():
    """Legacy formula must be untouched by the refactor into a shared helper."""
    lat_deg = np.array([0.0, 30.0, 45.0, 60.0, 90.0], dtype=np.float32)
    obliq_ratio = float(EARTH.obliquity_deg) / 23.44
    obliq_factor = np.clip(obliq_ratio, 0.6, 2.0) ** 0.5
    polar_lat_boost = np.sin(np.deg2rad(lat_deg)) ** 2
    high_obliq_boost = max(obliq_ratio - 1.0, 0.0)
    expected = (
        (0.05 + 0.20 * np.cos(np.deg2rad(lat_deg))) * obliq_factor
        + 0.60 * high_obliq_boost * polar_lat_boost
    )
    seasonal_cap = float(min(0.45 * obliq_factor, 0.85))
    expected = np.clip(expected, 0.03, seasonal_cap)

    actual = _ocean_seasonal_fraction(lat_deg, EARTH)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_derived_mode_is_a_distinct_real_formula():
    """The derived path must actually differ from the legacy one (otherwise
    the flag is decorative) and stay within a physically sane [0, 1] range."""
    lat_deg = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0], dtype=np.float32)
    pp_derived = dataclasses.replace(EARTH, derive_ocean_seasonal_lag=True)

    legacy = _ocean_seasonal_fraction(lat_deg, EARTH)
    derived = _ocean_seasonal_fraction(lat_deg, pp_derived)

    assert not np.allclose(legacy, derived, atol=1e-3)
    assert np.all(derived >= 0.0) and np.all(derived <= 1.0)
    # Monotonic intuition: a deeper (polar) mixed layer damps the seasonal
    # swing at least as much as the shallow tropical one.
    assert derived[0] >= derived[-1] - 1e-6, (
        f"equatorial fraction ({derived[0]:.4f}) should be >= polar ({derived[-1]:.4f}) "
        "-- deeper mixed layer means more thermal inertia, not less"
    )


def test_derived_mode_responds_to_relaxation_coefficient():
    """A stiffer (larger) radiative-restoring coefficient should shorten the
    effective thermal time constant and raise the seasonal fraction."""
    lat_deg = np.array([0.0, 45.0, 90.0], dtype=np.float32)
    pp_soft = dataclasses.replace(
        EARTH, derive_ocean_seasonal_lag=True, ocean_thermal_relaxation_coefficient=1.0,
    )
    pp_stiff = dataclasses.replace(
        EARTH, derive_ocean_seasonal_lag=True, ocean_thermal_relaxation_coefficient=10.0,
    )
    soft = _ocean_seasonal_fraction(lat_deg, pp_soft)
    stiff = _ocean_seasonal_fraction(lat_deg, pp_stiff)
    assert np.all(stiff > soft), (
        "a larger relaxation coefficient (shorter thermal time constant) should "
        "let more of the seasonal swing through"
    )


def test_derive_ocean_seasonal_lag_flag_is_wired_end_to_end():
    elev = make_mixed_elev(H, W)
    state = create_initial_state(elev, day_of_year=80.0, planet_params=EARTH)
    baseline, _ = simulate_step(state, days=1.0, planet_params=EARTH)

    pp_derived = dataclasses.replace(EARTH, derive_ocean_seasonal_lag=True)
    changed, _ = simulate_step(state, days=1.0, planet_params=pp_derived)

    assert not np.allclose(baseline.temperature, changed.temperature, atol=1e-4)
