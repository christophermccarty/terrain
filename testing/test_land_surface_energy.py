from __future__ import annotations

import dataclasses

import numpy as np

from planet_params import EARTH
from simulate import create_initial_state, simulate_step


def _land_state():
    elevation = np.zeros((16, 32), dtype=np.float32)
    elevation[4:12, 8:24] = 0.5
    return elevation, create_initial_state(elevation, planet_params=EARTH)


def test_land_surface_energy_gate_is_inert_by_default():
    _, state = _land_state()
    baseline, _ = simulate_step(state, days=1.0, planet_params=EARTH)
    explicit_off, _ = simulate_step(
        state, days=1.0, planet_params=dataclasses.replace(EARTH, enable_land_surface_energy=False)
    )
    np.testing.assert_array_equal(baseline.temperature, explicit_off.temperature)


def test_land_surface_energy_changes_enabled_land_state():
    elevation, state = _land_state()
    baseline, _ = simulate_step(state, days=1.0, planet_params=EARTH)
    enabled, _ = simulate_step(
        state,
        days=1.0,
        planet_params=dataclasses.replace(EARTH, enable_land_surface_energy=True),
    )
    land = elevation > 0.0
    assert not np.array_equal(enabled.temperature[land], baseline.temperature[land])


def test_force_restore_land_is_a_gated_replacement_with_persistent_deep_soil():
    elevation, state = _land_state()
    enabled, _ = simulate_step(
        state,
        days=1.0,
        planet_params=dataclasses.replace(EARTH, enable_force_restore_land=True),
    )
    assert enabled.land_deep_temperature is not None
    land = elevation > 0.0
    assert np.all(np.isfinite(enabled.land_deep_temperature[land]))
