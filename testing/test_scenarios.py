from __future__ import annotations

import numpy as np
import pytest

from scenarios import SCENARIOS, scenario_planet_params
from simulate import create_initial_state, simulate_step


def test_scenario_names_are_unique_and_parameters_valid():
    names = [scenario.name for scenario in SCENARIOS]
    assert len(names) == len(set(names))
    for scenario in SCENARIOS:
        pp = scenario.planet_params
        assert pp.orbital_period_days > 0.0
        assert pp.sidereal_day_hours > 0.0
        assert pp.surface_pressure_pa > 0.0
        assert pp.solar_constant > 0.0


@pytest.mark.parametrize(
    "name",
    [
        "High CO₂ Earth",
        "Snowball Earth",
        "High Obliquity Earth",
        "Eccentric Earth",
        "Milankovitch Earth",
        "Hydrological Earth",
        "Slow-Rotating Earth",
    ],
)
def test_earth_scenarios_advance_without_nan(name):
    pp = scenario_planet_params(name)
    elevation = np.zeros((8, 16), dtype=np.float32)
    state = create_initial_state(elevation, day_of_year=80.0, planet_params=pp)
    state, _ = simulate_step(
        state,
        days=1.0,
        block_size=2,
        wind_block_size=2,
        planet_params=pp,
    )
    assert np.all(np.isfinite(state.temperature))
    assert state.planet_params == pp


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError, match="Unknown scenario"):
        scenario_planet_params("Not a planet")
