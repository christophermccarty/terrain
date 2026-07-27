from __future__ import annotations

import numpy as np

import simulate
import state_persistence
from gui_view_cache import (
    OCEAN_CURRENT_CACHE,
    PRECIP_VIEW_CACHE,
    WIND_CACHE,
    invalidate_gui_view_caches,
)
from simulation_runner import initialize_state, run_multiple_steps
from simulation_state import PlanetState, TimeScaleMode
from sim_grid import _coarsen, _coarsen_elevation_cached, clear_grid_caches


def test_simulate_reexports_stable_state_and_persistence_api():
    assert simulate.PlanetState is PlanetState
    assert simulate.TimeScaleMode is TimeScaleMode
    assert simulate.save_state is state_persistence.save_state
    assert simulate.load_state is state_persistence.load_state
    assert simulate.STATE_SCHEMA_VERSION == state_persistence.STATE_SCHEMA_VERSION


def test_runner_orchestration_accepts_injected_physics():
    initial = PlanetState(day_of_year=0.0, elevation=np.zeros((2, 4), dtype=np.float32))
    calls: list[float] = []

    def step(state, *, days, **_kwargs):
        calls.append(days)
        return state._replace(total_days=state.total_days + days), {"days": days}

    states, components = run_multiple_steps(
        initial,
        total_days=2.5,
        step_days=1.0,
        step_function=step,
        step_kwargs={},
    )

    assert calls == [1.0, 1.0, 0.5]
    assert states[-1].total_days == 2.5
    assert components[-1] == {"days": 0.5}


def test_state_initialization_seeds_planet_composition():
    from planet_params import MARS

    elevation = np.zeros((2, 4), dtype=np.float32)

    def step(state, *, days, **_kwargs):
        assert days == 0.0
        return state, {}

    state = initialize_state(
        elevation,
        42.0,
        step_function=step,
        step_kwargs={"planet_params": MARS},
    )

    assert state.day_of_year == 42.0
    assert state.co2_atmosphere == MARS.co2_initial_ppm
    assert state.ch4_atmosphere == MARS.ch4_initial_ppb
    assert state.planet_params is MARS


def test_grid_coarsening_preserves_edge_padding_and_cache_contract():
    elevation = np.arange(15, dtype=np.float32).reshape(3, 5)
    expected = np.pad(elevation, ((0, 1), (0, 1)), mode="edge")
    expected = expected.reshape(2, 2, 3, 2).mean(axis=(1, 3))

    result = _coarsen(elevation, 2, 3, 2)
    np.testing.assert_array_equal(result, expected)

    cached = _coarsen_elevation_cached(elevation, 2, 3, 2)
    assert _coarsen_elevation_cached(elevation, 2, 3, 2) is cached
    assert not cached.flags.writeable
    clear_grid_caches()
    assert _coarsen_elevation_cached(elevation, 2, 3, 2) is not cached


def test_gui_view_cache_invalidation_is_centralized():
    WIND_CACHE.update({"key": "wind", "u": object(), "v": object()})
    OCEAN_CURRENT_CACHE.update(
        {"key": "ocean", "u": object(), "v": object(), "computed_at": 42.0}
    )
    PRECIP_VIEW_CACHE.update({"key": "rain", "P": object()})

    invalidate_gui_view_caches()

    assert WIND_CACHE == {"key": None, "u": None, "v": None}
    assert OCEAN_CURRENT_CACHE == {
        "key": None,
        "u": None,
        "v": None,
        "computed_at": 0.0,
    }
    assert PRECIP_VIEW_CACHE == {"key": None, "P": None}
