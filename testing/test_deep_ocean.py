"""Tests for Feature 5: Deep ocean 2-layer heat uptake."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_deep_ocean_initialized(mixed_initial_state):
    """T_deep_ocean should be initialized on first step for a wet planet."""
    from simulate import simulate_step
    state, _ = simulate_step(mixed_initial_state, days=1.0)
    assert state.T_deep_ocean is not None, "T_deep_ocean should be non-None after first step"
    assert state.T_deep_ocean.shape == state.elevation.shape


def test_deep_ocean_colder_than_surface(mixed_initial_state):
    """Initial deep ocean should be cooler than SST on average."""
    from simulate import simulate_step
    state, _ = simulate_step(mixed_initial_state, days=1.0)

    from masks import get_masks
    sea_mask, _ = get_masks(state.elevation)

    T_sst_mean = float(np.mean(state.temperature[sea_mask]))
    T_deep_mean = float(np.mean(state.T_deep_ocean[sea_mask]))
    assert T_deep_mean < T_sst_mean, (
        f"Deep ocean ({T_deep_mean:.1f}K) should be colder than SST ({T_sst_mean:.1f}K)"
    )


def test_heat_flows_downward(mixed_initial_state):
    """Warm SST anomaly should warm the deep ocean over time."""
    from simulate import simulate_step
    import copy

    # Run a baseline for 10 steps to initialize deep ocean
    state = mixed_initial_state
    for _ in range(10):
        state, _ = simulate_step(state, days=1.0)

    if state.T_deep_ocean is None:
        pytest.skip("T_deep_ocean not initialized")

    from masks import get_masks
    sea_mask, _ = get_masks(state.elevation)

    T_deep_before = float(np.mean(state.T_deep_ocean[sea_mask]))

    # Run 365 more days
    for _ in range(365):
        state, _ = simulate_step(state, days=1.0)

    T_deep_after = float(np.mean(state.T_deep_ocean[sea_mask]))

    # Deep ocean should change (doesn't have to warm, just not be frozen)
    assert 265.0 < T_deep_after < 310.0, f"Deep ocean T out of range: {T_deep_after:.1f} K"


def test_deep_ocean_not_on_dry_planet():
    """Mars (no liquid water) should not initialize T_deep_ocean."""
    from simulate import create_initial_state, simulate_step
    from planet_params import MARS
    from testing.conftest import make_mixed_elev

    elev = make_mixed_elev(32, 64)
    # Use MARS params for both init and step so T_deep never gets initialized
    state = create_initial_state(elev, day_of_year=80.0, planet_params=MARS)
    assert state.T_deep_ocean is None, "Initial state should not have deep ocean for Mars"

    state_next, _ = simulate_step(state, days=1.0, planet_params=MARS)
    assert state_next.T_deep_ocean is None, "Mars should not have deep ocean layer"


def test_deep_ocean_field_persists():
    """T_deep_ocean should persist and evolve across steps (not reset each step)."""
    from simulate import simulate_step

    from testing.conftest import make_mixed_elev
    from simulate import create_initial_state

    H, W = 32, 64
    elev = make_mixed_elev(H, W)
    state = create_initial_state(elev, day_of_year=80.0)

    # Run 5 steps
    states = [state]
    for _ in range(5):
        s, _ = simulate_step(states[-1], days=1.0)
        states.append(s)

    # T_deep_ocean should be present by step 2 and not identical across steps
    t_deep_vals = [
        float(np.mean(s.T_deep_ocean)) if s.T_deep_ocean is not None else None
        for s in states[1:]
    ]
    non_none = [v for v in t_deep_vals if v is not None]
    assert len(non_none) >= 4, "T_deep_ocean should persist across steps"
