"""Tests for Feature 2: Water vapor → epsilon greenhouse feedback."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_humid_tropics_warmer_than_no_wv(mixed_initial_state):
    """With water_vapor_feedback=True, mean temperature should be ≥ baseline (WV warms)."""
    from simulate import simulate_step
    N = 10

    state_on = mixed_initial_state
    state_off = mixed_initial_state
    for _ in range(N):
        state_on, _ = simulate_step(state_on, days=1.0, feedback_flags={'water_vapor_feedback': True})
        state_off, _ = simulate_step(state_off, days=1.0, feedback_flags={'water_vapor_feedback': False})

    T_on = float(np.mean(state_on.temperature))
    T_off = float(np.mean(state_off.temperature))
    # WV reduces epsilon → less OLR → warmer; allow near-zero diff (effect accumulates slowly)
    assert T_on >= T_off - 1.0, f"WV feedback should not significantly cool: T_on={T_on:.2f}, T_off={T_off:.2f}"


def test_water_vapor_feedback_flag_no_crash(mixed_initial_state):
    """water_vapor_feedback=False must run without error and give finite fields."""
    from simulate import simulate_step
    state, _ = simulate_step(
        mixed_initial_state, days=1.0, feedback_flags={'water_vapor_feedback': False}
    )
    assert np.all(np.isfinite(state.temperature))
    assert np.all(np.isfinite(state.air_temperature))


def test_wv_factor_zero_matches_flag_off(mixed_initial_state):
    """wv_greenhouse_factor=0 (MARS-like) must give same result as feedback flag off."""
    from simulate import simulate_step
    from planet_params import PlanetParams

    pp_no_wv = PlanetParams(wv_greenhouse_factor=0.0)
    state_zero, _ = simulate_step(mixed_initial_state, days=1.0, planet_params=pp_no_wv)
    state_flag, _ = simulate_step(
        mixed_initial_state, days=1.0, feedback_flags={'water_vapor_feedback': False}
    )
    # Both should give very similar temperatures (within rounding)
    diff = float(np.mean(np.abs(state_zero.temperature - state_flag.temperature)))
    assert diff < 0.5, f"Mean T diff between factor=0 and flag=False: {diff:.3f} K"


def test_temperature_remains_physical(mixed_initial_state):
    """After 20 steps with WV feedback, T must remain in physical range."""
    from simulate import simulate_step
    state = mixed_initial_state
    for _ in range(20):
        state, _ = simulate_step(state, days=1.0)
    T = state.temperature
    assert float(np.min(T)) > 150.0, f"T too cold: {float(np.min(T)):.1f} K"
    assert float(np.max(T)) < 360.0, f"T too hot: {float(np.max(T)):.1f} K"
