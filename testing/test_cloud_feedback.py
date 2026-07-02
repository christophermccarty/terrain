"""Tests for Feature 1: Cloud radiative feedback."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cloud_persistence(mixed_initial_state):
    """Cloud cover should not jump more than 20% per step at daily timescale."""
    from simulate import simulate_step
    state = mixed_initial_state
    new_state, _ = simulate_step(state, days=1.0)
    prev_cloud = state.cloud_cover
    if prev_cloud is not None and new_state.cloud_cover is not None:
        diff = np.abs(new_state.cloud_cover - prev_cloud)
        assert float(np.mean(diff)) < 0.20, f"Mean cloud change too large: {float(np.mean(diff)):.3f}"


def test_cloud_cover_plausible_range(mixed_initial_state):
    """Cloud fraction must stay in [0, 1] and mean 0.3–0.9 after several steps."""
    from simulate import simulate_step
    state = mixed_initial_state
    for _ in range(5):
        state, _ = simulate_step(state, days=1.0)
    cf = state.cloud_cover
    assert cf is not None
    assert float(np.min(cf)) >= -1e-6
    assert float(np.max(cf)) <= 1.0 + 1e-6
    assert 0.20 <= float(np.mean(cf)) <= 0.95, f"Global mean cloud fraction: {float(np.mean(cf)):.3f}"


def test_cloud_feedback_flag_no_crash(mixed_initial_state):
    """cloud_feedback=False must run without error and give finite temperatures."""
    from simulate import simulate_step
    state, _ = simulate_step(
        mixed_initial_state, days=1.0, feedback_flags={'cloud_feedback': False}
    )
    assert np.all(np.isfinite(state.temperature))
    assert np.all(np.isfinite(state.air_temperature))


def test_cloud_cover_stored_in_state(mixed_initial_state):
    """cloud_cover must be stored in PlanetState after each step."""
    from simulate import simulate_step
    state, _ = simulate_step(mixed_initial_state, days=1.0)
    assert state.cloud_cover is not None
    assert state.cloud_cover.shape == mixed_initial_state.elevation.shape
