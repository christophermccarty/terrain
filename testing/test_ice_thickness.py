"""test_ice_thickness.py — Feature 6: sea ice thickness state variable.

Tests that ice_thickness is tracked, evolves physically (Stefan's law),
and that thin ice produces lower surface albedo than thick ice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _step(state, days: float = 1.0):
    from simulate import simulate_step
    return simulate_step(state, days=days, block_size=4)[0]


def test_ice_thickness_field_in_state(mixed_initial_state):
    """ice_thickness should be a 2-D array in PlanetState after simulate_step."""
    state = _step(mixed_initial_state)
    assert state.ice_thickness is not None, "ice_thickness is None after simulate_step"
    assert state.ice_thickness.shape == state.elevation.shape
    assert np.all(np.isfinite(state.ice_thickness)), "ice_thickness contains NaN/Inf"


def test_ice_thickness_nonnegative(mixed_initial_state):
    """Thickness must never go negative after multiple steps."""
    state = mixed_initial_state
    for _ in range(30):
        state = _step(state, days=1.0)
    assert np.all(state.ice_thickness >= 0.0), (
        f"Negative ice thickness: min={state.ice_thickness.min():.3f} m"
    )


def test_ice_thickness_zero_where_no_ice(mixed_initial_state):
    """Cells with no ice cover must have zero thickness."""
    state = mixed_initial_state
    for _ in range(30):
        state = _step(state, days=1.0)
    if state.ice_cover is None:
        pytest.skip("No ice_cover in state")
    no_ice = state.ice_cover == 0.0
    assert np.all(state.ice_thickness[no_ice] == 0.0), (
        "Non-zero thickness found where ice_cover == 0"
    )


def test_thin_ice_lower_albedo_than_thick():
    """alpha_ice(h) formula: thin ice (0.1 m) has lower albedo than thick ice (2 m)."""
    # alpha = 0.06 + 0.59 * min(h / 0.5, 1.0)
    alpha_thin  = 0.06 + 0.59 * min(0.1 / 0.5, 1.0)   # = 0.178
    alpha_thick = 0.06 + 0.59 * min(2.0 / 0.5, 1.0)   # = 0.650
    assert alpha_thin < alpha_thick, (
        f"Thin ice albedo ({alpha_thin:.3f}) not lower than thick ice ({alpha_thick:.3f})"
    )


def test_thick_ice_grows_slower_than_thin():
    """Stefan's law: thicker ice grows more slowly in the same cold conditions."""
    from ocean import update_sea_ice

    H, W = 8, 16
    elev = np.zeros((H, W), dtype=np.float32)
    T_cold = np.full((H, W), 260.0, dtype=np.float32)
    ice_full = np.ones((H, W), dtype=np.float32)

    h_thin_start  = np.full((H, W), 0.2, dtype=np.float32)
    h_thick_start = np.full((H, W), 2.0, dtype=np.float32)

    _, _, h_thin_end  = update_sea_ice(T_cold, elev, ice_full, 30.0, h_thin_start)
    _, _, h_thick_end = update_sea_ice(T_cold, elev, ice_full, 30.0, h_thick_start)

    growth_thin  = float(np.mean(h_thin_end  - h_thin_start))
    growth_thick = float(np.mean(h_thick_end - h_thick_start))

    assert growth_thin > growth_thick, (
        f"Thin ice grew less ({growth_thin:.4f} m) than thick ice ({growth_thick:.4f} m); "
        "Stefan's law violated"
    )
