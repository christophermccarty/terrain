from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hydrology import route_surface_water
from masks import get_masks
from planet_params import EARTH
from simulate import create_initial_state, simulate_step


def _area_weight(shape):
    H, W = shape
    lat = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * np.pi
    return np.broadcast_to(np.cos(lat)[:, None], (H, W))


def test_runoff_routing_conserves_area_weighted_water():
    H, W = 12, 24
    elevation = np.broadcast_to(
        np.linspace(0.8, 0.05, W, dtype=np.float32)[None, :], (H, W)
    ).copy()
    elevation[:, -1] = 0.0  # ocean outlet
    _, land = get_masks(elevation, use_cache=False)
    runoff = np.where(land, 2.0, 0.0).astype(np.float32)

    storage, throughflow, ocean_outflow, ocean_river_input = route_surface_water(
        elevation,
        runoff,
        None,
        dt_days=1.0,
        routing_passes=24,
        routing_fraction=0.6,
    )

    weights = _area_weight(elevation.shape)
    water_in = float(np.sum(runoff * weights))
    water_left = float(np.sum(storage * weights))
    water_out = float(np.sum(ocean_outflow * weights))
    water_received = float(np.sum(ocean_river_input * weights))
    assert water_left + water_out == pytest.approx(water_in, rel=2e-6)
    assert float(np.max(throughflow)) > 0.0
    assert water_out > 0.0
    # ocean_outflow (reported at the draining land cell) and ocean_river_input
    # (reported at the receiving ocean cell) must carry the same total flux --
    # they're two views of the same water, not independent quantities.
    assert water_received == pytest.approx(water_out, rel=2e-6)


def test_ocean_river_input_lands_at_receiving_cell_not_source():
    """ocean_river_input_mm_day must be indexed at the ocean neighbor that
    actually receives the flow, not at the draining land cell (that's what
    ocean_outflow_mm_day is for) -- the two must not just have the same total,
    they must be spatially distinct."""
    H, W = 5, 5
    elevation = np.full((H, W), 0.8, dtype=np.float32)
    elevation[:, -1] = 0.0  # ocean column on the east edge
    runoff = np.zeros_like(elevation)
    runoff[2, 3] = 5.0  # land cell directly west of the ocean column

    storage, throughflow, ocean_outflow, ocean_river_input = route_surface_water(
        elevation, runoff, None, dt_days=1.0, routing_passes=4, routing_fraction=1.0
    )

    assert ocean_outflow[2, 3] > 0.0
    assert ocean_river_input[2, 3] == 0.0  # source cell is land, not the receiver
    # D8 ties (three ocean neighbors all at elevation 0.0) are broken by the
    # implementation's fixed neighbor-scan order, not guaranteed to be due
    # east -- check the flow landed in *some* ocean cell adjacent to the
    # source, not a specific one. (Totals aren't compared as plain mm sums
    # here since source and receiver rows can have different area weights --
    # see test_runoff_routing_conserves_area_weighted_water for the
    # area-weighted conservation check.)
    assert float(ocean_river_input[1:4, 4].sum()) > 0.0


def test_closed_depression_retains_surface_water():
    elevation = np.full((7, 9), 0.8, dtype=np.float32)
    elevation[[0, -1], :] = 0.0
    elevation[:, [0, -1]] = 0.0
    elevation[3, 4] = 0.2
    runoff = np.zeros_like(elevation)
    runoff[3, 4] = 10.0

    storage, _, ocean_outflow, ocean_river_input = route_surface_water(
        elevation, runoff, None, dt_days=1.0, routing_passes=12
    )

    assert storage[3, 4] == pytest.approx(10.0)
    assert float(np.sum(ocean_outflow)) == 0.0
    assert float(np.sum(ocean_river_input)) == 0.0


def test_hydrology_state_is_gated_and_persistent(mixed_elev):
    enabled = dataclasses.replace(EARTH, enable_surface_hydrology=True)
    state = create_initial_state(mixed_elev, planet_params=enabled)
    assert state.surface_water_mm is not None
    assert state.river_discharge_mm_day is not None
    assert state.runoff_to_ocean_mm_day is not None

    next_state, _ = simulate_step(
        state,
        days=1.0,
        block_size=4,
        wind_block_size=4,
        planet_params=enabled,
    )
    assert np.all(np.isfinite(next_state.surface_water_mm))
    assert np.all(next_state.surface_water_mm >= 0.0)


def test_surface_water_overlay_is_transparent_when_dry():
    from terrain import surface_water_to_rgb

    storage = np.array([[0.0, 1.0, 100.0]], dtype=np.float32)
    rgb, alpha = surface_water_to_rgb(storage)
    assert rgb.shape == (1, 3, 3)
    assert alpha[0, 0] == 0.0
    assert 0.0 < alpha[0, 1] < alpha[0, 2] <= 1.0
