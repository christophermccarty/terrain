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

    storage, throughflow, ocean_outflow = route_surface_water(
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
    assert water_left + water_out == pytest.approx(water_in, rel=2e-6)
    assert float(np.max(throughflow)) > 0.0
    assert water_out > 0.0


def test_closed_depression_retains_surface_water():
    elevation = np.full((7, 9), 0.8, dtype=np.float32)
    elevation[[0, -1], :] = 0.0
    elevation[:, [0, -1]] = 0.0
    elevation[3, 4] = 0.2
    runoff = np.zeros_like(elevation)
    runoff[3, 4] = 10.0

    storage, _, ocean_outflow = route_surface_water(
        elevation, runoff, None, dt_days=1.0, routing_passes=12
    )

    assert storage[3, 4] == pytest.approx(10.0)
    assert float(np.sum(ocean_outflow)) == 0.0


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
