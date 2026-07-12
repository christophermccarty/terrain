"""Focused regressions for correctness defects found by the 2026-07 audit."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest


def test_salinity_brine_response_is_interval_not_dt_squared():
    from ocean import evolve_salinity

    H, W = 8, 16
    elevation = np.zeros((H, W), dtype=np.float32)
    salinity = np.full((H, W), 35.0, dtype=np.float32)
    temperature = np.full((H, W), 269.0, dtype=np.float32)
    precipitation = np.zeros((H, W), dtype=np.float32)
    interval_ice_change = np.full((H, W), 0.2, dtype=np.float32)

    one_day = evolve_salinity(
        salinity, temperature, elevation, precipitation,
        interval_ice_change, dt_days=1.0,
    )
    thirty_days = evolve_salinity(
        salinity, temperature, elevation, precipitation,
        interval_ice_change, dt_days=30.0,
    )

    # Remove the separately time-scaled E-P/restoring tendencies by comparing
    # against otherwise identical calls with no ice change.
    one_control = evolve_salinity(
        salinity, temperature, elevation, precipitation,
        np.zeros_like(interval_ice_change), dt_days=1.0,
    )
    thirty_control = evolve_salinity(
        salinity, temperature, elevation, precipitation,
        np.zeros_like(interval_ice_change), dt_days=30.0,
    )
    assert np.mean(one_day - one_control) == pytest.approx(
        float(np.mean(thirty_days - thirty_control)), rel=1e-5
    )


def test_wildfire_emissions_follow_burned_carbon_at_any_partition():
    from carbon_cycle import wildfire_dynamics

    biomass = np.full((8, 16), 5.0, dtype=np.float32)
    temperature = np.full_like(biomass, 318.0)
    precipitation = np.zeros_like(biomass)
    soil = np.zeros_like(biomass)

    _, emitted_1 = wildfire_dynamics(
        biomass, temperature, precipitation, soil, dt_days=1.0
    )
    _, emitted_4 = wildfire_dynamics(
        biomass, temperature, precipitation, soil, dt_days=4.0
    )
    # The deterministic fractional-burn model is linear below its probability
    # cap, so a four-day lump must emit four one-day increments, not sixteen.
    assert emitted_4 == pytest.approx(4.0 * emitted_1, rel=1e-6)


def test_ocean_empty_row_fallback_is_a_temperature_mean():
    from ocean import calculate_ocean_heat_transport

    H, W = 12, 24
    temperature = np.full((H, W), 280.0, dtype=np.float32)
    # Loaded-DEM convention: zero is ocean. Make one full latitude row land.
    elevation = np.zeros((H, W), dtype=np.float32)
    elevation[H // 2, :] = 0.5
    adjustment = calculate_ocean_heat_transport(
        temperature, elevation, H, W, day_of_year=80, dt_days=1.0
    )
    assert np.all(np.isfinite(adjustment))
    assert float(np.max(np.abs(adjustment))) < 1e-5


def test_grid_search_forwards_custom_planet_and_reference(monkeypatch):
    from optimizer import sweep
    from optimizer.scoring import EARTH_REFERENCE
    from planet_params import EARTH

    custom_planet = dataclasses.replace(EARTH, solar_constant=777.0)
    custom_reference = dataclasses.replace(
        EARTH_REFERENCE, global_mean_t=(210.0, 220.0, 1.0)
    )
    captured: list[tuple[float, tuple[float, float, float]]] = []

    def fake_worker(args):
        _, _, planet_kwargs, _, reference_kwargs = args
        captured.append((
            planet_kwargs["solar_constant"],
            reference_kwargs["global_mean_t"],
        ))
        return {"trial_id": 0, "score": 0.0}

    monkeypatch.setattr(sweep, "_worker", fake_worker)
    sweep.grid_search(
        {"thermal_diffusion": [0.04]},
        planet_params=custom_planet,
        reference=custom_reference,
    )
    assert captured == [(777.0, (210.0, 220.0, 1.0))]
