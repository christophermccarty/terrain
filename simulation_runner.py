"""Simulation orchestration helpers independent of the physics implementation."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from planet_params import EARTH
from simulation_state import PlanetState


def run_multiple_steps(
    initial_state: PlanetState,
    total_days: float,
    step_days: float,
    *,
    step_function: Callable,
    step_kwargs: dict,
) -> tuple[list[PlanetState], list[dict]]:
    """Advance in bounded increments and retain each completed snapshot."""
    states = [initial_state]
    components_list = [{}]
    current = initial_state
    n_steps = int(np.ceil(total_days / step_days))
    for _ in range(n_steps):
        dt = min(step_days, total_days - (len(states) - 1) * step_days)
        if dt <= 0:
            break
        current, components = step_function(current, days=dt, **step_kwargs)
        states.append(current)
        components_list.append(components)
    return states, components_list


def initialize_state(
    elevation: np.ndarray,
    day_of_year: float,
    *,
    step_function: Callable,
    step_kwargs: dict,
) -> PlanetState:
    """Seed planet composition and run the zero-day initialization path."""
    planet_params = step_kwargs.get("planet_params") or EARTH
    state = PlanetState(
        day_of_year=day_of_year,
        total_days=0.0,
        elevation=elevation,
        temperature=None,
        wind_u=None,
        wind_v=None,
        precipitation=None,
        humidity=None,
        co2_atmosphere=float(planet_params.co2_initial_ppm),
        ch4_atmosphere=float(planet_params.ch4_initial_ppb),
        planet_params=planet_params,
    )
    initialized, _ = step_function(state, days=0.0, **step_kwargs)
    return initialized
