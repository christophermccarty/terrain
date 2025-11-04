"""Time simulation for planet conditions.

Advances atmospheric systems (temperature, wind, precipitation) forward in time
with configurable time scales. Default unit is one day.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple
from atmosphere import generate_wind_field, generate_precipitation
from temperature import temperature_kelvin_for_lat


class PlanetState(NamedTuple):
    """Current planet state snapshot."""
    day_of_year: float  # Fractional day (0-365.2422)
    elevation: np.ndarray  # (H, W) terrain elevation [0,1]
    temperature: np.ndarray | None = None  # (H, W) temperature (K)
    wind_u: np.ndarray | None = None  # (H, W) eastward wind (m/s)
    wind_v: np.ndarray | None = None  # (H, W) northward wind (m/s)
    precipitation: np.ndarray | None = None  # (H, W) precipitation (mm/day)
    humidity: np.ndarray | None = None  # (H, W) specific humidity


def simulate_step(
    state: PlanetState,
    days: float = 1.0,
    *,
    block_size: int = 3,
    evap_coeff: float = 1.0,
    uplift_coeff: float = 1.0,
    rain_efficiency: float = 0.7,
    precip_iterations: int = 48,
    update_wind: bool = True,
    update_precip: bool = True,
) -> PlanetState:
    """Advance planet state forward by `days`.

    Updates temperature, wind, and precipitation based on new day_of_year.
    Interactions:
    - Temperature depends on insolation (day_of_year)
    - Wind depends on temperature gradients
    - Precipitation depends on wind (advection/convergence), temperature (evaporation),
      and elevation (orographic uplift)

    Args:
        state: Current planet state
        days: Time step in days (default 1.0)
        block_size: Coarse resolution for simulation (larger = faster, less accurate)
        evap_coeff: Evaporation coefficient
        uplift_coeff: Orographic uplift coefficient
        rain_efficiency: Rain efficiency
        precip_iterations: Precipitation solver iterations
        update_wind: Whether to recompute wind field
        update_precip: Whether to recompute precipitation

    Returns:
        New state with updated day_of_year and atmospheric fields
    """
    new_day = (state.day_of_year + days) % 365.2422
    H, W = state.elevation.shape
    Hc, Wc = (max(1, (H + block_size - 1) // block_size),
              max(1, (W + block_size - 1) // block_size))

    # Update temperature from insolation (always computed)
    lat = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    T_lat = temperature_kelvin_for_lat(lat, day_of_year=int(new_day))
    T_coarse = np.repeat(T_lat[:, None], Wc, axis=1).astype(np.float32)
    if block_size > 1:
        T_full = np.repeat(np.repeat(T_coarse, block_size, axis=0), block_size, axis=1)[:H, :W]
    else:
        T_full = T_coarse

    # Update wind from temperature gradients (if requested or needed for precipitation)
    if update_wind or (update_precip and (state.wind_u is None or state.wind_v is None)):
        u_coarse, v_coarse = generate_wind_field(
            Hc, Wc, day_of_year=int(new_day), block_size=1
        )
        if block_size > 1:
            u_full = np.repeat(np.repeat(u_coarse, block_size, axis=0), block_size, axis=1)[:H, :W]
            v_full = np.repeat(np.repeat(v_coarse, block_size, axis=0), block_size, axis=1)[:H, :W]
        else:
            u_full, v_full = u_coarse, v_coarse
    else:
        u_full, v_full = state.wind_u, state.wind_v

    # Update precipitation from wind, temperature, elevation (if requested)
    if update_precip:
        P_full, q_full = generate_precipitation(
            H, W, state.elevation,
            day_of_year=int(new_day),
            evap_coeff=evap_coeff,
            uplift_coeff=uplift_coeff,
            rain_efficiency=rain_efficiency,
            iterations=precip_iterations,
            block_size=block_size,
        )
    else:
        P_full, q_full = state.precipitation, state.humidity

    return PlanetState(
        day_of_year=new_day,
        elevation=state.elevation,
        temperature=T_full,
        wind_u=u_full,
        wind_v=v_full,
        precipitation=P_full,
        humidity=q_full,
    )


def simulate_multiple_steps(
    initial_state: PlanetState,
    total_days: float,
    step_days: float = 1.0,
    **kwargs,
) -> list[PlanetState]:
    """Simulate multiple steps, returning intermediate states.

    Args:
        initial_state: Starting state
        total_days: Total simulation time
        step_days: Time per step
        **kwargs: Passed to simulate_step

    Returns:
        List of states at each step (including initial)
    """
    states = [initial_state]
    current = initial_state
    n_steps = int(np.ceil(total_days / step_days))
    for _ in range(n_steps):
        dt = min(step_days, total_days - (len(states) - 1) * step_days)
        if dt <= 0:
            break
        current = simulate_step(current, days=dt, **kwargs)
        states.append(current)
    return states


def create_initial_state(
    elevation: np.ndarray,
    day_of_year: float = 80.0,
    **kwargs,
) -> PlanetState:
    """Create initial planet state from elevation map.

    Args:
        elevation: (H, W) terrain elevation [0,1]
        day_of_year: Starting day (0-365.2422)
        **kwargs: Passed to simulate_step for initial computation

    Returns:
        Initialized state with all fields computed
    """
    state = PlanetState(
        day_of_year=day_of_year,
        elevation=elevation,
        temperature=None,
        wind_u=None,
        wind_v=None,
        precipitation=None,
        humidity=None,
    )
    return simulate_step(state, days=0.0, **kwargs)

