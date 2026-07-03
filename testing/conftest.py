"""conftest.py — pytest fixtures for PlanetSim test suite.

All fixtures use synthetic elevation arrays so the test suite has zero
dependency on opensimplex or any terrain-generation library.  Core physics
tests can run with just numpy, numba, and scipy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path regardless of how pytest is invoked
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Elevation helpers
# ---------------------------------------------------------------------------

def make_ocean_elev(H: int = 64, W: int = 128) -> np.ndarray:
    """All-ocean flat elevation (loaded-DEM convention: 0.0 = ocean)."""
    return np.zeros((H, W), dtype=np.float32)


def make_mixed_elev(H: int = 64, W: int = 128, land_frac: float = 0.35) -> np.ndarray:
    """Simple synthetic elevation: ocean=0.0, land uses sin/cos harmonics.

    No opensimplex required.  Produces a plausible ~35% land fraction with
    some topographic relief so orographic effects are non-trivial.
    """
    lon = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False, dtype=np.float64)
    lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, H, dtype=np.float64)
    lon_g, lat_g = np.meshgrid(lon, lat)

    # Rough landmass pattern from low-order harmonics
    signal = (
        0.5 * np.sin(2.0 * lon_g + 0.5) * np.cos(lat_g)
        + 0.3 * np.sin(5.0 * lon_g + 1.2) * np.cos(2.0 * lat_g - 0.3)
        + 0.2 * np.cos(3.0 * lon_g - 0.8) * np.sin(lat_g + 0.7)
    )
    threshold = np.percentile(signal, (1.0 - land_frac) * 100.0)
    elev = np.where(signal > threshold, (signal - threshold) * 0.6, 0.0)
    return elev.astype(np.float32)


# ---------------------------------------------------------------------------
# Session-scoped simulation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mixed_elev() -> np.ndarray:
    """Reusable 64×128 synthetic elevation (35% land)."""
    return make_mixed_elev(64, 128)


@pytest.fixture(scope="session")
def flat_ocean_state():
    """Initial PlanetState on a 64×128 all-ocean grid, no opensimplex."""
    from simulate import create_initial_state
    elev = make_ocean_elev(64, 128)
    return create_initial_state(elev, day_of_year=80.0)


@pytest.fixture(scope="session")
def mixed_initial_state(mixed_elev):
    """Initial PlanetState on a 64×128 mixed land/ocean grid."""
    from simulate import create_initial_state
    return create_initial_state(mixed_elev, day_of_year=80.0)


@pytest.fixture(scope="session")
def earth_spinup_state(mixed_elev):
    """Earth-like state after a 2-year spinup at coarse resolution (block_size=4).

    This is slow (~seconds) but session-scoped so it runs once per test session.
    Marked with a custom marker so it can be skipped with -m 'not slow'.
    """
    from simulate import create_initial_state, simulate_step
    state = create_initial_state(mixed_elev, day_of_year=80.0)
    for _ in range(730):   # 2 years
        state, _ = simulate_step(state, days=1.0, block_size=4, wind_block_size=4)
    return state


@pytest.fixture(scope="session")
def earth_long_spinup_state(mixed_elev):
    """Earth-like state after a 60-year MONTHLY spinup at coarse resolution.

    Some biases (e.g. mid-latitude winter land temperature collapsing well below
    Koppen's Dwd threshold, or dry-belt precipitation not settling into a desert
    range) only emerge after decades of simulated time -- ice-sheet-age hysteresis
    and other slow reservoirs need tens of years to reach their own equilibrium.
    The 2-year earth_spinup_state fixture above is too short to see this class of
    bug. ~60s wall time; session-scoped and slow-marked like earth_spinup_state.
    """
    from simulate import create_initial_state, simulate_step, TimeScaleMode
    state = create_initial_state(mixed_elev, day_of_year=80.0)
    for _ in range(60 * 12):   # 60 years, monthly steps
        state, _ = simulate_step(
            state, days=30.44, block_size=4, wind_block_size=4,
            time_scale=TimeScaleMode.MONTHLY,
        )
    return state
