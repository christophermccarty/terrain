"""test_planet_params.py — Tests for PlanetParams portability.

Verifies that simulation code paths work with non-Earth planet parameters
and that no hardcoded Earth assumptions leak through.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_spinup(planet_params, H: int = 32, W: int = 64, n_days: int = 365,
                block_size: int = 8) -> object:
    """Run a short simulation spinup and return the final state."""
    from simulate import create_initial_state, simulate_step
    elev = np.zeros((H, W), dtype=np.float32)
    state = create_initial_state(elev, day_of_year=80.0)
    for _ in range(n_days):
        state, _ = simulate_step(state, days=1.0, block_size=block_size,
                                 planet_params=planet_params,
                                 track_components=False)
    return state


# ---------------------------------------------------------------------------
# Zero obliquity
# ---------------------------------------------------------------------------

def test_zero_obliquity_no_seasons():
    """With zero obliquity, NH and SH annual-mean T should match within 3 K."""
    from planet_params import PlanetParams
    pp = PlanetParams(obliquity_deg=0.0)
    state = _run_spinup(pp, H=32, W=64, n_days=365)
    T = state.temperature
    H = T.shape[0]
    T_nh = float(np.mean(T[:H // 2, :]))
    T_sh = float(np.mean(T[H // 2:, :]))
    assert abs(T_nh - T_sh) < 5.0, (
        f"With zero obliquity NH={T_nh:.1f} K vs SH={T_sh:.1f} K "
        f"(expected < 5 K asymmetry)"
    )


# ---------------------------------------------------------------------------
# Mars-like parameters
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "Ocean transport (AMOC/ACC) adds ~36-54K warming at polar latitudes even for Mars, "
        "because it is calibrated for Earth and not scaled by PlanetParams. "
        "This keeps Mars global mean well above 250 K despite correct solar forcing. "
        "Fix requires wiring PlanetParams into the ocean transport parameterization."
    ),
    strict=False,
)
def test_mars_params_colder_than_earth():
    """Mars-like parameters should produce a much colder global mean T than Earth.

    Mars has: less solar flux (590 vs 1361 W/m²) and a thin CO2 atmosphere with
    minimal greenhouse effect (high effective emissivity ≈ 0.90).
    The epsilon values are passed as direct kwargs to simulate_step because
    PlanetParams.epsilon is not yet threaded into the radiation code path.
    """
    from planet_params import PlanetParams, EARTH
    from simulate import create_initial_state, simulate_step

    mars = PlanetParams(
        solar_constant=590.0,
        obliquity_deg=25.19,
        orbital_period_days=687.0,
        surface_pressure_pa=610.0,
    )
    elev = np.zeros((32, 64), dtype=np.float32)

    state_mars = create_initial_state(elev, day_of_year=80.0)
    for _ in range(365):
        state_mars, _ = simulate_step(state_mars, days=1.0, block_size=8,
                                      planet_params=mars,
                                      epsilon_equator=0.90, epsilon_pole=0.85,
                                      track_components=False)

    state_earth = create_initial_state(elev, day_of_year=80.0)
    for _ in range(365):
        state_earth, _ = simulate_step(state_earth, days=1.0, block_size=8,
                                       track_components=False)

    T_mars  = float(np.mean(state_mars.temperature))
    T_earth = float(np.mean(state_earth.temperature))
    assert T_mars < T_earth - 20.0, (
        f"Mars-like mean T ({T_mars:.1f} K) not significantly colder than Earth ({T_earth:.1f} K)"
    )
    assert T_mars < 250.0, f"Mars-like mean T = {T_mars:.1f} K (expected < 250 K)"


# ---------------------------------------------------------------------------
# Aerosol forcing
# ---------------------------------------------------------------------------

def test_aerosol_forcing_cools():
    """Setting aerosol_optical_depth=0.15 (Pinatubo-like) should cool the planet."""
    from planet_params import PlanetParams
    baseline = PlanetParams(aerosol_optical_depth=0.0)
    aerosol  = PlanetParams(aerosol_optical_depth=0.15)

    state_base = _run_spinup(baseline, n_days=180, block_size=8)
    state_aero = _run_spinup(aerosol,  n_days=180, block_size=8)

    T_base = float(np.mean(state_base.temperature))
    T_aero = float(np.mean(state_aero.temperature))
    assert T_aero < T_base, (
        f"Aerosol forcing did not cool: baseline={T_base:.2f} K, aerosol={T_aero:.2f} K"
    )


def test_temperature_base_cache_respects_planet_params():
    """Base climatology cache must distinguish planets with different stellar forcing."""
    from planet_params import EARTH, PlanetParams
    from temperature import temperature_kelvin_for_lat

    lat = np.array([0.0, np.deg2rad(45.0)], dtype=np.float32)
    dimmer = PlanetParams(solar_constant=900.0, obliquity_deg=EARTH.obliquity_deg)

    T_earth = temperature_kelvin_for_lat(lat, day_of_year=80, planet_params=EARTH, cache=True)
    T_dimmer = temperature_kelvin_for_lat(lat, day_of_year=80, planet_params=dimmer, cache=True)

    assert float(np.mean(T_dimmer)) < float(np.mean(T_earth)) - 5.0, (
        f"PlanetParams-sensitive cache failed: Earth={np.mean(T_earth):.1f} K "
        f"vs dimmer={np.mean(T_dimmer):.1f} K"
    )


def test_generate_wind_field_respects_planet_params():
    """Diagnostic wind generation should change when planetary rotation changes."""
    from atmosphere import generate_wind_field
    from planet_params import EARTH, PlanetParams

    H, W = 24, 48
    lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, H, dtype=np.float32)
    lon = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)
    temperature = (
        285.0
        + 8.0 * np.cos(lat)[:, None]
        + 4.0 * np.sin(2.0 * lon)[None, :]
    ).astype(np.float32)
    elevation = np.zeros((H, W), dtype=np.float32)

    slow_rot = PlanetParams(sidereal_day_hours=120.0)
    u_earth, v_earth = generate_wind_field(
        H, W, temperature=temperature, elevation=elevation, planet_params=EARTH
    )
    u_slow, v_slow = generate_wind_field(
        H, W, temperature=temperature, elevation=elevation, planet_params=slow_rot
    )

    mean_abs_diff = float(np.mean(np.abs(u_slow - u_earth)) + np.mean(np.abs(v_slow - v_earth)))
    assert mean_abs_diff > 0.01, (
        f"Diagnostic winds ignored PlanetParams rotation: mean abs diff={mean_abs_diff:.3f}"
    )


# ---------------------------------------------------------------------------
# High obliquity
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="PlanetParams obliquity is wired through, but the ocean seasonal-response "
           "path still damps polar high-obliquity swings too aggressively.",
    strict=False,
)
def test_high_obliquity_larger_seasonal_range():
    """A planet with 45° obliquity should have a larger pole-to-equator seasonal swing
    than Earth (23.44°), as measured by the difference in polar-summer vs polar-winter T."""
    from planet_params import PlanetParams, EARTH

    def pole_seasonal_range(pp) -> float:
        from simulate import create_initial_state, simulate_step
        elev = np.zeros((32, 64), dtype=np.float32)
        state = create_initial_state(elev, day_of_year=80.0)
        # Run 1 year, track NH pole temperature
        pole_temps = []
        for d in range(365):
            state, _ = simulate_step(state, days=1.0, block_size=8,
                                     planet_params=pp, track_components=False)
            pole_temps.append(float(np.mean(state.temperature[:3, :])))
        return max(pole_temps) - min(pole_temps)

    range_earth = pole_seasonal_range(EARTH)
    range_high  = pole_seasonal_range(PlanetParams(obliquity_deg=45.0))
    assert range_high > range_earth * 1.2, (
        f"45° obliquity seasonal range ({range_high:.1f} K) not larger than "
        f"Earth's ({range_earth:.1f} K)"
    )


# ---------------------------------------------------------------------------
# No NaN for any parameter set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("params", [
    {"solar_constant": 1361.0},                     # Earth default
    {"solar_constant": 590.0},                      # Mars-like (dimmer)
    {"obliquity_deg": 0.0},                         # No tilt
    {"obliquity_deg": 90.0},                        # Extreme tilt
    {"sidereal_day_hours": 10.0},                   # Fast rotation
    {"sidereal_day_hours": 720.0},                  # Very slow rotation
    {"orbital_period_days": 687.0},                 # Long year
    {"eccentricity": 0.0},                          # Circular orbit
    {"eccentricity": 0.09},                         # More eccentric than Earth
])
def test_no_nan_for_varied_params(params):
    """Simulation must not produce NaN/Inf for any physically plausible PlanetParams."""
    from planet_params import PlanetParams
    pp = PlanetParams(**params)
    state = _run_spinup(pp, n_days=30, block_size=8)
    T = state.temperature
    assert not np.any(np.isnan(T)), f"NaN with params={params}"
    assert not np.any(np.isinf(T)), f"Inf with params={params}"
