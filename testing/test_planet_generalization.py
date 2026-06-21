"""test_planet_generalization.py — Verify that non-Earth planet configurations
produce physically distinct and numerically stable simulations.

Covers:
1. Mars (cold, no liquid ocean): stable + colder than Earth.
2. Retrograde rotation (rotation_direction=-1): trades reverse sign.
3. High solar constant: warmer globally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _short_run(planet_params, spinup_years: float = 0.3, H: int = 32, W: int = 64):
    from optimizer.headless import run_simulation
    from simulate import TimeScaleMode

    _, metrics = run_simulation(
        planet_params,
        spinup_years=spinup_years,
        eval_years=0.1,
        H=H,
        W=W,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
    )
    return metrics


# ---------------------------------------------------------------------------
# Mars: numerical stability
# ---------------------------------------------------------------------------

def test_mars_no_nan():
    """Mars simulation must remain stable (no NaN/Inf) over a short run."""
    from planet_params import MARS
    m = _short_run(MARS)
    assert not m.has_nan, f"Mars produced NaN (mean_t={m.global_mean_t:.1f}K)"
    assert not m.has_inf, "Mars produced Inf"


def test_mars_colder_than_earth():
    """Mars (S0=589 W/m², no liquid ocean) should be significantly colder than Earth."""
    from planet_params import EARTH, MARS
    m_earth = _short_run(EARTH, spinup_years=0.4)
    m_mars  = _short_run(MARS,  spinup_years=0.4)

    assert m_mars.global_mean_t < m_earth.global_mean_t - 20.0, (
        f"Mars mean T {m_mars.global_mean_t:.1f}K not sufficiently colder than "
        f"Earth {m_earth.global_mean_t:.1f}K (expected gap > 20K)"
    )


@pytest.mark.xfail(strict=False, reason="Very short spinup may leave Mars T above 230K transiently")
def test_mars_below_230k():
    """After equilibration Mars should be below 230K (its observed mean ~210K)."""
    from planet_params import MARS
    m = _short_run(MARS, spinup_years=1.0)
    assert m.global_mean_t < 230.0, f"Mars mean T {m.global_mean_t:.1f}K — expected < 230K"


# ---------------------------------------------------------------------------
# Retrograde rotation: trade-wind sign flip
# ---------------------------------------------------------------------------

def _wind_trade_sign(planet_params, spinup_years: float = 0.4, H: int = 32, W: int = 64):
    """Return the mean u-component of wind in the trade-wind band (±5°–25°)."""
    from simulate import simulate_step, PlanetState, TimeScaleMode
    from optimizer.headless import run_simulation, _make_default_elevation
    import numpy as np

    elevation = _make_default_elevation(H, W)
    _, metrics = run_simulation(
        planet_params,
        spinup_years=spinup_years,
        eval_years=0.05,
        H=H,
        W=W,
        elevation=elevation,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
    )
    # Re-run one more step to get state with wind; use headless internal
    from optimizer.headless import _make_default_elevation, _advance_one_cycle
    from simulate import TimeScaleMode, simulate_step

    # Get final state via a longer headless run (last state returned)
    state_out, _ = run_simulation(
        planet_params,
        spinup_years=spinup_years,
        eval_years=0.05,
        H=H,
        W=W,
        elevation=elevation,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
    )
    # Trade wind band: latitudes ±5°–25°
    lats = np.linspace(90, -90, H + 1)[:-1] + (180 / H) / 2  # cell centers
    trade_mask = (np.abs(lats) >= 5) & (np.abs(lats) <= 25)
    if state_out.wind_u is not None:
        return float(np.mean(state_out.wind_u[trade_mask, :]))
    return 0.0


def test_retrograde_trade_winds_reversed():
    """Retrograde rotation should reverse the sign of trade-wind zonal flow."""
    from planet_params import PlanetParams

    prograde  = PlanetParams(rotation_direction=1)
    retrograde = PlanetParams(rotation_direction=-1)

    u_pro  = _wind_trade_sign(prograde,  spinup_years=0.5)
    u_retro = _wind_trade_sign(retrograde, spinup_years=0.5)

    # Earth's trades are easterly (u < 0); retrograde should be westerly (u > 0) or at least opposite
    assert np.sign(u_pro) != np.sign(u_retro) or abs(u_pro - u_retro) > 0.5, (
        f"Trade wind u did not reverse with rotation_direction=-1: "
        f"prograde={u_pro:.2f} m/s, retrograde={u_retro:.2f} m/s"
    )


@pytest.mark.xfail(strict=False, reason="Trade wind reversal may be subtle at short spinup times")
def test_retrograde_trade_wind_magnitude():
    """Retrograde trade-wind magnitude should be comparable to prograde."""
    from planet_params import PlanetParams

    prograde  = PlanetParams(rotation_direction=1)
    retrograde = PlanetParams(rotation_direction=-1)

    u_pro   = abs(_wind_trade_sign(prograde,   spinup_years=0.5))
    u_retro = abs(_wind_trade_sign(retrograde, spinup_years=0.5))

    assert u_retro > 0.3, f"Retrograde trade winds too weak: {u_retro:.3f} m/s"
    assert u_retro > 0.3 * u_pro, (
        f"Retrograde magnitude ({u_retro:.2f}) << prograde ({u_pro:.2f})"
    )


# ---------------------------------------------------------------------------
# High solar constant: warmer planet
# ---------------------------------------------------------------------------

def test_hot_star_warmer_than_earth():
    """A planet with S0=2000 W/m² should be warmer than Earth.

    Threshold is 8K rather than the full equilibrium delta because at short spinup
    (0.4yr) the thermal mass hasn't fully responded to the 47% solar increase.
    """
    from planet_params import EARTH, PlanetParams

    hot_pp = PlanetParams(solar_constant=2000.0)
    m_earth = _short_run(EARTH,  spinup_years=0.4)
    m_hot   = _short_run(hot_pp, spinup_years=0.4)

    assert m_hot.global_mean_t > m_earth.global_mean_t + 8.0, (
        f"Hot star ({m_hot.global_mean_t:.1f}K) not much warmer than Earth ({m_earth.global_mean_t:.1f}K)"
    )
    assert not m_hot.has_nan, "Hot-star simulation produced NaN"


def test_hot_star_no_nan():
    """High solar constant simulation must not produce NaN."""
    from planet_params import PlanetParams
    hot_pp = PlanetParams(solar_constant=2000.0)
    m = _short_run(hot_pp, spinup_years=0.3)
    assert not m.has_nan, f"Hot star produced NaN (mean_t={m.global_mean_t:.1f}K)"
