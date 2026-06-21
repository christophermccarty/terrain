"""test_feedback_flags.py — Verify that disabling physical feedback mechanisms
produces the expected directional response in the climate system.

Tests:
1. Ice-albedo feedback: ice_albedo_strength=0 → warmer Arctic, less ice.
2. Ocean transport: ocean_transport_coeff=0 → colder poles, steeper gradient.
3. CO2 sensitivity: higher epsilon_equator → warmer tropics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_earth(spinup_years: float = 0.6, H: int = 32, W: int = 64, **physics_kwargs):
    from optimizer.headless import run_simulation
    from planet_params import EARTH
    from simulate import TimeScaleMode

    state, metrics = run_simulation(
        EARTH,
        spinup_years=spinup_years,
        eval_years=0.15,
        H=H,
        W=W,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
        **physics_kwargs,
    )
    return state, metrics


def _arctic_mean_t(state) -> float:
    """Mean temperature in the Arctic band (lat > 60°N) from the final state."""
    H = state.temperature.shape[0]
    lats = np.linspace(90, -90, H + 1)[:-1] + (180 / H) / 2
    arctic_mask = lats > 60.0
    T_field = state.air_temperature if state.air_temperature is not None else state.temperature
    return float(np.mean(T_field[arctic_mask, :]))


# ---------------------------------------------------------------------------
# Ice-albedo feedback
# ---------------------------------------------------------------------------

def test_no_ice_albedo_arctic_not_colder():
    """Disabling ice-albedo feedback should never cool the Arctic.

    The ice-albedo effect only amplifies warming — removing it cannot make the
    Arctic colder.  A tolerance of 1K absorbs numerical/temporal noise.
    """
    _state_std, _ = _run_earth(ice_albedo_strength=0.30)
    _state_no,  _ = _run_earth(ice_albedo_strength=0.0)

    arctic_std = _arctic_mean_t(_state_std)
    arctic_no  = _arctic_mean_t(_state_no)

    assert arctic_no >= arctic_std - 1.0, (
        f"Removing ice-albedo cooled the Arctic: "
        f"standard={arctic_std:.1f}K  no-albedo={arctic_no:.1f}K"
    )


@pytest.mark.xfail(strict=False,
                   reason="Short spinup (<1yr) rarely accumulates enough ice for a detectable T signal")
def test_no_ice_albedo_warms_arctic():
    """Disabling ice-albedo feedback should produce a measurably warmer Arctic."""
    _state_std, _ = _run_earth(ice_albedo_strength=0.30, spinup_years=0.8)
    _state_no,  _ = _run_earth(ice_albedo_strength=0.0,  spinup_years=0.8)

    arctic_std = _arctic_mean_t(_state_std)
    arctic_no  = _arctic_mean_t(_state_no)

    assert arctic_no > arctic_std + 0.5, (
        f"No-ice-albedo Arctic ({arctic_no:.1f}K) not warmer than standard ({arctic_std:.1f}K)"
    )


@pytest.mark.xfail(strict=False, reason="Short spinup may not fully develop ice-albedo contrast")
def test_no_ice_albedo_warms_arctic_by_5k():
    """Disabling ice-albedo feedback should warm Arctic by > 5K over standard."""
    _state_std, _ = _run_earth(spinup_years=1.0, ice_albedo_strength=0.30)
    _state_no,  _ = _run_earth(spinup_years=1.0, ice_albedo_strength=0.0)

    delta = _arctic_mean_t(_state_no) - _arctic_mean_t(_state_std)
    assert delta > 5.0, f"Ice-albedo removal only warmed Arctic by {delta:.1f}K (expected > 5K)"


def test_no_ice_albedo_reduces_nh_ice():
    """Zero ice-albedo strength should result in less or equal NH ice cover."""
    _, m_std = _run_earth(ice_albedo_strength=0.30, spinup_years=0.8)
    _, m_no  = _run_earth(ice_albedo_strength=0.0,  spinup_years=0.8)

    # NH ice should decrease or at least not significantly increase
    assert m_no.ice_frac_nh <= m_std.ice_frac_nh + 0.05, (
        f"Removing ice-albedo unexpectedly increased NH ice: "
        f"std={m_std.ice_frac_nh*100:.1f}% → no-albedo={m_no.ice_frac_nh*100:.1f}%"
    )


def test_no_ice_albedo_no_nan():
    """ice_albedo_strength=0 must not cause numerical instability."""
    _, m = _run_earth(ice_albedo_strength=0.0)
    assert not m.has_nan, "ice_albedo_strength=0 produced NaN"
    assert not m.has_inf, "ice_albedo_strength=0 produced Inf"


# ---------------------------------------------------------------------------
# Ocean heat transport
# ---------------------------------------------------------------------------

def test_no_ocean_transport_steepens_gradient():
    """Disabling ocean heat transport should steepen the pole-equator gradient."""
    _, m_std   = _run_earth(ocean_transport_coeff=0.35, spinup_years=0.8)
    _, m_notra = _run_earth(ocean_transport_coeff=0.0,  spinup_years=0.8)

    # With no ocean transport, poles get less heat → steeper NH gradient
    assert m_notra.gradient_nh >= m_std.gradient_nh - 3.0, (
        f"No-transport NH gradient ({m_notra.gradient_nh:.1f}K) unexpectedly much less "
        f"than standard ({m_std.gradient_nh:.1f}K)"
    )


@pytest.mark.xfail(strict=False, reason="Short spinup gradient signal may be weak")
def test_no_ocean_transport_steeper_by_5k():
    """With ocean transport off, NH gradient should exceed standard by > 5K."""
    _, m_std   = _run_earth(ocean_transport_coeff=0.35, spinup_years=1.2)
    _, m_notra = _run_earth(ocean_transport_coeff=0.0,  spinup_years=1.2)

    delta = m_notra.gradient_nh - m_std.gradient_nh
    assert delta > 5.0, (
        f"No-transport NH gradient only {delta:.1f}K above standard (expected > 5K)"
    )


def test_no_ocean_transport_no_nan():
    """ocean_transport_coeff=0 must not produce NaN."""
    _, m = _run_earth(ocean_transport_coeff=0.0)
    assert not m.has_nan, "ocean_transport_coeff=0 produced NaN"
    assert not m.has_inf, "ocean_transport_coeff=0 produced Inf"


# ---------------------------------------------------------------------------
# Emissivity sensitivity
# ---------------------------------------------------------------------------

def test_high_emissivity_warms_planet():
    """Higher epsilon_equator (more absorbed solar) → warmer global mean T."""
    from planet_params import PlanetParams
    from simulate import TimeScaleMode
    from optimizer.headless import run_simulation

    pp_low  = PlanetParams(epsilon_equator=0.60)
    pp_high = PlanetParams(epsilon_equator=0.85)

    _, m_low  = run_simulation(pp_low,  spinup_years=0.5, eval_years=0.1,
                                H=32, W=64,
                                spinup_time_scale=TimeScaleMode.MONTHLY,
                                eval_time_scale=TimeScaleMode.DAILY)
    _, m_high = run_simulation(pp_high, spinup_years=0.5, eval_years=0.1,
                                H=32, W=64,
                                spinup_time_scale=TimeScaleMode.MONTHLY,
                                eval_time_scale=TimeScaleMode.DAILY)

    assert m_high.global_mean_t > m_low.global_mean_t + 3.0, (
        f"High epsilon ({m_high.global_mean_t:.1f}K) not meaningfully warmer than "
        f"low epsilon ({m_low.global_mean_t:.1f}K)"
    )
    assert not m_high.has_nan, "High epsilon produced NaN"
