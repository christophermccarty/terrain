"""test_ecs_sensitivity.py — Fast sanity check for Equilibrium Climate Sensitivity (ECS).

Earth's ECS is approximately 2.5-4.0 K per CO2 doubling (IPCC AR6 best estimate: 3.0 K).
These tests verify the model's CO2 forcing produces a physically plausible temperature
response, using a short 15yr ANNUAL run for speed (not a true equilibrium measurement).

The true ECS experiment is in scripts/run_ecs_experiment.py (50yr run).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ecs_run(co2_ppm: float, years: int = 15, spinup_years: float = 1.0,
             H: int = 32, W: int = 64):
    from optimizer.headless import run_long_simulation
    from planet_params import PlanetParams
    pp = PlanetParams(co2_initial_ppm=co2_ppm)
    _, records = run_long_simulation(pp, years=years, spinup_years=spinup_years,
                                     H=H, W=W, sample_every=5)
    return records


# ---------------------------------------------------------------------------
# Directional signal (must pass — clear warming from 2x CO2 within 15yr)
# ---------------------------------------------------------------------------

def test_ecs_positive_signal():
    """Doubling CO2 (280->560 ppm) must produce a measurable temperature increase."""
    rec_280 = _ecs_run(280.0)
    rec_560 = _ecs_run(560.0)

    T_280 = rec_280[-1]["global_mean_t"]
    T_560 = rec_560[-1]["global_mean_t"]
    dT = T_560 - T_280

    assert dT > 0.5, (
        f"2x CO2 warming signal too weak: {dT:.3f} K "
        f"(280 ppm: {T_280:.2f} K, 560 ppm: {T_560:.2f} K)"
    )


def test_ecs_no_nan():
    """Both 280 ppm and 560 ppm runs must remain NaN-free."""
    for co2 in [280.0, 560.0]:
        records = _ecs_run(co2, years=15)
        assert records, f"No records for {co2} ppm"
        r = records[-1]
        assert not r["has_nan"], f"NaN at {co2} ppm"
        assert not r["has_inf"], f"Inf at {co2} ppm"


def test_4x_co2_warmer_than_2x():
    """4x CO2 (1120 ppm) should produce a warmer climate than 2x CO2 (560 ppm)."""
    rec_2x = _ecs_run(560.0)
    rec_4x = _ecs_run(1120.0)

    T_2x = rec_2x[-1]["global_mean_t"]
    T_4x = rec_4x[-1]["global_mean_t"]

    assert T_4x > T_2x + 0.3, (
        f"4x CO2 ({T_4x:.2f} K) should be warmer than 2x CO2 ({T_2x:.2f} K)"
    )


# ---------------------------------------------------------------------------
# Magnitude plausibility (xfail — short 15yr run underestimates true ECS)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False,
                   reason="15yr run gives transient response, not equilibrium ECS; "
                          "expect 1-2.5 K, true equilibrium ECS (50yr) needed for 2.5-4 K")
def test_ecs_plausible_magnitude():
    """2x CO2 warming should approach Earth's ECS range (2.5-4.0 K) after 15yr."""
    rec_280 = _ecs_run(280.0, years=15, spinup_years=2.0)
    rec_560 = _ecs_run(560.0, years=15, spinup_years=2.0)

    T_280 = rec_280[-1]["global_mean_t"]
    T_560 = rec_560[-1]["global_mean_t"]
    dT = T_560 - T_280

    assert 2.0 <= dT <= 5.5, (
        f"ECS estimate {dT:.3f} K outside plausible range [2.0, 5.5] K "
        f"(280 ppm: {T_280:.2f} K, 560 ppm: {T_560:.2f} K)"
    )


@pytest.mark.xfail(strict=False,
                   reason="Short spinup; 560 ppm may not fully reduce ice vs 280 ppm")
def test_2x_co2_less_ice():
    """Doubled CO2 should produce less global ice after 15yr."""
    rec_280 = _ecs_run(280.0)
    rec_560 = _ecs_run(560.0)

    ice_280 = rec_280[-1]["ice_frac_global"]
    ice_560 = rec_560[-1]["ice_frac_global"]

    assert ice_560 < ice_280 - 0.005, (
        f"2x CO2 ice ({ice_560:.4f}) not meaningfully less than 1x CO2 ({ice_280:.4f})"
    )
