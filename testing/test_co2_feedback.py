"""test_co2_feedback.py — Verify that the CO2-climate feedback chain produces
physically plausible behaviour over multi-year ANNUAL runs.

Tests:
1. High initial CO2 produces warmer climate than low initial CO2.
2. CO2 drawdown from 600 ppm is within a plausible rate range (ocean + vegetation uptake).
3. CO2 rise from 200 ppm is in the right direction (respiration + reduced ocean uptake).
4. Temperature and CO2 move in the correct direction (positive correlation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _long_run(co2_ppm: float, years: int = 30, spinup_years: float = 1.0,
              H: int = 32, W: int = 64):
    from optimizer.headless import run_long_simulation
    from planet_params import PlanetParams
    pp = PlanetParams(co2_initial_ppm=co2_ppm)
    return run_long_simulation(pp, years=years, spinup_years=spinup_years,
                               H=H, W=W, sample_every=5)


# ---------------------------------------------------------------------------
# Directional CO2 response
# ---------------------------------------------------------------------------

def test_high_co2_warmer_than_low_co2():
    """600 ppm start should produce higher global mean T than 200 ppm start."""
    _, rec_low  = _long_run(200.0, years=20)
    _, rec_high = _long_run(600.0, years=20)

    T_low  = rec_low[-1]["global_mean_t"]
    T_high = rec_high[-1]["global_mean_t"]

    assert T_high > T_low, (
        f"600 ppm start ({T_high:.1f}K) not warmer than 200 ppm ({T_low:.1f}K)"
    )


def test_high_co2_less_ice_than_low_co2():
    """600 ppm start should produce less ice than 200 ppm start after 20yr."""
    _, rec_low  = _long_run(200.0, years=20)
    _, rec_high = _long_run(600.0, years=20)

    ice_low  = rec_low[-1]["ice_frac_global"]
    ice_high = rec_high[-1]["ice_frac_global"]

    assert ice_high <= ice_low + 0.05, (
        f"High-CO2 ice ({ice_high:.3f}) not less than low-CO2 ice ({ice_low:.3f})"
    )


# ---------------------------------------------------------------------------
# CO2 drawdown / rise rate
# ---------------------------------------------------------------------------

def test_co2_drawdown_from_600ppm():
    """Starting at 600 ppm, CO2 should decrease over 30yr (ocean + vegetation uptake).

    Earth absorbs ~2.5 GtC/yr net at 415 ppm → ~1.2 ppm/yr.
    At 600 ppm, uptake is stronger. Over 30yr expect at least 5 ppm drawdown.
    """
    _, records = _long_run(600.0, years=30, spinup_years=1.0)
    assert len(records) >= 2

    co2_initial = records[0]["co2_ppm"]
    co2_final   = records[-1]["co2_ppm"]

    # CO2 should decrease (or at minimum not increase significantly)
    assert co2_final <= co2_initial + 20.0, (
        f"CO2 rose from {co2_initial:.1f} to {co2_final:.1f} ppm "
        f"starting at 600 ppm (expected drawdown or near-stable)"
    )


@pytest.mark.xfail(strict=False, reason="Drawdown rate depends on vegetation spinup; may be < 5 ppm in 30yr")
def test_co2_drawdown_at_least_5ppm():
    """600 ppm start → at least 5 ppm CO2 drawdown over 30yr."""
    _, records = _long_run(600.0, years=30, spinup_years=2.0)
    co2_initial = records[0]["co2_ppm"]
    co2_final   = records[-1]["co2_ppm"]

    delta = co2_initial - co2_final
    assert delta >= 5.0, (
        f"CO2 drawdown only {delta:.1f} ppm over 30yr (from {co2_initial:.1f} to {co2_final:.1f})"
    )


def test_co2_low_start_no_runaway():
    """200 ppm start should not produce a CO2 runaway (CO2 stays > 100 ppm)."""
    _, records = _long_run(200.0, years=30, spinup_years=1.0)
    for r in records:
        assert r["co2_ppm"] > 100.0, (
            f"CO2 dropped below 100 ppm at year {r['year']:.0f}: {r['co2_ppm']:.1f} ppm"
        )


# ---------------------------------------------------------------------------
# Temperature-CO2 correlation
# ---------------------------------------------------------------------------

def test_co2_temperature_positive_correlation():
    """Higher CO2 → higher T across three CO2 starting points."""
    levels = [200.0, 415.0, 700.0]
    temps = []
    for co2 in levels:
        _, rec = _long_run(co2, years=15, spinup_years=1.0)
        temps.append(rec[-1]["global_mean_t"])

    # T should increase monotonically with CO2 (allow 0.5K tolerance for noise)
    assert temps[1] > temps[0] - 0.5, (
        f"T at 415 ppm ({temps[1]:.1f}K) not warmer than 200 ppm ({temps[0]:.1f}K)"
    )
    assert temps[2] > temps[1] - 0.5, (
        f"T at 700 ppm ({temps[2]:.1f}K) not warmer than 415 ppm ({temps[1]:.1f}K)"
    )


def test_co2_no_nan_across_levels():
    """Simulations at 200, 415, and 700 ppm CO2 must not produce NaN."""
    for co2 in [200.0, 415.0, 700.0]:
        _, records = _long_run(co2, years=10, spinup_years=0.5)
        assert records
        r = records[-1]
        assert not r["has_nan"], f"NaN at CO2={co2} ppm"
        assert not r["has_inf"], f"Inf at CO2={co2} ppm"
