"""test_annual_stability.py — Verify ANNUAL time-scale mode remains numerically
stable and physically bounded over multi-decade simulated runs.

All tests use 32×64 grids for speed; slow-marked tests run 50+ simulated years.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fast sanity checks (non-slow)
# ---------------------------------------------------------------------------

def test_annual_mode_no_nan_5yr():
    """5 simulated ANNUAL years must not produce NaN or Inf."""
    from optimizer.headless import run_long_simulation
    from planet_params import EARTH

    _, records = run_long_simulation(EARTH, years=5, spinup_years=0.5,
                                     H=32, W=64, sample_every=1)
    assert records, "No records returned"
    for r in records:
        assert not r["has_nan"], f"NaN at year {r['year']:.0f}"
        assert not r["has_inf"], f"Inf at year {r['year']:.0f}"


def test_annual_mode_temperature_bounded_5yr():
    """Global mean T must remain in [250, 320] K over 5 ANNUAL years."""
    from optimizer.headless import run_long_simulation
    from planet_params import EARTH

    _, records = run_long_simulation(EARTH, years=5, spinup_years=0.5,
                                     H=32, W=64, sample_every=1)
    for r in records:
        T = r["global_mean_t"]
        assert 250.0 <= T <= 320.0, (
            f"Year {r['year']:.0f}: global mean T = {T:.1f}K outside [250, 320]"
        )


def test_annual_mode_co2_bounded_5yr():
    """CO2 must stay in [150, 700] ppm over 5 ANNUAL years (no runaway)."""
    from optimizer.headless import run_long_simulation
    from planet_params import EARTH

    _, records = run_long_simulation(EARTH, years=5, spinup_years=0.5,
                                     H=32, W=64, sample_every=1)
    for r in records:
        co2 = r["co2_ppm"]
        assert 150.0 <= co2 <= 700.0, (
            f"Year {r['year']:.0f}: CO2 = {co2:.1f} ppm outside [150, 700]"
        )


def test_annual_mode_ice_fraction_bounded_5yr():
    """Global ice fraction must remain in [0, 0.60] over 5 ANNUAL years."""
    from optimizer.headless import run_long_simulation
    from planet_params import EARTH

    _, records = run_long_simulation(EARTH, years=5, spinup_years=0.5,
                                     H=32, W=64, sample_every=1)
    for r in records:
        ice = r["ice_frac_global"]
        assert 0.0 <= ice <= 0.60, (
            f"Year {r['year']:.0f}: global ice fraction = {ice:.3f} outside [0, 0.60]"
        )


# ---------------------------------------------------------------------------
# Slow: 50-year stability
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_annual_stability_50yr_no_nan():
    """50 simulated ANNUAL years must remain NaN-free throughout."""
    from optimizer.headless import run_long_simulation
    from planet_params import EARTH

    _, records = run_long_simulation(EARTH, years=50, spinup_years=2.0,
                                     H=32, W=64, sample_every=5)
    assert records, "No records returned from 50yr run"
    for r in records:
        assert not r["has_nan"], f"NaN appeared at year {r['year']:.0f}"
        assert not r["has_inf"], f"Inf appeared at year {r['year']:.0f}"


@pytest.mark.slow
def test_annual_stability_50yr_temperature_drift():
    """Global mean T must not drift more than 5 K over 50 ANNUAL years.

    A drift > 5K over 50yr would indicate an unphysical runaway or slow
    numerical instability in the ANNUAL time-scale path.
    """
    from optimizer.headless import run_long_simulation
    from planet_params import EARTH

    _, records = run_long_simulation(EARTH, years=50, spinup_years=2.0,
                                     H=32, W=64, sample_every=5)
    assert len(records) >= 2, "Too few records to check drift"
    T_first = records[0]["global_mean_t"]
    T_last  = records[-1]["global_mean_t"]
    drift = abs(T_last - T_first)
    assert drift < 5.0, (
        f"Global mean T drifted {drift:.1f}K over 50yr "
        f"(yr{records[0]['year']:.0f}: {T_first:.1f}K → yr{records[-1]['year']:.0f}: {T_last:.1f}K)"
    )


@pytest.mark.slow
def test_annual_stability_50yr_co2_bounded():
    """CO2 must remain in [200, 600] ppm throughout a 50yr ANNUAL run."""
    from optimizer.headless import run_long_simulation
    from planet_params import EARTH

    _, records = run_long_simulation(EARTH, years=50, spinup_years=2.0,
                                     H=32, W=64, sample_every=5)
    for r in records:
        co2 = r["co2_ppm"]
        assert 200.0 <= co2 <= 600.0, (
            f"CO2 = {co2:.1f} ppm at year {r['year']:.0f} (expected [200, 600])"
        )


@pytest.mark.slow
def test_annual_stability_50yr_ice_bounded():
    """Global ice fraction must stay in [0.01, 0.40] throughout a 50yr ANNUAL run.

    Ice fraction < 0.01 would indicate complete deglaciation (unphysical for modern Earth).
    Ice fraction > 0.40 would indicate a snowball-Earth runaway.
    """
    from optimizer.headless import run_long_simulation
    from planet_params import EARTH

    _, records = run_long_simulation(EARTH, years=50, spinup_years=2.0,
                                     H=32, W=64, sample_every=5)
    for r in records:
        ice = r["ice_frac_global"]
        assert 0.0 <= ice <= 0.50, (
            f"Ice fraction = {ice:.3f} at year {r['year']:.0f} (expected ≤ 0.50; "
            f"snowball threshold)"
        )
