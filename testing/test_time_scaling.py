"""test_time_scaling.py — Verify that different TimeScaleMode values produce
consistent, physically plausible climates and never diverge to NaN/Inf.

All tests use small grids (32×64) and short spinups to stay fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(scale_name: str, spinup_years: float = 0.5, H: int = 32, W: int = 64):
    from optimizer.headless import run_simulation
    from planet_params import EARTH
    from simulate import TimeScaleMode

    scale_map = {
        "daily":   TimeScaleMode.DAILY,
        "weekly":  TimeScaleMode.WEEKLY,
        "monthly": TimeScaleMode.MONTHLY,
    }
    _, metrics = run_simulation(
        EARTH,
        spinup_years=spinup_years,
        eval_years=0.1,
        H=H,
        W=W,
        spinup_time_scale=scale_map[scale_name],
        eval_time_scale=scale_map[scale_name],
    )
    return metrics


# ---------------------------------------------------------------------------
# Stability: no NaN/Inf in any mode
# ---------------------------------------------------------------------------

def test_daily_no_nan():
    m = _run("daily", spinup_years=0.2)
    assert not m.has_nan, f"DAILY produced NaN (mean_t={m.global_mean_t:.1f}K)"
    assert not m.has_inf, "DAILY produced Inf"


def test_weekly_no_nan():
    m = _run("weekly", spinup_years=0.3)
    assert not m.has_nan, f"WEEKLY produced NaN (mean_t={m.global_mean_t:.1f}K)"
    assert not m.has_inf, "WEEKLY produced Inf"


def test_monthly_no_nan():
    m = _run("monthly", spinup_years=0.5)
    assert not m.has_nan, f"MONTHLY produced NaN (mean_t={m.global_mean_t:.1f}K)"
    assert not m.has_inf, "MONTHLY produced Inf"


# ---------------------------------------------------------------------------
# Plausibility: all modes land in a reasonable temperature window
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["daily", "weekly", "monthly"])
def test_global_mean_t_in_plausible_range(mode):
    """All time-scale modes must produce a global mean T between 260K and 320K."""
    m = _run(mode, spinup_years=0.4)
    assert 260.0 <= m.global_mean_t <= 320.0, (
        f"{mode.upper()} global mean T = {m.global_mean_t:.1f}K — outside plausible window [260, 320]"
    )


# ---------------------------------------------------------------------------
# Consistency: MONTHLY and DAILY should agree within ±15K after same spinup
# ---------------------------------------------------------------------------

def test_monthly_daily_global_mean_agreement():
    """MONTHLY and DAILY spinups should converge to similar global mean T."""
    m_monthly = _run("monthly", spinup_years=0.5)
    m_daily   = _run("daily",   spinup_years=0.5)

    diff = abs(m_monthly.global_mean_t - m_daily.global_mean_t)
    assert diff < 15.0, (
        f"MONTHLY ({m_monthly.global_mean_t:.1f}K) vs DAILY ({m_daily.global_mean_t:.1f}K) "
        f"differ by {diff:.1f}K — expected < 15K"
    )


@pytest.mark.xfail(strict=False, reason="Short spinup may not equilibrate enough for <8K agreement")
def test_monthly_daily_tight_agreement():
    """After 0.5yr spinup, MONTHLY and DAILY should agree within ±8K."""
    m_monthly = _run("monthly", spinup_years=0.5)
    m_daily   = _run("daily",   spinup_years=0.5)
    diff = abs(m_monthly.global_mean_t - m_daily.global_mean_t)
    assert diff < 8.0, (
        f"MONTHLY ({m_monthly.global_mean_t:.1f}K) vs DAILY ({m_daily.global_mean_t:.1f}K) "
        f"differ by {diff:.1f}K"
    )


# ---------------------------------------------------------------------------
# Precipitation: non-zero in all modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["daily", "weekly", "monthly"])
def test_mean_precip_positive(mode):
    m = _run(mode, spinup_years=0.3)
    assert m.mean_precip > 0.0, f"{mode.upper()} produced zero precipitation"
