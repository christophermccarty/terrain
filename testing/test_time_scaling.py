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


def _run(scale_name: str, spinup_years: float = 0.5, H: int = 32, W: int = 64, planet_params=None):
    from optimizer.headless import run_simulation
    from planet_params import EARTH
    from simulate import TimeScaleMode

    scale_map = {
        "daily":   TimeScaleMode.DAILY,
        "weekly":  TimeScaleMode.WEEKLY,
        "monthly": TimeScaleMode.MONTHLY,
    }
    _, metrics = run_simulation(
        planet_params if planet_params is not None else EARTH,
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
# Opt-in gate: PlanetParams.wind_prognostic_substep_days (default 0.0/off)
# swaps MONTHLY/ANNUAL's diagnostic wind for the real prognostic evolve_wind
# path, sub-stepped internally. Default-off must stay a no-op (covered by
# every other test above, all of which use EARTH unmodified); when opted in,
# MONTHLY should track DAILY *more* closely than the default diagnostic path,
# since both are now the same wind solver rather than structurally different
# ones. See PlanetParams.wind_prognostic_substep_days docstring.
# ---------------------------------------------------------------------------

def test_wind_prognostic_gate_off_matches_default():
    """wind_prognostic_substep_days=0.0 (the field's default) must be an exact
    no-op — same result as not setting it at all."""
    import dataclasses
    from planet_params import EARTH

    pp_explicit_off = dataclasses.replace(EARTH, wind_prognostic_substep_days=0.0)
    m_default = _run("monthly", spinup_years=0.3)
    m_explicit_off = _run("monthly", spinup_years=0.3, planet_params=pp_explicit_off)

    assert m_default.global_mean_t == m_explicit_off.global_mean_t
    assert m_default.mean_precip == m_explicit_off.mean_precip


def test_wind_prognostic_gate_tightens_monthly_daily_agreement():
    """Opting into wind_prognostic_substep_days should bring MONTHLY's global
    mean T closer to DAILY's than the default diagnostic-wind path does --
    quantifying that the gate is actually closing the cross-speed divergence,
    not just changing the numbers arbitrarily."""
    import dataclasses
    from planet_params import EARTH

    pp_gate_on = dataclasses.replace(EARTH, wind_prognostic_substep_days=1.0)

    m_daily = _run("daily", spinup_years=0.5)
    m_monthly_off = _run("monthly", spinup_years=0.5)
    m_monthly_on = _run("monthly", spinup_years=0.5, planet_params=pp_gate_on)

    diff_off = abs(m_monthly_off.global_mean_t - m_daily.global_mean_t)
    diff_on = abs(m_monthly_on.global_mean_t - m_daily.global_mean_t)

    assert not m_monthly_on.has_nan and not m_monthly_on.has_inf
    assert diff_on < diff_off, (
        f"Expected wind_prognostic_substep_days=1.0 to tighten MONTHLY/DAILY "
        f"agreement: gate-off diff={diff_off:.2f}K, gate-on diff={diff_on:.2f}K "
        f"(daily={m_daily.global_mean_t:.1f}K, monthly_off={m_monthly_off.global_mean_t:.1f}K, "
        f"monthly_on={m_monthly_on.global_mean_t:.1f}K)"
    )


# ---------------------------------------------------------------------------
# Opt-in gate: PlanetParams.precip_substep_days (default 0.0/off) overrides
# _generate_precipitation_substepped's hardcoded 8.0-day chunk threshold.
# Default-off must stay a no-op; when opted in, MONTHLY's mean_precip should
# track DAILY's much more closely, since generate_precipitation's internal
# dt_evap/remove_frac caps (tuned for ~1-2 day calls) stop being silently
# violated by a too-large single call. See PlanetParams.precip_substep_days
# docstring.
# ---------------------------------------------------------------------------

def test_precip_substep_gate_off_matches_default():
    """precip_substep_days=0.0 (the field's default) must be an exact no-op —
    same result as not setting it at all."""
    import dataclasses
    from planet_params import EARTH

    pp_explicit_off = dataclasses.replace(EARTH, precip_substep_days=0.0)
    m_default = _run("monthly", spinup_years=0.3)
    m_explicit_off = _run("monthly", spinup_years=0.3, planet_params=pp_explicit_off)

    assert m_default.global_mean_t == m_explicit_off.global_mean_t
    assert m_default.mean_precip == m_explicit_off.mean_precip


def test_precip_substep_gate_tightens_monthly_daily_precip_agreement():
    """Opting into precip_substep_days should bring MONTHLY's mean_precip
    much closer to DAILY's than the default (chunked-snapshot) path does --
    this is the metric precip_substep_days directly targets, unlike the
    wind gate's global-mean-T proxy."""
    import dataclasses
    from planet_params import EARTH

    pp_gate_on = dataclasses.replace(EARTH, precip_substep_days=1.0)

    m_daily = _run("daily", spinup_years=0.5)
    m_monthly_off = _run("monthly", spinup_years=0.5)
    m_monthly_on = _run("monthly", spinup_years=0.5, planet_params=pp_gate_on)

    diff_off = abs(m_monthly_off.mean_precip - m_daily.mean_precip)
    diff_on = abs(m_monthly_on.mean_precip - m_daily.mean_precip)

    assert not m_monthly_on.has_nan and not m_monthly_on.has_inf
    assert diff_on < diff_off, (
        f"Expected precip_substep_days=1.0 to tighten MONTHLY/DAILY mean_precip "
        f"agreement: gate-off diff={diff_off:.3f}, gate-on diff={diff_on:.3f} "
        f"mm/day (daily={m_daily.mean_precip:.3f}, monthly_off={m_monthly_off.mean_precip:.3f}, "
        f"monthly_on={m_monthly_on.mean_precip:.3f})"
    )


# ---------------------------------------------------------------------------
# Opt-in gate: PlanetParams.temperature_substep_days (default 0.0/off) splits
# _evolve_temperature's large outer `days` span into repeated inner calls
# advancing day_of_year each time, instead of one call sampling the seasonal
# cycle at a single day_of_year snapshot. Default-off must stay a no-op.
#
# Unlike wind_prognostic_substep_days/precip_substep_days, enabling this gate
# does NOT bring MONTHLY closer to DAILY -- a real-terrain trace found it
# overshoots MONTHLY's winter cold bias instead (root cause: T_base_land is
# computed once per outer call and stays stale across inner substeps -- see
# PlanetParams.temperature_substep_days docstring for the full diagnosis).
# So this section only asserts the no-op contract and basic stability, not a
# convergence claim.
# ---------------------------------------------------------------------------

def test_temperature_substep_gate_off_matches_default():
    """temperature_substep_days=0.0 (the field's default) must be an exact
    no-op — same result as not setting it at all."""
    import dataclasses
    from planet_params import EARTH

    pp_explicit_off = dataclasses.replace(EARTH, temperature_substep_days=0.0)
    m_default = _run("monthly", spinup_years=0.3)
    m_explicit_off = _run("monthly", spinup_years=0.3, planet_params=pp_explicit_off)

    assert m_default.global_mean_t == m_explicit_off.global_mean_t
    assert m_default.mean_precip == m_explicit_off.mean_precip


def test_temperature_substep_gate_on_stable():
    """temperature_substep_days=1.0 must stay numerically stable (no NaN/Inf)
    even though it's not currently a net improvement -- see module docstring
    above and PlanetParams.temperature_substep_days for why it's not yet
    recommended for use."""
    import dataclasses
    from planet_params import EARTH

    pp_gate_on = dataclasses.replace(EARTH, temperature_substep_days=1.0)
    m_monthly_on = _run("monthly", spinup_years=0.5, planet_params=pp_gate_on)

    assert not m_monthly_on.has_nan and not m_monthly_on.has_inf


# ---------------------------------------------------------------------------
# Precipitation: non-zero in all modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["daily", "weekly", "monthly"])
def test_mean_precip_positive(mode):
    m = _run(mode, spinup_years=0.3)
    assert m.mean_precip > 0.0, f"{mode.upper()} produced zero precipitation"
