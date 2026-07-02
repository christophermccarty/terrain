"""test_latitude_band_regression.py — Per-latitude-band temperature and precipitation gates.

These tests capture the key latitude-band biases identified in the 6-year benchmark
run and encode them as failing tests that will pass as physics is improved.

Philosophy:
- Hard asserts: absolute floors/ceilings that indicate a broken simulation (not just
  a bias).  These are regression guards.
- xfail asserts: current known biases that require specific physics work to fix.
  They document the target and will be promoted to hard asserts once met.

Benchmark reference values (6-year run, annotated):
  Band          Sim T (°C)   Earth T (°C)   Bias
  80-90°N       -3.4         -17.4          +14.0 (NH polar warm bias — ice collapse)
  50-60°N       +9.6         +4.0           +5.6
  40-50°N       +14.8        +9.0           +5.8
  10-20°N       +27.3        +25.5          +1.8
  0-10°N        +28.6        +26.5          +2.1
  0-10°S        +28.3        +25.5          +2.8
  20-30°S       +24.0        +20.3          +3.7
  40-50°S       +7.4         +12.0          -4.6  (SH cool bias)
  50-60°S       +2.0         +7.0           -5.0  (SH cool bias)
  60-70°S       -11.2        -8.7           -2.5
  80-90°S       -26.4        -49.4          +23.0 (Antarctic warm — no real ice sheet on synth terrain)

  Band          Sim P (mm/yr)  Earth P (mm/yr)  Notes
  0-10°N        ~3200          2000             ITCZ too wet
  0-10°S        ~2500          1500             ITCZ too wet
  30-40°N       ~800           600              slightly wet
  40-50°S       ~300           800              SH precip deficit
  50-60°S       ~60            850              severe SH precip deficit
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.slow


def _lat_rows(H: int) -> np.ndarray:
    return 90.0 - (np.arange(H) + 0.5) / H * 180.0


def _row_slice(H: int, lat_n: float, lat_s: float) -> slice:
    row0 = int(H * (90.0 - lat_n) / 180.0)
    row1 = int(H * (90.0 - lat_s) / 180.0)
    return slice(max(0, row0), min(H, row1))


def _zonal_mean_T_c(state, lat_n, lat_s):
    """Zonal mean T_air (°C) for the latitude band [lat_s, lat_n]."""
    T = state.air_temperature if state.air_temperature is not None else state.temperature
    H = T.shape[0]
    return float(np.mean(T[_row_slice(H, lat_n, lat_s), :])) - 273.15


def _zonal_mean_P_mm_yr(state, lat_n, lat_s):
    """Zonal mean precipitation (mm/yr) for the latitude band [lat_s, lat_n]."""
    P = state.precipitation
    if P is None:
        return None
    H = P.shape[0]
    return float(np.mean(P[_row_slice(H, lat_n, lat_s), :])) * 365.25


# ---------------------------------------------------------------------------
# Tropical temperature (regression guards)
# ---------------------------------------------------------------------------

def test_tropical_temperature_sane(earth_spinup_state):
    """Tropics (10°S–10°N) must be 20–35°C after spinup.

    Hard floor/ceiling — outside this range the radiation balance is broken.
    Benchmark: ~28°C.
    """
    T_c = _zonal_mean_T_c(earth_spinup_state, 10, -10)
    assert 20.0 < T_c < 35.0, f"Tropical mean T = {T_c:.1f}°C (expected 20–35°C)"


def test_tropical_temperature_not_too_hot(earth_spinup_state):
    """Tropics must not exceed 31°C (Earth ~26–28°C).

    Benchmark: ~28.5°C.  31°C cap catches runaway tropical heating.
    """
    T_c = _zonal_mean_T_c(earth_spinup_state, 10, -10)
    assert T_c < 31.0, f"Tropical T too hot: {T_c:.1f}°C (cap 31°C)"


def test_tropical_temperature_target(earth_spinup_state):
    """Tropical T should be ≤27°C (Earth annual mean ~26–27°C)."""
    T_c = _zonal_mean_T_c(earth_spinup_state, 10, -10)
    assert T_c <= 27.0, f"Tropical T {T_c:.1f}°C > 27°C target"


# ---------------------------------------------------------------------------
# NH mid-latitudes temperature
# ---------------------------------------------------------------------------

def test_nh_midlat_temperature_sane(earth_spinup_state):
    """NH mid-latitudes (30–60°N) must be 0–25°C.

    Benchmark: ~12–15°C.  Regression guard against thermal collapse.
    """
    T_c = _zonal_mean_T_c(earth_spinup_state, 60, 30)
    assert 0.0 < T_c < 25.0, f"NH mid-lat T = {T_c:.1f}°C (expected 0–25°C)"


@pytest.mark.xfail(strict=False, reason="NH mid-lat currently ~+5°C warm bias; target within ±3°C of Earth")
def test_nh_midlat_temperature_bias(earth_spinup_state):
    """NH mid-lat (30–60°N) mean T should be within ±3°C of Earth (~7°C annual mean).

    Current benchmark: ~12°C → +5°C warm bias.  Likely driven by NH polar
    warm advection when ice collapses in summer.
    """
    T_c = _zonal_mean_T_c(earth_spinup_state, 60, 30)
    earth_ref = 7.0  # approximate annual mean for 30-60°N
    bias = T_c - earth_ref
    assert abs(bias) < 3.0, f"NH mid-lat T bias {bias:+.1f}°C (target ±3°C)"


# ---------------------------------------------------------------------------
# SH mid-latitudes temperature (known cold bias)
# ---------------------------------------------------------------------------

def test_sh_midlat_temperature_sane(earth_spinup_state):
    """SH mid-latitudes (30–60°S) must be −10 to 30°C.

    Regression guard.  Benchmark: ~2–21°C (warmer on synthetic terrain,
    which lacks the real Antarctic ice sheet).  30°C cap catches clearly
    broken radiation; −10°C floor catches runaway SH glaciation.
    """
    T_c = _zonal_mean_T_c(earth_spinup_state, -30, -60)
    assert -10.0 < T_c < 30.0, f"SH mid-lat T = {T_c:.1f}°C (expected −10 to 30°C)"


@pytest.mark.xfail(strict=False, reason="SH mid-lat currently ~3°C cold bias vs Earth ~10°C; needs ACC/ocean heat tuning")
def test_sh_midlat_temperature_target(earth_spinup_state):
    """SH mid-lat (30–60°S) T should be within ±3°C of Earth (~10°C annual mean).

    Benchmark 40–50°S: +7.4°C (Earth 12°C, bias −4.6°C).
    Benchmark 50–60°S: +2.0°C (Earth 7°C, bias −5.0°C).
    """
    T_c = _zonal_mean_T_c(earth_spinup_state, -30, -60)
    earth_ref = 9.5  # approximate annual mean for 30-60°S
    bias = T_c - earth_ref
    assert abs(bias) < 3.0, f"SH mid-lat T bias {bias:+.1f}°C (target ±3°C)"


# ---------------------------------------------------------------------------
# NH polar temperature (known warm bias)
# ---------------------------------------------------------------------------

def test_nh_polar_temperature_sane(earth_spinup_state):
    """NH polar cap (70–90°N) must be −40 to +10°C.

    Regression guard — above +10°C means Arctic summer T has gone runaway.
    Benchmark: ~−3°C annual mean (Earth ~−17°C → +14°C warm bias).
    """
    T_c = _zonal_mean_T_c(earth_spinup_state, 90, 70)
    assert -40.0 < T_c < 10.0, f"NH polar T = {T_c:.1f}°C (expected −40 to +10°C)"


@pytest.mark.xfail(strict=False, reason="NH polar T ~−3°C vs Earth −17°C; +14°C warm bias from ice-albedo collapse in summer")
def test_nh_polar_temperature_target(earth_spinup_state):
    """NH polar (70–90°N) annual mean should be ≤ −10°C (Earth ~−17°C).

    Current: ~−3°C.  Requires fixing NH summer ice-albedo runaway so that
    summer polar T stays cold enough to maintain the annual mean.
    """
    T_c = _zonal_mean_T_c(earth_spinup_state, 90, 70)
    assert T_c < -10.0, f"NH polar T {T_c:.1f}°C > −10°C target"


# ---------------------------------------------------------------------------
# ITCZ precipitation (currently too wet)
# ---------------------------------------------------------------------------

def test_itcz_precip_not_zero(earth_spinup_state):
    """ITCZ (10°S–10°N) must have nonzero precipitation.

    Regression guard.
    """
    P = earth_spinup_state.precipitation
    if P is None:
        pytest.skip("No precipitation in state")
    P_mm_yr = _zonal_mean_P_mm_yr(earth_spinup_state, 10, -10)
    assert P_mm_yr > 100.0, f"ITCZ precipitation nearly zero: {P_mm_yr:.0f} mm/yr"


def test_itcz_precip_upper_bound(earth_spinup_state):
    """ITCZ (10°S–10°N) must not exceed 5000 mm/yr (Earth ~2000 mm/yr).

    Current benchmark: ~2500–3200 mm/yr.  5000 mm/yr is the regression guard
    against ITCZ becoming a permanent monsoon of infinite intensity.
    """
    if earth_spinup_state.precipitation is None:
        pytest.skip("No precipitation in state")
    P_mm_yr = _zonal_mean_P_mm_yr(earth_spinup_state, 10, -10)
    assert P_mm_yr < 5000.0, f"ITCZ precipitation too large: {P_mm_yr:.0f} mm/yr (cap 5000)"


@pytest.mark.xfail(strict=False, reason="ITCZ currently ~2700 mm/yr; target ≤2200 requires conv_driver/itcz_window tuning")
def test_itcz_precip_target(earth_spinup_state):
    """ITCZ precip should be ≤2200 mm/yr (Earth ~2000 mm/yr).

    Current benchmark: ~2500–3200 mm/yr.
    """
    if earth_spinup_state.precipitation is None:
        pytest.skip("No precipitation in state")
    P_mm_yr = _zonal_mean_P_mm_yr(earth_spinup_state, 10, -10)
    assert P_mm_yr <= 2200.0, f"ITCZ precipitation {P_mm_yr:.0f} mm/yr > 2200 target"


# ---------------------------------------------------------------------------
# Subtropical precipitation (should be drier than ITCZ)
# ---------------------------------------------------------------------------

def test_subtropical_precip_drier_guard(earth_spinup_state):
    """Subtropics (20–35°) must receive less than 1500 mm/yr.

    Earth: ~400–600 mm/yr in subtropics.  1500 mm/yr is the regression guard
    (catches subtropical precipitation becoming wetter than the ITCZ).
    """
    if earth_spinup_state.precipitation is None:
        pytest.skip("No precipitation in state")
    P_nh = _zonal_mean_P_mm_yr(earth_spinup_state, 35, 20)
    P_sh = _zonal_mean_P_mm_yr(earth_spinup_state, -20, -35)
    assert P_nh < 1500.0, f"NH subtropical P too high: {P_nh:.0f} mm/yr"
    assert P_sh < 1500.0, f"SH subtropical P too high: {P_sh:.0f} mm/yr"


# ---------------------------------------------------------------------------
# SH mid-latitude precipitation deficit (known major bias)
# ---------------------------------------------------------------------------

def test_sh_midlat_precip_nonzero(earth_spinup_state):
    """SH mid-latitudes (40–65°S) must have meaningful precipitation.

    Regression guard: ≥50 mm/yr (Earth ~800 mm/yr).
    Current benchmark: ~60–300 mm/yr — barely above this floor.
    """
    if earth_spinup_state.precipitation is None:
        pytest.skip("No precipitation in state")
    P_mm_yr = _zonal_mean_P_mm_yr(earth_spinup_state, -40, -65)
    assert P_mm_yr >= 50.0, f"SH mid-lat 40–65°S precipitation nearly absent: {P_mm_yr:.0f} mm/yr"


def test_sh_midlat_precip_target(earth_spinup_state):
    """SH mid-lat (40–65°S) precipitation should reach ≥400 mm/yr (Earth ~800 mm/yr)."""
    if earth_spinup_state.precipitation is None:
        pytest.skip("No precipitation in state")
    P_mm_yr = _zonal_mean_P_mm_yr(earth_spinup_state, -40, -65)
    assert P_mm_yr >= 400.0, f"SH mid-lat 40–65°S precipitation {P_mm_yr:.0f} mm/yr < 400 target"


# ---------------------------------------------------------------------------
# Precipitation global mean sanity
# ---------------------------------------------------------------------------

def test_global_mean_precip_sane(earth_spinup_state):
    """Global mean precipitation must be 500–2500 mm/yr.

    Earth: ~990 mm/yr.  This is a water-balance sanity check; values outside
    this range indicate a fundamental problem with the moisture cycle.
    """
    P = earth_spinup_state.precipitation
    if P is None:
        pytest.skip("No precipitation in state")
    P_global = float(np.mean(P)) * 365.25
    assert 500.0 < P_global < 2500.0, (
        f"Global mean precip {P_global:.0f} mm/yr (expected 500–2500)"
    )
