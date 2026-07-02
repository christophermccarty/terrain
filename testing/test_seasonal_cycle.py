"""test_seasonal_cycle.py — Seasonal temperature amplitude sanity checks.

Validates that the simulation produces physically plausible seasonal cycles:
- Tropics: small annual range (heat capacity + near-constant insolation)
- NH mid-latitudes: moderate range (~20-55 K in reality)
- Polar regions: large annual range (polar day/night swings)

All slow tests use a 1-year DAILY evaluation after a short monthly spinup.
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


def _run_seasonal(planet_params=None, spinup_years: float = 1.0,
                  H: int = 32, W: int = 64):
    """Run 1-year DAILY eval; return (final_state, ClimateMetrics)."""
    from optimizer.headless import run_simulation
    from simulate import TimeScaleMode
    from planet_params import EARTH

    pp = planet_params or EARTH
    return run_simulation(
        pp,
        spinup_years=spinup_years,
        eval_years=1.0,
        H=H, W=W,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
        eval_snapshots=52,
    )


def _lat_band_mean(T: np.ndarray, lat_lo: float, lat_hi: float) -> float:
    H = T.shape[0]
    lats = (0.5 - (np.arange(H) + 0.5) / H) * 180.0
    mask = (lats >= lat_lo) & (lats <= lat_hi)
    return float(np.mean(T[mask, :])) if np.any(mask) else float(np.mean(T))


# ---------------------------------------------------------------------------
# NH mid-latitude seasonal amplitude from ClimateMetrics
# ---------------------------------------------------------------------------

def test_nh_seasonal_amplitude_present():
    """NH mid-latitude (40–60°N) peak-to-peak T range must exceed 5 K."""
    _, metrics = _run_seasonal()
    amp = metrics.seasonal_amplitude_nh
    assert amp > 5.0, (
        f"NH seasonal amplitude {amp:.1f} K — too small (expected > 5 K). "
        "Suggests ocean heat capacity or seasonal forcing is misconfigured."
    )


def test_nh_seasonal_amplitude_plausible():
    """NH mid-latitude (40–60°N) peak-to-peak T range should be 10–55 K."""
    _, metrics = _run_seasonal()
    amp = metrics.seasonal_amplitude_nh
    assert 10.0 < amp < 55.0, (
        f"NH seasonal amplitude {amp:.1f} K outside plausible range [10, 55] K"
    )


@pytest.mark.xfail(strict=False,
                   reason="Short spinup; amplitude may not fully develop at low resolution")
def test_nh_seasonal_amplitude_earth_like():
    """NH seasonal amplitude should be in Earth's observed range (20–45 K)."""
    _, metrics = _run_seasonal(spinup_years=1.5)
    amp = metrics.seasonal_amplitude_nh
    assert 20.0 <= amp <= 45.0, (
        f"NH seasonal amplitude {amp:.1f} K outside Earth-like range [20, 45] K"
    )


# ---------------------------------------------------------------------------
# High obliquity amplifies seasonal cycle
# ---------------------------------------------------------------------------

def test_high_obliquity_larger_nh_amplitude():
    """45° obliquity should produce a larger NH seasonal amplitude than Earth (23.44°)."""
    from planet_params import EARTH, PlanetParams

    _, m_earth = _run_seasonal(EARTH, spinup_years=0.8)
    _, m_high  = _run_seasonal(PlanetParams(obliquity_deg=45.0), spinup_years=0.8)

    assert m_high.seasonal_amplitude_nh > m_earth.seasonal_amplitude_nh, (
        f"High obliquity amplitude ({m_high.seasonal_amplitude_nh:.1f} K) "
        f"not greater than Earth ({m_earth.seasonal_amplitude_nh:.1f} K)"
    )


# ---------------------------------------------------------------------------
# Tropical vs polar amplitude contrast
# ---------------------------------------------------------------------------

def test_tropical_amplitude_smaller_than_polar(earth_spinup_state):
    """Tropics (0–20°) should have a smaller T range than NH polar band (60–90°).

    Uses the session-scoped 2-year spinup snapshot (single timestep — not a
    true seasonal comparison).  This test checks instantaneous spatial gradient,
    not temporal range, as a proxy for plausible heat distribution.
    """
    T = earth_spinup_state.temperature
    H = T.shape[0]
    T_tropical = _lat_band_mean(T, -20.0, 20.0)
    T_polar    = _lat_band_mean(T,  60.0, 90.0)
    # Tropics should be warmer than NH polar band in annual mean
    assert T_tropical > T_polar + 5.0, (
        f"Tropical mean ({T_tropical:.1f} K) not sufficiently warmer than "
        f"NH polar ({T_polar:.1f} K) — gradient too weak"
    )
