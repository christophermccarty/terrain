"""test_climate_drift.py — Long-horizon (decades) equilibrium regression guards.

Some biases only emerge after decades of simulated time (ice-sheet-age
hysteresis and other slow reservoirs) and are invisible to the 2-year
earth_spinup_state fixture used elsewhere in the suite. These tests use the
60-year earth_long_spinup_state fixture to catch that class of bug.

Background (2026-07 biome-map audit of a ~1100-year production run):
- 45-60°N/S land coldest-month temperatures had collapsed to -37 to -40°C
  (Earth: roughly -5 to -20°C depending on region). That crosses Koppen's Dwd
  (extreme continental) threshold of -38°C, so most of Canada/Siberia/
  Kazakhstan-latitude land was misclassified as Dwd instead of Dfb/Dfc.
- The same run showed subtropical dry-belt land precipitation (Sahara/Arabian/
  Kalahari-analogue latitudes) at 700-1200 mm/yr vs Earth's well under
  200 mm/yr for true desert, misclassifying those regions as savanna/humid
  subtropical instead of desert/steppe.
Fixed by: a mid-latitude (22-50°) storm-track land-warming term in
simulate.py (_midlat_storm_bonus_1d), and deepened subsidence/rain-shadow
precipitation suppression in atmosphere.py (subsidence_suppression,
rain_shadow_suppression).
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


def _row_slice(H: int, lat_n: float, lat_s: float) -> slice:
    row0 = int(H * (90.0 - lat_n) / 180.0)
    row1 = int(H * (90.0 - lat_s) / 180.0)
    return slice(max(0, row0), min(H, row1))


def _land_coldest_month_c(state, elevation, lat_n, lat_s):
    H = elevation.shape[0]
    rows = _row_slice(H, lat_n, lat_s)
    land = elevation[rows, :] > 0
    if land.sum() == 0 or state.monthly_temp is None:
        return None
    coldest = state.monthly_temp.min(axis=0)[rows, :][land] - 273.15
    return float(np.mean(coldest))


def _land_annual_precip_mm_yr(state, elevation, lat_n, lat_s):
    H = elevation.shape[0]
    rows = _row_slice(H, lat_n, lat_s)
    land = elevation[rows, :] > 0
    if land.sum() == 0 or state.climate_precip_avg is None:
        return None
    p = state.climate_precip_avg[rows, :][land]
    return float(np.mean(p)) * 365.25


# ---------------------------------------------------------------------------
# Mid-latitude land winter temperature (Dwd over-extension guard)
# ---------------------------------------------------------------------------

def test_nh_midlat_land_winter_not_extreme_continental(earth_long_spinup_state, mixed_elev):
    """45-65°N land coldest-month mean must stay above -35°C after a 60yr spinup.

    -38°C is Koppen's Dwd (extreme continental) threshold; in reality only
    interior Siberia gets that cold. -35°C leaves margin while still catching
    the observed collapse to -37..-40°C.
    """
    t = _land_coldest_month_c(earth_long_spinup_state, mixed_elev, 65, 45)
    if t is None:
        pytest.skip("No land in band")
    assert t > -35.0, f"NH mid-lat land coldest month = {t:.1f}C (expected > -35C)"


def test_sh_midlat_land_winter_not_extreme_continental(earth_long_spinup_state, mixed_elev):
    """45-65°S land coldest-month mean must stay above -35°C after a 60yr spinup."""
    t = _land_coldest_month_c(earth_long_spinup_state, mixed_elev, -45, -65)
    if t is None:
        pytest.skip("No land in band")
    assert t > -35.0, f"SH mid-lat land coldest month = {t:.1f}C (expected > -35C)"


# ---------------------------------------------------------------------------
# Dry-belt land precipitation (desert-vanishing guard)
# ---------------------------------------------------------------------------

def test_nh_drybelt_land_precip_desert_range(earth_long_spinup_state, mixed_elev):
    """15-30°N land-only precip should land in [5, 400] mm/yr after 60yr spinup.

    Regression guard against the observed 700-1200 mm/yr savanna-range values
    (deserts misclassified as Cfa/Aw) as well as against total desiccation.
    """
    p = _land_annual_precip_mm_yr(earth_long_spinup_state, mixed_elev, 30, 15)
    if p is None:
        pytest.skip("No land in band")
    assert 5.0 < p < 400.0, f"NH dry-belt land precip {p:.0f} mm/yr outside [5, 400]"


def test_sh_drybelt_land_precip_desert_range(earth_long_spinup_state, mixed_elev):
    """15-30°S land-only precip should land in [5, 400] mm/yr after 60yr spinup."""
    p = _land_annual_precip_mm_yr(earth_long_spinup_state, mixed_elev, -15, -30)
    if p is None:
        pytest.skip("No land in band")
    assert 5.0 < p < 400.0, f"SH dry-belt land precip {p:.0f} mm/yr outside [5, 400]"
