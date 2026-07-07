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

UPDATE (2026-07, moisture-transport-fix session): re-measured the cold-bias
guard directly on this exact fixture and found it fully resolved -- NH -15.4°C,
SH -16.6°C, both within Earth's real range. An intervening, undocumented
session apparently closed the remaining ~10-14K gap noted in an earlier
biome-map-audit note; tightened the guard below from -35°C to -25°C to match.
The desert/continental-interior precipitation gap, by contrast, is NOT
resolved -- see known-physics-gaps.md's moisture-transport investigation
(same session): a genuine attempt at a CFL-linked transport fix, plus a
precip-formula reweight toward moisture convergence, both failed to help
(the model's precip trigger depends on local RH, and land moisture is
soil-bucket-limited, so more transport exports it away faster than it can be
captured -- fixing this likely needs the wind model to have genuine
convergence over continental interior, which it may not, a larger
undertaking than a moisture or precip-formula change alone).

Also fixed (same audit, follow-up pass): continental-interior soil moisture
was collapsing to its 0.05 floor within a few decades of MONTHLY-mode
spinup, because `atmosphere.generate_precipitation`'s soil-moisture bucket
drained faster than it could be replenished at large dt -- the evaporation
drain term scaled with the full dt while the precip replenishment term
didn't scale with dt at all, and separately the drain term used the
*uncapped* dt while land_evap only actually reached the humidity field up to
dt_evap (<=1.5d). Low soil then throttled land_evap itself, starving
humidity and precip in a self-reinforcing spiral (observed: Canadian-
Prairies-latitude precip collapsing to ~12 mm/yr vs Earth's 350-450 mm/yr).
Fixed by matching both terms to the same dt basis, plus sub-stepping
generate_precipitation in ~8-day chunks at dt>8 (simulate.py
_generate_precipitation_substepped) so a month isn't rained out as a single
snapshot event.

A follow-up investigation retracted the "regional wind-speed roughly doubling
over decades" claim that used to appear here: it was a measurement artifact
from comparing two single-month snapshots 55 years apart, not a real trend
(decade-mean wind speed in that band is flat across 6 decades). See
test_circulation_strength.py for the correct zonal-mean wind metric.

KNOWN TRADE-OFF: fixing the soil-moisture floor globally overshot into the
opposite bug -- soil now saturated to its 1.0 *ceiling* almost everywhere on
land except near the poles (measured 0.96-1.00 in every non-polar latitude
band), losing essentially all spatial discrimination between wet and dry
regions and re-inflating dry-belt desert precip to ~250-460 mm/yr (vs. the
original 700-1200 mm/yr bug, still an improvement, but well short of true
desert). An attempted counter-fix (suppressing evaporation itself in
subsidence/rain-shadow zones, mirroring the precipitation suppression) was
tried and reverted at the time: it didn't recover much of the lost dryness
and it introduced a regression in test_earth_benchmark.py::
test_midlat_precip_quantity (SH mid-lat ocean precip pushed to 4.08 mm/day
via the shared target_mean_mm_day global rescale).

FIXED (2026-07, later pass): the soil gain coefficient in
`atmosphere.generate_precipitation` (0.0006 -> 0.00015) was found to sit on
one side of a sharp bifurcation in the gain/drain balance (via land_evap's
soil feedback) -- values above ~0.00025 leave soil pinned at its ceiling
with no desert improvement, values at/below 0.00015 let it properly
de-saturate and differentiate by region (drybelt land precip now ~210-460
mm/yr depending on terrain/resolution, continental-interior soil moisture
0.05-0.4 rather than uniformly ~1.0). This reproduces the same SH mid-lat
ocean rescale regression as the reverted evap-suppression attempt above, via
a different mechanism -- accepted this time as a worthwhile trade (see
test_earth_benchmark.py::test_midlat_precip_quantity's updated 4.2 mm/day
cap) given the substantial realism gain and because the underlying dynamics
are apparently genuinely bistable, not a simple tuning target. The dry-belt
bounds below stay at [5, 700] (comfortably covers the new, lower typical
values too).
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
    """45-65°N land coldest-month mean must stay above -25°C after a 60yr spinup.

    -38°C is Koppen's Dwd (extreme continental) threshold; in reality only
    interior Siberia gets that cold. This guard was originally -35°C, sized
    just to catch the original -37..-40°C bug. Tightened 2026-07 (moisture-
    transport-fix session) after directly re-measuring this exact fixture and
    finding the bias had already been fully closed by an earlier, undocumented
    session -- current value is -15.4°C (NH) / -16.6°C (SH), both comfortably
    within Earth's real range (-5 to -20°C). -25°C leaves ~10K margin below
    the measured value while being a much more meaningful regression guard
    than the old -35°C (which only ever caught total collapse).
    """
    t = _land_coldest_month_c(earth_long_spinup_state, mixed_elev, 65, 45)
    if t is None:
        pytest.skip("No land in band")
    assert t > -25.0, f"NH mid-lat land coldest month = {t:.1f}C (expected > -25C)"


def test_sh_midlat_land_winter_not_extreme_continental(earth_long_spinup_state, mixed_elev):
    """45-65°S land coldest-month mean must stay above -25°C after a 60yr spinup.

    See test_nh_midlat_land_winter_not_extreme_continental's docstring --
    same tightening, measured -16.6°C on this fixture.
    """
    t = _land_coldest_month_c(earth_long_spinup_state, mixed_elev, -45, -65)
    if t is None:
        pytest.skip("No land in band")
    assert t > -25.0, f"SH mid-lat land coldest month = {t:.1f}C (expected > -25C)"


# ---------------------------------------------------------------------------
# Dry-belt land precipitation (desert-vanishing guard)
# ---------------------------------------------------------------------------

def test_nh_drybelt_land_precip_desert_range(earth_long_spinup_state, mixed_elev):
    """15-30°N land-only precip should land in [5, 700] mm/yr after 60yr spinup.

    Regression guard against the observed 700-1200 mm/yr savanna-range values
    (deserts misclassified as Cfa/Aw) as well as against total desiccation.
    700 (not 400) mm/yr upper bound matches the existing codebase's own
    "Earth: ~400-600 mm/yr in subtropics" reference (test_latitude_band_regression.py)
    plus headroom for the synthetic (non-real-geography) terrain fixture.
    """
    p = _land_annual_precip_mm_yr(earth_long_spinup_state, mixed_elev, 30, 15)
    if p is None:
        pytest.skip("No land in band")
    assert 5.0 < p < 700.0, f"NH dry-belt land precip {p:.0f} mm/yr outside [5, 700]"


def test_sh_drybelt_land_precip_desert_range(earth_long_spinup_state, mixed_elev):
    """15-30°S land-only precip should land in [5, 700] mm/yr after 60yr spinup."""
    p = _land_annual_precip_mm_yr(earth_long_spinup_state, mixed_elev, -15, -30)
    if p is None:
        pytest.skip("No land in band")
    assert 5.0 < p < 700.0, f"SH dry-belt land precip {p:.0f} mm/yr outside [5, 700]"


# ---------------------------------------------------------------------------
# Continental-interior soil-moisture / precip desiccation-spiral guard
# ---------------------------------------------------------------------------

def _land_soil_moisture(state, elevation, lat_n, lat_s):
    H = elevation.shape[0]
    rows = _row_slice(H, lat_n, lat_s)
    land = elevation[rows, :] > 0
    if land.sum() == 0 or state.soil_moisture is None:
        return None
    return float(np.mean(state.soil_moisture[rows, :][land]))


def test_nh_midlat_soil_moisture_not_floored(earth_long_spinup_state, mixed_elev):
    """45-65°N land soil moisture must stay above its 0.05 floor after 60yr spinup.

    Direct regression guard for the desiccation-spiral bug: soil moisture
    hitting its floor throttles land_evap (0.35+0.65*soil factor), starving
    humidity and precip in a self-reinforcing loop untied to temperature.
    """
    soil = _land_soil_moisture(earth_long_spinup_state, mixed_elev, 65, 45)
    if soil is None:
        pytest.skip("No land in band")
    assert soil > 0.15, f"NH mid-lat soil moisture = {soil:.3f} (expected > 0.15, floor is 0.05)"


def test_sh_midlat_soil_moisture_not_floored(earth_long_spinup_state, mixed_elev):
    """45-65°S land soil moisture must stay above its 0.05 floor after 60yr spinup."""
    soil = _land_soil_moisture(earth_long_spinup_state, mixed_elev, -45, -65)
    if soil is None:
        pytest.skip("No land in band")
    assert soil > 0.15, f"SH mid-lat soil moisture = {soil:.3f} (expected > 0.15, floor is 0.05)"


def test_nh_midlat_land_precip_not_collapsed(earth_long_spinup_state, mixed_elev):
    """45-65°N land-only precip must exceed 30 mm/yr after 60yr spinup.

    Regression guard against the desiccation-spiral collapse (observed:
    ~12-13 mm/yr). Not asserting Earth-realism (~350-450 mm/yr) here -- the
    2026-07 soil-ceiling-saturation fix (see module docstring) improved this
    but a real gap to Earth-realistic continental-interior precip remains,
    of unidentified origin (the "wind-speed doubling" theory once suspected
    here was investigated and retracted as a measurement artifact).
    """
    p = _land_annual_precip_mm_yr(earth_long_spinup_state, mixed_elev, 65, 45)
    if p is None:
        pytest.skip("No land in band")
    assert p > 30.0, f"NH mid-lat land precip {p:.0f} mm/yr (expected > 30, desiccation-spiral guard)"


def test_sh_midlat_land_precip_not_collapsed(earth_long_spinup_state, mixed_elev):
    """45-65°S land-only precip must exceed 30 mm/yr after 60yr spinup."""
    p = _land_annual_precip_mm_yr(earth_long_spinup_state, mixed_elev, -45, -65)
    if p is None:
        pytest.skip("No land in band")
    assert p > 30.0, f"SH mid-lat land precip {p:.0f} mm/yr (expected > 30, desiccation-spiral guard)"
