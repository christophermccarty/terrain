"""test_polar_balance.py — NH/SH polar asymmetry and cryosphere gates.

Tests that the two hemispheres are physically plausible relative to each
other and to Earth observations.  Known biases are encoded as xfail so they
can be tracked and promoted to hard asserts when fixed.

Key benchmarks (6-year run, synthetic terrain — no real Antarctic ice sheet):
  NH polar annual mean T_air:   −3.4°C  (Earth −17.4°C → +14°C warm bias)
  SH polar annual mean T_air:  −26.4°C  (Earth −49.4°C → +23°C warm bias on synth terrain)
  NH ice fraction 60–75°N:     ~28%     (Earth ~5–8%)
  SH ice fraction 60–75°S:     ~9%      (Earth ~10–15%)
  NH/SH ice ratio:             ~3.1     (Earth ~0.75, SH-dominant)
  NH ice edge:                 ~68°N
  SH ice edge:                 ~65°S
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


def _diag(state):
    from diagnostics import ClimateDiagnostics
    diag = ClimateDiagnostics(track_history=False)
    stats = diag.analyze_snapshot(state)
    return stats


def _row_slice(H: int, lat_n: float, lat_s: float) -> slice:
    row0 = int(H * (90.0 - lat_n) / 180.0)
    row1 = int(H * (90.0 - lat_s) / 180.0)
    return slice(max(0, row0), min(H, row1))


def _polar_T_c(state, is_nh: bool) -> float:
    """Mean T_air (°C) in the polar cap (70–90° of the chosen hemisphere)."""
    T = state.air_temperature if state.air_temperature is not None else state.temperature
    H = T.shape[0]
    if is_nh:
        sl = _row_slice(H, 90, 70)
    else:
        sl = _row_slice(H, -70, -90)
    return float(np.mean(T[sl, :])) - 273.15


# ---------------------------------------------------------------------------
# Polar temperatures — regression guards
# ---------------------------------------------------------------------------

def test_nh_polar_temperature_below_freezing(earth_spinup_state):
    """NH annual mean polar T must stay below +5°C.

    Above +5°C means the Arctic has undergone runaway warming and lost all
    sea ice permanently.  Earth: −17°C annual mean.
    Benchmark: −3°C (marginal but passing).
    """
    T_c = _polar_T_c(earth_spinup_state, is_nh=True)
    assert T_c < 5.0, f"NH polar mean T = {T_c:.1f}°C (must stay < +5°C)"


def test_sh_polar_temperature_frozen(earth_spinup_state):
    """SH annual mean polar T must stay below +10°C.

    On synthetic terrain (no real Antarctic ice sheet), the SH pole lacks
    the high-albedo ice sheet and can run significantly warmer than Earth
    (−49°C).  This regression guard catches runaway SH warming (e.g. pole
    hotter than mid-latitudes) rather than targeting Earth-like temperatures.
    When tested with a real Earth DEM + Antarctic seeding, a tighter bound
    would be appropriate.
    """
    T_c = _polar_T_c(earth_spinup_state, is_nh=False)
    assert T_c < 10.0, f"SH polar mean T = {T_c:.1f}°C (must stay < +10°C)"


def test_both_poles_colder_than_tropics(earth_spinup_state):
    """Both polar caps must be colder than the tropics (10°S–10°N).

    This is the most fundamental thermal structure requirement.
    """
    T = earth_spinup_state.air_temperature if earth_spinup_state.air_temperature is not None \
        else earth_spinup_state.temperature
    H = T.shape[0]
    T_eq  = float(np.mean(T[_row_slice(H, 10, -10), :])) - 273.15
    T_nh  = _polar_T_c(earth_spinup_state, is_nh=True)
    T_sh  = _polar_T_c(earth_spinup_state, is_nh=False)
    assert T_nh < T_eq, f"NH pole ({T_nh:.1f}°C) not colder than tropics ({T_eq:.1f}°C)"
    assert T_sh < T_eq, f"SH pole ({T_sh:.1f}°C) not colder than tropics ({T_eq:.1f}°C)"


# ---------------------------------------------------------------------------
# Equator–pole gradient
# ---------------------------------------------------------------------------

def test_nh_equator_pole_gradient_minimum(earth_spinup_state):
    """NH equator–pole gradient must be ≥15°C (regression guard).

    Below 15°C the large-scale thermal structure has collapsed.
    Benchmark: ~35–40°C.  Earth: ~45–50°C.
    """
    T = earth_spinup_state.air_temperature if earth_spinup_state.air_temperature is not None \
        else earth_spinup_state.temperature
    H = T.shape[0]
    T_eq   = float(np.mean(T[_row_slice(H, 10, -10), :]))
    T_pole = float(np.mean(T[_row_slice(H, 90, 70), :]))
    grad   = T_eq - T_pole
    assert grad > 15.0, f"NH equator–pole gradient {grad:.1f} K < 15 K floor"


def test_sh_equator_pole_gradient_minimum(earth_spinup_state):
    """SH equator–pole gradient must be ≥15°C (regression guard).

    Benchmark: ~44°C.  Earth: ~45–55°C.
    """
    T = earth_spinup_state.air_temperature if earth_spinup_state.air_temperature is not None \
        else earth_spinup_state.temperature
    H = T.shape[0]
    T_eq   = float(np.mean(T[_row_slice(H, 10, -10), :]))
    T_pole = float(np.mean(T[_row_slice(H, -70, -90), :]))
    grad   = T_eq - T_pole
    assert grad > 15.0, f"SH equator–pole gradient {grad:.1f} K < 15 K floor"


@pytest.mark.xfail(strict=False, reason="NH gradient currently ~38K; target ≥40K needs NH warm-bias fix")
def test_nh_equator_pole_gradient_target(earth_spinup_state):
    """NH equator–pole gradient should be ≥40 K (Earth ~45–50 K).

    Current: ~35–40 K.  The NH polar warm bias from summer ice collapse keeps
    the annual-mean gradient below Earth's value.
    """
    T = earth_spinup_state.air_temperature if earth_spinup_state.air_temperature is not None \
        else earth_spinup_state.temperature
    H = T.shape[0]
    T_eq   = float(np.mean(T[_row_slice(H, 10, -10), :]))
    T_pole = float(np.mean(T[_row_slice(H, 90, 70), :]))
    grad   = T_eq - T_pole
    assert grad >= 40.0, f"NH equator–pole gradient {grad:.1f} K < 40 K target"


# ---------------------------------------------------------------------------
# Sea ice extent
# ---------------------------------------------------------------------------

def test_sea_ice_present_in_both_hemispheres(earth_spinup_state):
    """Both hemispheres must have at least some sea ice (>0.5% of ocean).

    Regression guard — complete ice-free poles indicate a thermal collapse.
    """
    ice = earth_spinup_state.ice_cover
    if ice is None:
        pytest.skip("No ice_cover in state")
    stats = _diag(earth_spinup_state)
    nh = float(stats["ice_frac_nh"])
    sh = float(stats["ice_frac_sh"])
    assert nh > 0.005, f"NH sea ice nearly absent: {nh*100:.1f}%"
    assert sh > 0.005, f"SH sea ice nearly absent: {sh*100:.1f}%"


def test_sea_ice_not_covering_all_ocean(earth_spinup_state):
    """Neither hemisphere should have >60% ocean ice cover.

    Regression guard against ice-albedo runaway locking the ocean.
    Benchmark: NH ~28%, SH ~9%.
    """
    ice = earth_spinup_state.ice_cover
    if ice is None:
        pytest.skip("No ice_cover in state")
    stats = _diag(earth_spinup_state)
    nh = float(stats["ice_frac_nh"])
    sh = float(stats["ice_frac_sh"])
    assert nh < 0.60, f"NH sea ice covers {nh*100:.1f}% of ocean (ceiling 60%)"
    assert sh < 0.60, f"SH sea ice covers {sh*100:.1f}% of ocean (ceiling 60%)"


@pytest.mark.xfail(strict=False, reason="NH ice ~28% vs Earth ~5-8%; requires fixing NH summer ice-albedo collapse")
def test_nh_sea_ice_fraction_target(earth_spinup_state):
    """NH sea ice annual mean should be <15% of NH ocean (Earth ~5–8%).

    Current benchmark: ~28%.  Fixing the NH summer ice collapse (warm T_air
    melts all ice June–August) should bring this toward Earth values.
    """
    ice = earth_spinup_state.ice_cover
    if ice is None:
        pytest.skip("No ice_cover in state")
    stats = _diag(earth_spinup_state)
    nh = float(stats["ice_frac_nh"])
    assert nh < 0.15, f"NH sea ice {nh*100:.1f}% > 15% target"


def test_ice_edge_latitudes_plausible(earth_spinup_state):
    """NH and SH ice edges must be poleward of 55° (regression guard).

    Ice reaching 55° would mean polar climate zones have expanded into the
    mid-latitudes — a sign of severe ice-albedo runaway.
    """
    ice = earth_spinup_state.ice_cover
    if ice is None:
        pytest.skip("No ice_cover in state")
    stats = _diag(earth_spinup_state)
    edge_n = float(stats["ice_edge_n"])
    edge_s = float(stats["ice_edge_s"])
    assert edge_n >= 55.0, f"NH ice edge at {edge_n:.1f}°N — too far equatorward"
    assert edge_s <= -55.0, f"SH ice edge at {edge_s:.1f}°S — too far equatorward"


# ---------------------------------------------------------------------------
# NH/SH temperature symmetry
# ---------------------------------------------------------------------------

def test_hemispheric_temperature_symmetry(earth_spinup_state):
    """NH and SH global means should be within 10°C of each other.

    Earth: NH ~14°C, SH ~13°C (nearly symmetric).  A >10°C difference
    indicates a fundamental hemispheric imbalance.
    """
    T = earth_spinup_state.air_temperature if earth_spinup_state.air_temperature is not None \
        else earth_spinup_state.temperature
    H = T.shape[0]
    T_nh = float(np.mean(T[:H // 2, :])) - 273.15
    T_sh = float(np.mean(T[H // 2:, :])) - 273.15
    diff = abs(T_nh - T_sh)
    assert diff < 10.0, (
        f"Hemispheric T imbalance: NH={T_nh:.1f}°C, SH={T_sh:.1f}°C, diff={diff:.1f}°C"
    )


@pytest.mark.xfail(strict=False, reason="NH currently warmer by ~6-8°C due to ice bias; target ≤5°C difference")
def test_hemispheric_temperature_symmetry_target(earth_spinup_state):
    """NH and SH global means should be within 5°C of each other (Earth: ~1°C).

    Current: NH is ~6-8°C warmer than SH due to the NH polar warm bias.
    Fixing the NH summer ice collapse should bring this closer to parity.
    """
    T = earth_spinup_state.air_temperature if earth_spinup_state.air_temperature is not None \
        else earth_spinup_state.temperature
    H = T.shape[0]
    T_nh = float(np.mean(T[:H // 2, :])) - 273.15
    T_sh = float(np.mean(T[H // 2:, :])) - 273.15
    diff = abs(T_nh - T_sh)
    assert diff < 5.0, (
        f"Hemispheric T imbalance: NH={T_nh:.1f}°C, SH={T_sh:.1f}°C, diff={diff:.1f}°C > 5°C"
    )


# ---------------------------------------------------------------------------
# SH polar vortex (circumpolar flow)
# ---------------------------------------------------------------------------

def test_sh_polar_winds_westerly(earth_spinup_state):
    """Zonal mean u at 60–75°S must be positive (westerly — SH circumpolar flow).

    The ACC / Southern Ocean westerlies are the strongest sustained surface
    winds on Earth.  Failure here means SH circulation is inverted.
    """
    if earth_spinup_state.wind_u is None:
        pytest.skip("No wind_u in state")
    U = earth_spinup_state.wind_u
    H = U.shape[0]
    sl = _row_slice(H, -60, -75)
    u_band = float(np.mean(U[sl, :]))
    assert u_band > 0.0, f"SH circumpolar u = {u_band:.2f} m/s (must be westerly > 0)"
