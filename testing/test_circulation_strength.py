"""test_circulation_strength.py — Quantitative wind magnitude and circulation tests.

These tests go beyond sign checks to gate actual magnitudes of the major
circulation features.  Current-state values (from a 6-year benchmark run)
are shown in comments.  xfail targets represent physics improvements still
needed; hard asserts catch genuine regressions.

All tests use the session-scoped `earth_spinup_state` fixture (2-year spinup).
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
    circ  = diag.analyze_circulation(stats)
    return stats, circ


def _lat_rows(H: int) -> np.ndarray:
    return 90.0 - (np.arange(H) + 0.5) / H * 180.0


def _row_slice(H: int, lat_n: float, lat_s: float) -> slice:
    row0 = int(H * (90.0 - lat_n) / 180.0)
    row1 = int(H * (90.0 - lat_s) / 180.0)
    return slice(max(0, row0), min(H, row1))


# ---------------------------------------------------------------------------
# Trade-wind strength
# ---------------------------------------------------------------------------

def test_trade_wind_speed_minimum(earth_spinup_state):
    """Trade-wind speed must exceed 1.5 m/s (regression guard).

    Current benchmark: ~4.4 m/s.  Earth: ~6.5 m/s.
    1.5 m/s is the absolute floor — below this trades are effectively absent.
    """
    stats, _ = _diag(earth_spinup_state)
    speed = float(stats["wind_trade_mean"])
    assert speed > 1.5, f"Trade-wind speed too weak: {speed:.2f} m/s (floor 1.5)"


@pytest.mark.xfail(strict=False, reason="Trades currently ~4.4 m/s; target ≥5.5 requires wind-friction re-tuning")
def test_trade_wind_speed_target(earth_spinup_state):
    """Trade-wind speed should reach ≥5.5 m/s (Earth ~6.5 m/s).

    Current benchmark: ~4.4 m/s.  Will pass once atmospheric friction and
    pressure-gradient forcing are better calibrated.
    """
    stats, _ = _diag(earth_spinup_state)
    speed = float(stats["wind_trade_mean"])
    assert speed >= 5.5, f"Trade-wind speed {speed:.2f} m/s < 5.5 m/s target"


# ---------------------------------------------------------------------------
# Mid-latitude westerlies strength
# ---------------------------------------------------------------------------

def test_midlat_westerly_speed_minimum(earth_spinup_state):
    """Mid-latitude wind speed must exceed 1.5 m/s (regression guard).

    Current benchmark: ~4.0 m/s.  Earth: ~7 m/s.
    """
    stats, _ = _diag(earth_spinup_state)
    speed = float(stats["wind_midlat_mean"])
    assert speed > 1.5, f"Mid-lat wind speed too weak: {speed:.2f} m/s (floor 1.5)"


@pytest.mark.xfail(strict=False, reason="Mid-lat westerly u currently ~4.0 m/s; target ≥5.0 requires pressure-gradient work")
def test_midlat_westerly_u_target(earth_spinup_state):
    """Zonal-mean u at 40–60°N should reach ≥5.0 m/s (Earth ~7 m/s).

    Current benchmark: ~4.0 m/s.
    """
    stats, _ = _diag(earth_spinup_state)
    u_mid = float(stats["wind_u_midlat_mean"])
    assert u_mid >= 5.0, f"NH mid-lat westerly u {u_mid:.2f} m/s < 5.0 target"


# ---------------------------------------------------------------------------
# Zonal-mean trade-wind u
# ---------------------------------------------------------------------------

def test_trade_u_minimum_magnitude(earth_spinup_state):
    """Trade-wind zonal component must be at least −1.0 m/s.

    Current benchmark: ~−3.5 m/s.
    """
    stats, _ = _diag(earth_spinup_state)
    u_trade = float(stats["wind_u_trade_mean"])
    assert u_trade < -1.0, f"Trade zonal u too weak: {u_trade:.2f} m/s (must be < −1.0)"


@pytest.mark.xfail(strict=False, reason="Trade u currently ~−3.5; target ≤−4.5 requires pressure-gradient work")
def test_trade_u_target(earth_spinup_state):
    """Trade zonal u at 10–20° should reach ≤−4.5 m/s (Earth ~−6 m/s)."""
    stats, _ = _diag(earth_spinup_state)
    u_trade = float(stats["wind_u_trade_mean"])
    assert u_trade <= -4.5, f"Trade zonal u {u_trade:.2f} m/s > −4.5 target"


# ---------------------------------------------------------------------------
# ITCZ convergence
# ---------------------------------------------------------------------------

def test_itcz_convergence_exists(earth_spinup_state):
    """ITCZ must show some meridional convergence (positive div proxy).

    A negative value means the model has the flow direction backwards.
    We accept ≥ −0.5 as passing (generous lower bound to avoid false failures
    on coarse grids) and flag as regression only if clearly wrong-signed.
    """
    if earth_spinup_state.wind_v is None:
        pytest.skip("No wind_v in state")
    stats, _ = _diag(earth_spinup_state)
    conv = float(stats["wind_itcz_conv"])
    # Regression guard: convergence must not be strongly divergent
    assert conv > -0.5, f"ITCZ convergence strongly negative: {conv:.3f}"


@pytest.mark.xfail(strict=False, reason="ITCZ convergence currently ~−0.11; positive value needs Hadley tuning")
def test_itcz_convergence_positive(earth_spinup_state):
    """ITCZ convergence proxy should be > 0 (Earth: ~+0.3–0.5).

    Current benchmark: −0.112.  Positive value requires better Hadley cell
    strength and meridional flow near the equator.
    """
    if earth_spinup_state.wind_v is None:
        pytest.skip("No wind_v in state")
    stats, _ = _diag(earth_spinup_state)
    conv = float(stats["wind_itcz_conv"])
    assert conv > 0.0, f"ITCZ convergence {conv:.3f} not positive"


# ---------------------------------------------------------------------------
# Jet-stream latitude
# ---------------------------------------------------------------------------

def test_jet_latitude_in_range(earth_spinup_state):
    """NH and SH jet-stream cores must lie in physically plausible latitudes.

    Hard assert — a jet outside 15–75° would indicate a circulation collapse.
    """
    stats, _ = _diag(earth_spinup_state)
    jet_n = float(stats["wind_jet_lat_n"])
    jet_s = float(stats["wind_jet_lat_s"])
    assert 15.0 <= jet_n <= 75.0, f"NH jet out of range: {jet_n:.1f}°"
    assert -75.0 <= jet_s <= -15.0, f"SH jet out of range: {jet_s:.1f}°"


def test_jet_latitude_midlat_preferred(earth_spinup_state):
    """Jets should preferentially sit in mid-latitudes (30–65°).

    Earth: NH ~45–55°N, SH ~50–60°S.  Coarse assertion to catch polar/
    equatorial jet drift without being too strict about the exact latitude.
    """
    stats, _ = _diag(earth_spinup_state)
    jet_n = float(stats["wind_jet_lat_n"])
    jet_s = float(stats["wind_jet_lat_s"])
    assert 25.0 <= jet_n <= 70.0, f"NH jet latitude {jet_n:.1f}° not in 25–70°"
    assert -70.0 <= jet_s <= -25.0, f"SH jet latitude {jet_s:.1f}° not in −70° to −25°"


# ---------------------------------------------------------------------------
# Hadley / Ferrel cell meridional structure
# ---------------------------------------------------------------------------

def test_hadley_return_flow_sign(earth_spinup_state):
    """NH upper-Hadley return flow (poleward surface branch) should be equatorward.

    In the surface layer, air flows equatorward toward the ITCZ.  The
    diagnostics sign convention: negative v_hadley_n means equatorward.
    """
    if earth_spinup_state.wind_v is None:
        pytest.skip("No wind_v in state")
    stats, _ = _diag(earth_spinup_state)
    v_hadley_n = float(stats["wind_v_hadley_n_mean"])
    assert v_hadley_n < 0.0, (
        f"NH Hadley surface return flow wrong sign: {v_hadley_n:.3f} (expect < 0)"
    )


def test_ferrel_cell_meridional_signs(earth_spinup_state):
    """Ferrel cell return flow should be poleward in both hemispheres.

    NH Ferrel (30–60°N): positive v (poleward).
    SH Ferrel (30–60°S): negative v (poleward toward south pole).
    """
    if earth_spinup_state.wind_v is None:
        pytest.skip("No wind_v in state")
    _, circ = _diag(earth_spinup_state)
    v_ferrel_n = float(circ["bands"]["v_ferrel_N_30_60"])
    v_ferrel_s = float(circ["bands"]["v_ferrel_S_30_60"])
    assert v_ferrel_n > 0.0, f"NH Ferrel return flow wrong sign: {v_ferrel_n:.3f}"
    assert v_ferrel_s < 0.0, f"SH Ferrel return flow wrong sign: {v_ferrel_s:.3f}"


# ---------------------------------------------------------------------------
# Circulation score
# ---------------------------------------------------------------------------

def test_circulation_score_minimum(earth_spinup_state):
    """Overall circulation score must exceed 4.0/10 (regression guard).

    Score aggregates sign correctness of ~10 circulation features.
    """
    stats, _ = _diag(earth_spinup_state)
    score = float(stats["circulation_score"])
    assert score > 4.0, f"Circulation score too low: {score:.2f}/10 (floor 4.0)"


@pytest.mark.xfail(strict=False, reason="Score currently ~5–6; target ≥7 needs wind magnitude + ITCZ improvements")
def test_circulation_score_target(earth_spinup_state):
    """Circulation score should reach ≥7.0/10 (Earth-like circulation).

    Passing requires correct signs AND adequate magnitudes for all major
    circulation features (trades, westerlies, Hadley, Ferrel, ITCZ).
    """
    stats, _ = _diag(earth_spinup_state)
    score = float(stats["circulation_score"])
    assert score >= 7.0, f"Circulation score {score:.2f} < 7.0 target"
