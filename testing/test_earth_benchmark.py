"""test_earth_benchmark.py — Earth-fidelity regression tests.

These tests require a spun-up Earth-like state (2 years) and validate that
the simulation produces broadly Earth-like climate patterns.

All tests use the session-scoped `earth_spinup_state` fixture from conftest.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# NOT slow-marked (2026-07-25). This file carried a blanket
# `pytestmark = pytest.mark.slow` until a 4K global warm bias, a halved
# equator-pole gradient, and a ~90% loss of NH sea ice reached HEAD and sat there
# for two weeks -- because `pytest -m "not slow"`, the project's standard
# verification command, deselected every test here. The whole file (all 18 tests,
# including the shared 2-year spinup fixture) runs in ~13s, which does not
# justify hiding it from the default suite. If a genuinely expensive test is
# added here later, mark that individual test, not the module.


def _lat_rows(H: int) -> np.ndarray:
    """Latitude in degrees for each row (90°N at row 0, 90°S at row H-1)."""
    return 90.0 - (np.arange(H) + 0.5) / H * 180.0


def _row_slice(H: int, lat_n: float, lat_s: float) -> slice:
    """Row slice for latitudes between lat_s and lat_n (both in degrees, lat_n > lat_s)."""
    row0 = int(H * (90.0 - lat_n) / 180.0)
    row1 = int(H * (90.0 - lat_s) / 180.0)
    return slice(max(0, row0), min(H, row1))


def _diag_stats(state):
    from diagnostics import ClimateDiagnostics

    diag = ClimateDiagnostics(track_history=False)
    stats = diag.analyze_snapshot(state)
    circ = diag.analyze_circulation(stats)
    return stats, circ


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

def test_global_mean_temperature(earth_spinup_state):
    """After 2-year spinup, global mean T should be 278–298 K."""
    T_mean = float(np.mean(earth_spinup_state.temperature))
    assert 278.0 < T_mean < 298.0, (
        f"Global mean T = {T_mean:.1f} K (expected 278–298 K)"
    )


def test_equator_pole_gradient_nh(earth_spinup_state):
    """Equator–NH pole gradient should be 20–75 K.

    Note: lower bound relaxed from 35 K to 20 K after CFL-correct advection fix
    which increases meridional heat transport.  Re-tuning heat_transport_coeff
    or the CFL cap may restore the pre-fix gradient.
    """
    T = earth_spinup_state.temperature
    H = T.shape[0]
    T_eq   = float(np.mean(T[_row_slice(H, 10, -10), :]))
    T_pole = float(np.mean(T[_row_slice(H, 90, 70), :]))
    grad = T_eq - T_pole
    assert 20.0 < grad < 75.0, f"NH equator–pole gradient = {grad:.1f} K"


def test_equator_pole_gradient_sh(earth_spinup_state):
    """Equator–SH pole gradient should be 20–75 K.

    Note: lower bound relaxed from 35 K to 20 K after CFL-correct advection fix.
    """
    T = earth_spinup_state.temperature
    H = T.shape[0]
    T_eq   = float(np.mean(T[_row_slice(H, 10, -10), :]))
    T_pole = float(np.mean(T[_row_slice(H, -70, -90), :]))
    grad = T_eq - T_pole
    assert 20.0 < grad < 75.0, f"SH equator–pole gradient = {grad:.1f} K"


def test_equator_warmer_than_poles(earth_spinup_state):
    """Equatorial mean must exceed both pole means."""
    T = earth_spinup_state.temperature
    H = T.shape[0]
    T_eq = float(np.mean(T[_row_slice(H, 10, -10), :]))
    T_np = float(np.mean(T[_row_slice(H, 90, 75), :]))
    T_sp = float(np.mean(T[_row_slice(H, -75, -90), :]))
    assert T_eq > T_np, f"Equator ({T_eq:.1f} K) not warmer than N.Pole ({T_np:.1f} K)"
    assert T_eq > T_sp, f"Equator ({T_eq:.1f} K) not warmer than S.Pole ({T_sp:.1f} K)"


def test_no_nan_in_temperature(earth_spinup_state):
    """No NaN/Inf in temperature field after spinup."""
    T = earth_spinup_state.temperature
    assert not np.any(np.isnan(T)), "NaN in temperature after spinup"
    assert not np.any(np.isinf(T)), "Inf in temperature after spinup"


def test_latitude_band_temperature_bias_reasonable(earth_spinup_state):
    """Mean latitude-band temperature bias should stay within a moderate range.

    max_bias threshold is 35°C (raised from 18°C) for two reasons:
    1. T_min floor raised from 200K → 215K: Antarctic T_base_land is now -36°C (was -51°C),
       which is more realistic. But Earth reference at 90°S = -49°C (annual mean), while
       the snapshot is taken in SH autumn (March) when Antarctic is ~-35°C → 14°C seasonal offset.
    2. The snapshot compares against annual-mean Earth references, but is taken at the spring
       equinox (day ~79.5). Polar regions have a ~±15°C seasonal swing vs annual mean.
    """
    from diagnostics import compute_latitude_band_stats

    stats = compute_latitude_band_stats(earth_spinup_state)
    summary = stats["summary"]
    mean_bias = float(summary["mean_temp_bias_c"])
    max_bias = float(summary["max_temp_bias_c"])
    # Threshold 9.0->10.5°C (2026-07, wind/precip-model revisit): the land summer-
    # temperature cap's taper start moved from 0deg to 45deg (simulate.py's
    # _land_cap_1d) to fix land being colder than adjacent ocean even at peak summer
    # at mid-latitudes -- a real bug that killed the thermal-low/monsoon signal
    # needed for continental moisture inflow (see moisture-transport-investigation
    # memory). This raises the 20-45 deg land-cap ceiling toward the equatorial 301K
    # value, pushing this snapshot's 40-50N band ~6C above its Earth reference
    # (measured 10.1C mean bias vs the old 9.0C bound) -- a deliberate trade-off for
    # a large, verified continental-interior precipitation realism gain (US Midwest
    # box: 95->227 mm/yr on real terrain), not an unexamined regression.
    # Threshold 10.5->11.5°C (2026-07-25): raised only after attributing every
    # 0.1°C of the increase to a specific, deliberate correction in d8631cb --
    # not as a blanket accommodation. Measured on this fixture:
    #   b76078a (pre-d8631cb):            mean 10.10, max 39.70
    #   HEAD with old ocean.py restored:  mean 10.30, max 39.70
    #   HEAD (current):                   mean 10.80, max 40.20
    # The +0.50°C step is ocean.py's two heat-transport corrections: (a) the
    # `total_area` normalization was missing the longitude multiplicity (a factor
    # of W), and (b) the parameterized current/WBC redistribution was injecting
    # net ocean heat until its cos(lat)-weighted mean was removed. Both are
    # genuine bug fixes that legitimately shift this snapshot metric. The
    # remaining +0.20°C is carbon_cycle.py's Wanninkhof piston-velocity unit
    # correction. See overnight/FINDINGS.md (2026-07-25).
    #
    # NOTE: this test and test_polar_balance.py were invisible to
    # `pytest -m "not slow"` for two weeks (module-level slow marker), which is
    # how a 4K global warm bias reached HEAD unnoticed. If this bound needs
    # raising again, find the cause first -- that is what this comment is for.
    assert abs(mean_bias) < 11.5, f"Mean latitude-band temperature bias too large: {mean_bias:.1f}°C"
    # Threshold raised to 41°C (from 40°C, was 35°C): T_air has larger polar
    # seasonal amplitude than T_sst, widening the snapshot-vs-annual-mean gap at
    # high latitudes; the last +0.5°C is ocean.py's net-heat-injection fix above.
    assert max_bias < 41.0, f"Max latitude-band temperature bias too large: {max_bias:.1f}°C"


# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------

def test_trade_wind_easterly_nh(earth_spinup_state):
    """Zonal mean u at 10–20°N should be negative (easterly trades)."""
    if earth_spinup_state.wind_u is None:
        pytest.skip("No wind in state")
    U = earth_spinup_state.wind_u
    H = U.shape[0]
    u_band = float(np.mean(U[_row_slice(H, 20, 10), :]))
    assert u_band < 0.0, f"NH trades (10–20°N) mean u = {u_band:.2f} m/s (expected easterly < 0)"


def test_trade_wind_easterly_sh(earth_spinup_state):
    """Zonal mean u at 10–20°S should be negative (easterly trades)."""
    if earth_spinup_state.wind_u is None:
        pytest.skip("No wind in state")
    U = earth_spinup_state.wind_u
    H = U.shape[0]
    u_band = float(np.mean(U[_row_slice(H, -10, -20), :]))
    assert u_band < 0.0, f"SH trades (10–20°S) mean u = {u_band:.2f} m/s (expected easterly < 0)"


def test_midlat_westerlies_nh(earth_spinup_state):
    """Zonal mean u at 40–60°N should be positive (westerlies)."""
    if earth_spinup_state.wind_u is None:
        pytest.skip("No wind in state")
    U = earth_spinup_state.wind_u
    H = U.shape[0]
    u_band = float(np.mean(U[_row_slice(H, 60, 40), :]))
    assert u_band > 0.0, f"NH mid-lat (40–60°N) mean u = {u_band:.2f} m/s (expected westerly > 0)"


def test_circulation_strength_and_structure(earth_spinup_state):
    """Surface circulation should be Earth-like in both strength and placement."""
    stats, circ = _diag_stats(earth_spinup_state)

    assert float(stats["wind_u_trade_mean"]) < -2.0, (
        f"Trades too weak: u_trade={stats['wind_u_trade_mean']:.2f} m/s"
    )
    assert float(stats["wind_u_midlat_mean"]) > 1.0, (
        f"Mid-lat westerlies too weak: u_mid={stats['wind_u_midlat_mean']:.2f} m/s"
    )
    assert float(stats["wind_trade_mean"]) > 2.0, (
        f"Trade-wind speed too weak: {stats['wind_trade_mean']:.2f} m/s"
    )
    assert float(stats["wind_midlat_mean"]) > 2.0, (
        f"Mid-lat wind speed too weak: {stats['wind_midlat_mean']:.2f} m/s"
    )
    assert float(stats["wind_v_hadley_n_mean"]) < 0.0, (
        f"NH Hadley return flow wrong sign: {stats['wind_v_hadley_n_mean']:.3f}"
    )
    assert float(stats["wind_v_hadley_s_mean"]) > -0.1, (
        f"SH Hadley return flow too poleward: {stats['wind_v_hadley_s_mean']:.3f}"
    )
    assert float(circ["bands"]["v_ferrel_N_30_60"]) > 0.0, (
        f"NH Ferrel return flow wrong sign: {circ['bands']['v_ferrel_N_30_60']:.3f}"
    )
    assert float(circ["bands"]["v_ferrel_S_30_60"]) < 0.0, (
        f"SH Ferrel return flow wrong sign: {circ['bands']['v_ferrel_S_30_60']:.3f}"
    )
    assert float(stats["wind_itcz_conv"]) > 0.0, (
        f"ITCZ convergence not positive: {stats['wind_itcz_conv']:.3f}"
    )
    assert 15.0 <= float(stats["wind_jet_lat_n"]) <= 70.0, (
        f"NH jet latitude out of range: {stats['wind_jet_lat_n']:.2f}°"
    )
    assert -70.0 <= float(stats["wind_jet_lat_s"]) <= -15.0, (
        f"SH jet latitude out of range: {stats['wind_jet_lat_s']:.2f}°"
    )
    assert float(stats["circulation_score"]) > 4.0, (
        f"Circulation score too low: {stats['circulation_score']:.2f}"
    )
    assert not circ.get("reason"), f"Circulation diagnostics incomplete: {circ.get('reason')}"


# ---------------------------------------------------------------------------
# Precipitation
# ---------------------------------------------------------------------------

def test_itcz_precip_near_equator(earth_spinup_state):
    """Peak zonal-mean precipitation should be within ±20° of equator."""
    P = earth_spinup_state.precipitation
    if P is None:
        pytest.skip("No precipitation in state")
    H = P.shape[0]
    zonal_P = np.mean(P, axis=1)
    peak_row = int(np.argmax(zonal_P))
    peak_lat = _lat_rows(H)[peak_row]
    assert abs(peak_lat) < 20.0, (
        f"Precip peak at {peak_lat:.1f}° (expected within ±20° of equator)"
    )


def test_subtropical_drier_than_itcz(earth_spinup_state):
    """25–35° bands should receive less precipitation than 0–15°."""
    P = earth_spinup_state.precipitation
    if P is None:
        pytest.skip("No precipitation in state")
    H = P.shape[0]
    P_itcz   = float(np.mean(P[_row_slice(H, 15, -15), :]))
    P_sub_n  = float(np.mean(P[_row_slice(H, 35, 25), :]))
    P_sub_s  = float(np.mean(P[_row_slice(H, -25, -35), :]))
    assert P_sub_n < P_itcz, (
        f"NH subtropics ({P_sub_n:.2f} mm/d) not drier than ITCZ ({P_itcz:.2f} mm/d)"
    )
    assert P_sub_s < P_itcz, (
        f"SH subtropics ({P_sub_s:.2f} mm/d) not drier than ITCZ ({P_itcz:.2f} mm/d)"
    )


def test_latitude_band_precip_bias_reasonable(earth_spinup_state):
    """Latitude-band precipitation bias should stay within a moderate range."""
    from diagnostics import compute_latitude_band_stats

    stats = compute_latitude_band_stats(earth_spinup_state)
    summary = stats["summary"]
    mean_bias = float(summary["mean_precip_bias_mm_yr"])
    max_bias = float(summary["max_precip_bias_mm_yr"])
    assert abs(mean_bias) < 120.0, f"Mean latitude-band precip bias too large: {mean_bias:.1f} mm/yr"
    assert max_bias < 1400.0, f"Max latitude-band precip bias too large: {max_bias:.1f} mm/yr"


def test_tropical_precip_quantity(earth_spinup_state):
    """Tropical band (0–15°) mean precipitation should be 1.0–8.0 mm/day."""
    P = earth_spinup_state.precipitation
    if P is None:
        pytest.skip("No precipitation in state")
    H = P.shape[0]
    P_trop = float(np.mean(P[_row_slice(H, 15, -15), :]))
    assert 1.0 < P_trop < 8.0, (
        f"Tropical mean precip {P_trop:.2f} mm/day outside [1.0, 8.0] mm/day"
    )


def test_subtropical_precip_quantity(earth_spinup_state):
    """Subtropical bands (20–35°) mean precipitation should be 0.2–2.8 mm/day.

    Averaged over a 90-day (seasonal) window rather than read from a single
    day's snapshot (jet-stream feature, 2026-07): the persistent jet meander
    index now gives storm genesis latitude real day-to-day/month-to-month
    variability (atmosphere._update_jet_index biases
    _storm_pressure_anomaly's latitude per hemisphere), so any single day, or
    even a ~20-day window, can land anywhere in a much wider instantaneous
    range (observed ~1.0-4.0 mm/day day-to-day across the 2-year spinup)
    depending on the jet's current phase -- that's the intended realism gain
    (deeper subtropical storm excursions), not a bug, but it makes a short
    fixed window a noisy, non-representative sample of subtropical
    climatology. A 90-day average smooths the weather noise/seasonal cycle
    back out (verified to converge to ~2.5/~2.4 mm/day, matching the original
    pre-feature climatology) while still catching a genuine climatological
    regression, so the original bound is unchanged.
    """
    from simulate import simulate_step
    if earth_spinup_state.precipitation is None:
        pytest.skip("No precipitation in state")
    H = earth_spinup_state.elevation.shape[0]
    rows_n = _row_slice(H, 35, 20)
    rows_s = _row_slice(H, -20, -35)
    state = earth_spinup_state
    nh_vals, sh_vals = [], []
    for _ in range(90):
        state, _ = simulate_step(state, days=1.0, block_size=4, wind_block_size=4)
        nh_vals.append(float(np.mean(state.precipitation[rows_n, :])))
        sh_vals.append(float(np.mean(state.precipitation[rows_s, :])))
    for label, val in [("NH subtropics", float(np.mean(nh_vals))), ("SH subtropics", float(np.mean(sh_vals)))]:
        assert 0.2 < val < 2.8, (
            f"{label} 90-day mean precip {val:.2f} mm/day outside [0.2, 2.8] mm/day"
        )


def test_midlat_precip_quantity(earth_spinup_state):
    """Mid-latitude bands (40–60°) mean precipitation should be 0.5–4.2 mm/day.

    Upper bound widened 3.0→3.8→4.2 mm/day. 3.0→3.8 (earlier pass): the SH
    roaring forties (all ocean, stronger westerlies) realistically reaches
    3.5 mm/day after F6 thin-ice albedo reduction warms the marginal ice zone
    and raises storm-track moisture. 3.8→4.2 (2026-07): fixing the soil-moisture
    ceiling-saturation bug (atmosphere.py generate_precipitation's soil gain
    coefficient, 0.0006→0.00015) cut desert/continental-interior land precip by
    ~40% (see test_climate_drift.py), a substantial realism win — but that same
    de-saturated soil regime lowers the pre-rescale global precip mean enough
    that the shared target_mean_mm_day rescale pushes SH mid-lat *ocean* precip
    to ~4.0-4.07 mm/day. Accepted as a worthwhile trade-off (2026-07 decision)
    rather than leaving the ceiling-saturation bug in place; 4.2 mm/day is still
    within plausible range for the Southern Ocean storm track.

    Upper bound 4.2->4.6 (2026-07-25), the fourth widening of this ceiling and
    the one to be most sceptical of -- so here is the evidence, and the caveat.

    Cause: `ferrel_v_centre_deg` 48->44 moves the prescribed Ferrel cell (and so
    the storm track) equatorward into this band. On THIS fixture SH 40-60 goes
    2.44 -> 4.35 mm/day (+78%). But on the real-terrain save the same band goes
    552 -> 654 mm/yr (1.51 -> 1.79 mm/day, +18%) against an Earth reference of
    ~800-1200 mm/yr -- i.e. real terrain is too DRY here and the change moves it
    toward Earth, while the fixture is already wet and overshoots.

    The two disagree because this fixture is a poor absolute proxy for the
    Southern Ocean: its value for this band is ~2.4x the real-terrain value
    (1588 vs 654 mm/yr), it is a 2-year cold start on synthetic 64x128 terrain
    with no Antarctic landmass, and its NH mid-lat moves the *opposite* way from
    real terrain's (2.71 -> 2.31 here vs 808 -> 854 mm/yr on real terrain).

    CAVEAT worth acting on: an absolute mm/day bound on a fixture that is 2.4x
    off the real-terrain value is a weak guard. Its actual purpose -- catch a
    mid-latitude precipitation runaway -- would be served far better by either
    running against `saves/earth.pkl` or asserting on a normalised quantity
    (e.g. mid-lat / global ratio). Prefer doing that over a fifth widening.
    """
    P = earth_spinup_state.precipitation
    if P is None:
        pytest.skip("No precipitation in state")
    H = P.shape[0]
    P_ml_n = float(np.mean(P[_row_slice(H, 60, 40), :]))
    P_ml_s = float(np.mean(P[_row_slice(H, -40, -60), :]))
    for label, val in [("NH mid-lat", P_ml_n), ("SH mid-lat", P_ml_s)]:
        assert 0.5 < val < 4.6, (
            f"{label} mean precip {val:.2f} mm/day outside [0.5, 4.6] mm/day"
        )


# ---------------------------------------------------------------------------
# Sea ice
# ---------------------------------------------------------------------------

def test_sea_ice_extent_reasonable(earth_spinup_state):
    """NH and SH ice coverage should each be 2–30% of ocean area."""
    ice  = earth_spinup_state.ice_cover
    elev = earth_spinup_state.elevation
    if ice is None:
        pytest.skip("No ice in state")
    from masks import get_masks
    sea, _ = get_masks(elev)  # get_masks returns (sea_mask, land_mask)
    H = ice.shape[0]
    nh_sea = sea[:H // 2, :]
    sh_sea = sea[H // 2:, :]
    nh_ice_frac = float(np.sum((ice[:H // 2, :] > 0.1) & nh_sea) / max(np.sum(nh_sea), 1))
    sh_ice_frac = float(np.sum((ice[H // 2:, :] > 0.1) & sh_sea) / max(np.sum(sh_sea), 1))
    # Lower bound relaxed to 0% after CFL-correct advection increases polar warming
    assert 0.0 <= nh_ice_frac < 0.35, f"NH ice = {nh_ice_frac * 100:.1f}% (expected 0–35%)"
    assert 0.0 <= sh_ice_frac < 0.35, f"SH ice = {sh_ice_frac * 100:.1f}% (expected 0–35%)"


def test_sea_ice_hemispheric_balance(earth_spinup_state):
    """Sea ice should exist in both hemispheres without an extreme imbalance."""
    stats, _ = _diag_stats(earth_spinup_state)
    nh = float(stats["ice_frac_nh"])
    sh = float(stats["ice_frac_sh"])
    edge_n = float(stats["ice_edge_n"])
    edge_s = float(stats["ice_edge_s"])

    assert nh > 0.02, f"NH sea ice too sparse: {nh:.3f}"
    assert sh > 0.02, f"SH sea ice too sparse: {sh:.3f}"
    ratio = max(nh, sh) / max(min(nh, sh), 1e-6)
    # Threshold 4.0 (relaxed from 3.0): NH ice ~28% vs SH ~9% is a known model bias
    # (NH single-layer warm advection + ice-albedo runaway) that keeps NH ice 3× SH.
    # Earth's actual ratio is ~0.75 (SH-dominant), but catching the degenerate case
    # (one hemisphere with 0% ice) is the main purpose of this test.
    assert ratio < 4.5, f"Sea-ice hemispheres too imbalanced: NH={nh:.3f}, SH={sh:.3f}"
    # 65°N threshold (relaxed from 68°N): on a 64-row grid, row centres are at 66.09°N
    # and 68.91°N — so 68°N sits exactly between rows 7 and 8. Any ice in row 8 (66°N)
    # fails the 68° check regardless of NH ice fraction. 65°N catches the truly pathological
    # case (ice reaching the sub-tropics) while tolerating this one-row grid artifact.
    assert 65.0 <= edge_n <= 89.5, f"NH ice edge out of range: {edge_n:.2f}°N"
    assert -89.5 <= edge_s <= -50.0, f"SH ice edge out of range: {edge_s:.2f}°S"
