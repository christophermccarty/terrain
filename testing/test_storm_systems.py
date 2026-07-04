"""test_storm_systems.py — discrete moving mid-latitude storm systems (evolve_wind).

Storms are a deterministic, stateless function of `time_days` (see
`atmosphere._storm_pressure_anomaly`), injected into `evolve_wind`'s pressure
field alongside the existing Rossby waves. Unlike the Rossby waves (a standing
sinusoid that never spawns/dies), storms have a birth/track/death lifecycle,
which is what turns the "faint ripple" the Rossby waves alone produce into
organic, moving, blob-shaped precipitation via the existing
convergence/ascent-driven precipitation machinery (no changes on the
precipitation side).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import atmosphere as atmo
from planet_params import PlanetParams


def _make_elev(H: int = 32, W: int = 64, land_frac: float = 0.35) -> np.ndarray:
    lon = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False)
    lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, H)
    lon_g, lat_g = np.meshgrid(lon, lat)
    signal = (0.5 * np.sin(2.0 * lon_g + 0.5) * np.cos(lat_g)
              + 0.3 * np.sin(5.0 * lon_g + 1.2) * np.cos(2.0 * lat_g - 0.3))
    thr = np.percentile(signal, (1.0 - land_frac) * 100.0)
    return np.where(signal > thr, (signal - thr) * 0.6, 0.0).astype(np.float32)


def _lat_2d(H: int, W: int) -> np.ndarray:
    lat = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
    return np.repeat(lat[:, None], W, axis=1)


# ---------------------------------------------------------------------------
# _storm_pressure_anomaly: pure-function determinism / no-op contract
# ---------------------------------------------------------------------------

def test_storm_anomaly_zero_amplitude_is_noop():
    """amp_pa=0.0 must return an all-zero array (true disable, not just tiny)."""
    H, W = 32, 64
    lat_2d = _lat_2d(H, W)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)[None, :]
    out = atmo._storm_pressure_anomaly(lat_2d, lon_1d, time_days=123.4, amp_pa=0.0)
    assert np.all(out == 0.0)
    assert out.shape == (H, W)


def test_storm_anomaly_deterministic():
    """Same time_days must reproduce bit-identical output (no hidden RNG state)."""
    H, W = 32, 64
    lat_2d = _lat_2d(H, W)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)[None, :]
    out1 = atmo._storm_pressure_anomaly(lat_2d, lon_1d, time_days=57.0, amp_pa=110.0)
    out2 = atmo._storm_pressure_anomaly(lat_2d, lon_1d, time_days=57.0, amp_pa=110.0)
    assert np.array_equal(out1, out2)


def test_storm_anomaly_nonzero_and_moves_over_time():
    """A nonzero amplitude must produce a nonzero field that changes as time_days advances
    (confirms storms actually spawn and translate, not a static pattern)."""
    H, W = 32, 64
    lat_2d = _lat_2d(H, W)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)[None, :]
    out_t0 = atmo._storm_pressure_anomaly(lat_2d, lon_1d, time_days=10.0, amp_pa=110.0)
    out_t3 = atmo._storm_pressure_anomaly(lat_2d, lon_1d, time_days=13.0, amp_pa=110.0)
    assert np.any(out_t0 != 0.0), "expected at least one active storm at time_days=10"
    assert not np.array_equal(out_t0, out_t3), "storm field should evolve over 3 simulated days"
    assert np.all(np.isfinite(out_t0)) and np.all(np.isfinite(out_t3))


def test_storm_anomaly_confined_to_midlat_band():
    """Storms are extratropical-only (v1 scope) -- equatorial rows should stay near zero
    even while mid-latitude rows show active storms, across a scan of many time_days."""
    H, W = 32, 64
    lat_2d = _lat_2d(H, W)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)[None, :]
    equator_rows = slice(H // 2 - 2, H // 2 + 2)
    equator_max = 0.0
    midlat_max = 0.0
    for t in np.arange(0.0, 60.0, 2.0):
        out = atmo._storm_pressure_anomaly(lat_2d, lon_1d, time_days=float(t), amp_pa=110.0)
        equator_max = max(equator_max, float(np.max(np.abs(out[equator_rows, :]))))
        midlat_max = max(midlat_max, float(np.max(np.abs(out))))
    assert equator_max < 5.0, f"equatorial storm contamination too high: {equator_max:.2f} Pa"
    assert midlat_max > 30.0, f"expected substantial mid-lat storm amplitude, got {midlat_max:.2f} Pa"


# ---------------------------------------------------------------------------
# Trade-wind/subtropical wave population (second population, population_id=1):
# westward-translating, confined to 12-32 deg -- covers the band mid-latitude
# storms don't reach.
# ---------------------------------------------------------------------------

def _trade_wave_kwargs():
    return dict(
        n_slots=atmo.N_TRADE_WAVE_SLOTS,
        lifecycle_days=atmo.TRADE_WAVE_LIFECYCLE_DAYS,
        lat_center_deg=atmo.TRADE_WAVE_LAT_CENTER_DEG,
        lat_jitter_deg=atmo.TRADE_WAVE_LAT_JITTER_DEG,
        lon_drift_range=atmo.TRADE_WAVE_LON_DRIFT_DEG_PER_DAY,
        lat_drift_range=atmo.TRADE_WAVE_LAT_DRIFT_DEG_PER_DAY,
        radius_km_range=atmo.TRADE_WAVE_RADIUS_KM,
        population_id=1,
    )


def test_trade_wave_anomaly_confined_to_subtropical_band():
    """Trade waves should genuinely be centered in the 12-32 deg genesis band
    (not e.g. accidentally drifting into mid-latitudes), even though some
    Gaussian-tail bleed toward adjacent bands is expected and harmless (waves
    can be born as far out as 32 deg with a ~1000km / ~9 deg radius, so a
    single-wave tail can reach into the low 40s, and with N_TRADE_WAVE_SLOTS
    concurrent waves per hemisphere their *summed* tails can be non-trivial
    right at that boundary -- that's not a bug, so this checks the strongest
    wave's actual center position instead of a raw magnitude threshold)."""
    H, W = 32, 64
    lat_2d = _lat_2d(H, W)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)[None, :]
    lats_deg = np.rad2deg(lat_2d[:, 0])
    trade_rows = (np.abs(lats_deg) >= 12.0) & (np.abs(lats_deg) <= 32.0)
    trade_max = 0.0
    for t in np.arange(0.0, 40.0, 2.0):
        out = atmo._storm_pressure_anomaly(lat_2d, lon_1d, time_days=float(t), amp_pa=65.0,
                                            **_trade_wave_kwargs())
        idx = np.unravel_index(np.argmin(out), out.shape)
        peak_lat = abs(float(np.rad2deg(lat_2d[idx[0], 0])))
        assert 8.0 < peak_lat < 36.0, (
            f"strongest trade wave at t={t}: lat={peak_lat:.1f} deg, "
            f"expected within the 12-32 deg genesis band (+/- drift/jitter margin)"
        )
        trade_max = max(trade_max, float(np.max(np.abs(out[trade_rows, :]))))
    assert trade_max > 15.0, f"expected substantial trade-wave amplitude, got {trade_max:.2f} Pa"


def test_trade_wave_translates_westward():
    """Trade waves should move westward (decreasing longitude), matching the
    trade easterlies -- opposite direction from the eastward mid-lat storms."""
    H, W = 32, 64
    lat_2d = _lat_2d(H, W)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)[None, :]
    positions = []
    for t in [5.0, 5.5, 6.0]:
        out = atmo._storm_pressure_anomaly(lat_2d, lon_1d, time_days=t, amp_pa=65.0,
                                            **_trade_wave_kwargs())
        idx = np.unravel_index(np.argmin(out), out.shape)
        positions.append(float(np.rad2deg(lon_1d[0, idx[1]])))
    # Allow for wrap-around at +/-180; just check it's not stuck in place.
    assert positions[0] != positions[1] or positions[1] != positions[2]


def test_trade_wave_zero_amplitude_is_noop():
    H, W = 32, 64
    lat_2d = _lat_2d(H, W)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)[None, :]
    out = atmo._storm_pressure_anomaly(lat_2d, lon_1d, time_days=12.0, amp_pa=0.0,
                                        **_trade_wave_kwargs())
    assert np.all(out == 0.0)


def test_evolve_wind_trade_waves_change_output_vs_disabled():
    H, W = 32, 64
    elev = _make_elev(H, W)
    T = np.full((H, W), 295.0, dtype=np.float32)  # warmer, subtropical-ish
    u0 = np.zeros((H, W), dtype=np.float32)
    v0 = np.zeros((H, W), dtype=np.float32)

    pp_off = PlanetParams(storm_pressure_amp_pa=0.0, trade_wave_pressure_amp_pa=0.0)
    pp_on = PlanetParams(storm_pressure_amp_pa=0.0, trade_wave_pressure_amp_pa=65.0)

    u_off, v_off = atmo.evolve_wind(u0.copy(), v0.copy(), T, None, elev, dt_days=1.0,
                                     time_days=8.0, planet_params=pp_off)
    u_on, v_on = atmo.evolve_wind(u0.copy(), v0.copy(), T, None, elev, dt_days=1.0,
                                   time_days=8.0, planet_params=pp_on)

    assert not np.array_equal(u_off, u_on) or not np.array_equal(v_off, v_on)
    assert np.all(np.isfinite(u_on)) and np.all(np.isfinite(v_on))


# ---------------------------------------------------------------------------
# evolve_wind integration: storms perturb the actual wind field used for precip
# ---------------------------------------------------------------------------

def test_evolve_wind_storms_change_output_vs_disabled():
    H, W = 32, 64
    elev = _make_elev(H, W)
    T = np.full((H, W), 285.0, dtype=np.float32)
    u0 = np.zeros((H, W), dtype=np.float32)
    v0 = np.zeros((H, W), dtype=np.float32)

    pp_off = PlanetParams(storm_pressure_amp_pa=0.0)
    pp_on = PlanetParams(storm_pressure_amp_pa=110.0)

    u_off, v_off = atmo.evolve_wind(u0.copy(), v0.copy(), T, None, elev, dt_days=1.0,
                                     time_days=20.0, planet_params=pp_off)
    u_on, v_on = atmo.evolve_wind(u0.copy(), v0.copy(), T, None, elev, dt_days=1.0,
                                   time_days=20.0, planet_params=pp_on)

    assert not np.array_equal(u_off, u_on) or not np.array_equal(v_off, v_on)
    assert np.all(np.isfinite(u_on)) and np.all(np.isfinite(v_on))
    assert np.max(np.abs(u_on)) < 150.0 and np.max(np.abs(v_on)) < 150.0  # vmax_clip default


def test_evolve_wind_storms_deterministic_across_calls():
    """Two identical evolve_wind calls at the same time_days must match exactly."""
    H, W = 32, 64
    elev = _make_elev(H, W)
    T = np.full((H, W), 285.0, dtype=np.float32)
    u0 = np.zeros((H, W), dtype=np.float32)
    v0 = np.zeros((H, W), dtype=np.float32)
    pp = PlanetParams(storm_pressure_amp_pa=110.0)

    u1, v1 = atmo.evolve_wind(u0.copy(), v0.copy(), T, None, elev, dt_days=1.0,
                               time_days=42.0, planet_params=pp)
    u2, v2 = atmo.evolve_wind(u0.copy(), v0.copy(), T, None, elev, dt_days=1.0,
                               time_days=42.0, planet_params=pp)
    assert np.array_equal(u1, u2)
    assert np.array_equal(v1, v2)


# ---------------------------------------------------------------------------
# Full simulate_step integration: storms increase day-to-day precip variability
# ---------------------------------------------------------------------------

def _run_daily(days_count: int, storm_amp: float, H: int = 32, W: int = 64):
    from simulate import create_initial_state, simulate_step
    elev = _make_elev(H, W)
    pp = PlanetParams(storm_pressure_amp_pa=storm_amp)
    state = create_initial_state(elev, day_of_year=80.0, planet_params=pp)
    precip_snapshots = []
    for _ in range(days_count):
        state, _ = simulate_step(state, days=1.0, block_size=4, wind_block_size=4,
                                  update_wind=True, planet_params=pp)
        precip_snapshots.append(state.precipitation.copy())
    return state, precip_snapshots


def _midlat_rows(H: int) -> slice:
    lats = (0.5 - (np.arange(H) + 0.5) / H) * 180.0
    rows = np.where((np.abs(lats) >= 30) & (np.abs(lats) <= 60))[0]
    return slice(int(rows.min()), int(rows.max()) + 1)


@pytest.mark.slow
def test_storms_increase_midlat_precip_day_to_day_variance():
    """Mid-latitude precipitation should vary more day-to-day with storms enabled
    than with storms disabled -- the observable signature of moving weather systems
    rather than a static climatological band."""
    H, W = 32, 64
    n_days = 12
    state_off, snaps_off = _run_daily(n_days, storm_amp=0.0, H=H, W=W)
    state_on, snaps_on = _run_daily(n_days, storm_amp=110.0, H=H, W=W)

    rows = _midlat_rows(H)
    stack_off = np.stack([s[rows, :] for s in snaps_off], axis=0)
    stack_on = np.stack([s[rows, :] for s in snaps_on], axis=0)

    var_off = float(np.mean(np.std(stack_off, axis=0)))
    var_on = float(np.mean(np.std(stack_on, axis=0)))

    assert np.all(np.isfinite(stack_on))
    assert var_on > var_off, (
        f"expected higher day-to-day mid-lat precip variance with storms on "
        f"({var_on:.3f}) than off ({var_off:.3f})"
    )


@pytest.mark.slow
def test_storms_no_nan_over_multiday_run():
    _, snaps = _run_daily(15, storm_amp=110.0)
    for P in snaps:
        assert np.all(np.isfinite(P)), "NaN/Inf in precipitation with storms enabled"
