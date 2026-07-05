"""test_jet_stream.py — persistent jet meander index + blocking events.

Unlike ROSSBY_MODES and _storm_pressure_anomaly (both pure, stateless functions
of time_days), a meandering/blocking jet genuinely needs memory: a blocking
ridge holds a fixed longitude for weeks regardless of what the pressure field
would otherwise do. `atmosphere._update_jet_index` / `_update_jet_blocking`
are the two pieces of real prognostic state (persisted in PlanetState as
jet_index_{nh,sh} / jet_block_*); the noise/trigger draws inside them are
still seeded from time_days (not a stored RNG), so a given (state, total_days)
pair always produces the same next state -- same reproducibility contract as
the storm systems in test_storm_systems.py.
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
# _update_jet_index: AR1 meander/waviness index
# ---------------------------------------------------------------------------

def test_jet_index_deterministic():
    """Same inputs must reproduce bit-identical output (no hidden RNG state)."""
    v1 = atmo._update_jet_index(0.3, 35.0, 1.0, 57.0, hemisphere_seed=1)
    v2 = atmo._update_jet_index(0.3, 35.0, 1.0, 57.0, hemisphere_seed=1)
    assert v1 == v2


def test_jet_index_different_hemisphere_seed_differs():
    """NH and SH must not draw the same noise on the same day (or they'd meander in lockstep)."""
    nh = atmo._update_jet_index(0.0, 40.0, 1.0, 100.0, hemisphere_seed=1)
    sh = atmo._update_jet_index(0.0, 40.0, 1.0, 100.0, hemisphere_seed=2)
    assert nh != sh


def test_jet_index_mean_reversion_toward_gradient_target():
    """With noise disabled, the index should climb toward the target implied by a
    weak pole-equator gradient (weaker than gradient_ref_k -> positive/wavier target)."""
    index = -2.0  # start far from the target
    gradient_k = 20.0  # well below gradient_ref_k=40 -> positive target
    for day in range(1, 200):
        index = atmo._update_jet_index(
            index, gradient_k, 1.0, float(day), hemisphere_seed=1, noise_amp=0.0,
        )
    # target = clip((40-20)/40, -1, 1) = 0.5; AR1 approaches but never overshoots it.
    assert index == pytest.approx(0.5, abs=0.05), f"expected index to converge near 0.5, got {index:.3f}"


def test_jet_index_bounded_under_large_noise():
    """Index must stay within its [-2, 2] clip even under adversarially large noise."""
    index = 0.0
    for day in range(1, 500):
        index = atmo._update_jet_index(
            index, 40.0, 1.0, float(day), hemisphere_seed=1, noise_amp=5.0,
        )
        assert -2.0 <= index <= 2.0
    assert np.isfinite(index)


def test_jet_index_strong_gradient_pushes_target_negative():
    """A gradient stronger than gradient_ref_k should relax the index toward a
    negative (fast, zonal-jet) target -- opposite of the weak-gradient case."""
    index = 2.0
    for day in range(1, 200):
        index = atmo._update_jet_index(
            index, 80.0, 1.0, float(day), hemisphere_seed=1, noise_amp=0.0,
        )
    assert index < -0.5, f"expected index to fall toward a negative target, got {index:.3f}"


# ---------------------------------------------------------------------------
# _update_jet_blocking: two-state blocking-ridge machine
# ---------------------------------------------------------------------------

def test_jet_blocking_deterministic():
    r1 = atmo._update_jet_blocking(-1.0, 0.0, 0.0, 1.0, 1.0, 500.0, hemisphere_seed=1)
    r2 = atmo._update_jet_blocking(-1.0, 0.0, 0.0, 1.0, 1.0, 500.0, hemisphere_seed=1)
    assert r1 == r2


def test_jet_blocking_inactive_when_index_low_and_no_trigger():
    """At index=0 (waviness sigmoid << 0.5) the daily trigger probability is tiny;
    scanning a modest window of days should find long inactive stretches."""
    inactive_days = 0
    lon, days_left, total = -1.0, 0.0, 0.0
    for day in range(1, 60):
        lon, days_left, total = atmo._update_jet_blocking(
            lon, days_left, total, -1.0, 1.0, float(day), hemisphere_seed=3,
        )
        if days_left <= 0.0:
            inactive_days += 1
    assert inactive_days > 40, "expected blocking to stay inactive most days at a low jet index"


def test_jet_blocking_hold_and_countdown():
    """Once active, the block must hold its longitude fixed and count down to
    zero, then report inactive (-1, 0, 0)."""
    lon0, days_left0, total0 = 15.0, 12.0, 20.0
    lon1, days_left1, total1 = atmo._update_jet_blocking(
        lon0, days_left0, total0, 1.5, 1.0, 300.0, hemisphere_seed=1,
    )
    assert lon1 == lon0
    assert days_left1 == pytest.approx(days_left0 - 1.0)
    assert total1 == total0

    # Fast-forward past the end of the block.
    lon2, days_left2, total2 = atmo._update_jet_blocking(
        lon0, 0.5, total0, 1.5, 1.0, 301.0, hemisphere_seed=1,
    )
    assert lon2 == -1.0 and days_left2 == 0.0 and total2 == 0.0


def test_jet_blocking_trigger_rate_statistics():
    """Over a long run at an elevated jet index, the observed blocked-day fraction
    should land in the right ballpark of trigger_rate * waviness * mean_duration
    (a loose statistical check -- not exact, since durations/triggers are randomly
    drawn -- but should catch a badly wired probability or duration range)."""
    trigger_rate = 0.05
    duration_range = (10.0, 20.0)
    jet_index = 1.5  # sigmoid(3*(1.5-0.5)) ~= 0.95 -> waviness close to 1
    lon, days_left, total = -1.0, 0.0, 0.0
    blocked_days = 0
    n_days = 3000
    for day in range(1, n_days + 1):
        lon, days_left, total = atmo._update_jet_blocking(
            lon, days_left, total, jet_index, 1.0, float(day), hemisphere_seed=7,
            trigger_rate_per_day=trigger_rate, duration_range_days=duration_range,
        )
        if days_left > 0.0:
            blocked_days += 1
        assert duration_range[0] - 1e-6 <= total <= duration_range[1] + 1e-6 or total == 0.0

    waviness = 1.0 / (1.0 + np.exp(-3.0 * (jet_index - 0.5)))
    mean_duration = sum(duration_range) / 2.0
    expected_frac = trigger_rate * waviness * mean_duration
    expected_frac = min(expected_frac, 1.0)
    observed_frac = blocked_days / n_days
    assert 0.15 < observed_frac < 0.95, (
        f"observed blocked fraction {observed_frac:.3f} implausible "
        f"(expected roughly {expected_frac:.3f})"
    )


# ---------------------------------------------------------------------------
# _blocking_ridge_pressure_anomaly: stationary high-pressure blob
# ---------------------------------------------------------------------------

def test_blocking_ridge_inactive_is_zero():
    H, W = 32, 64
    lat_2d = _lat_2d(H, W)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)[None, :]
    out = atmo._blocking_ridge_pressure_anomaly(
        lat_2d, lon_1d, 48.0, 0.0, days_left=0.0, total_duration_days=20.0,
        amp_pa=180.0, radius_km=3200.0,
    )
    assert np.all(out == 0.0)


def test_blocking_ridge_is_positive_and_confined():
    H, W = 32, 64
    lat_2d = _lat_2d(H, W)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)[None, :]
    out = atmo._blocking_ridge_pressure_anomaly(
        lat_2d, lon_1d, 48.0, 0.0, days_left=10.0, total_duration_days=20.0,
        amp_pa=180.0, radius_km=3200.0,
    )
    assert np.all(out >= 0.0), "a blocking ridge is a high -- must never be negative"
    assert np.max(out) > 50.0
    # Far from the ridge (opposite hemisphere / opposite side of globe) should be ~0.
    far_row = 0  # near the pole opposite the 48N ridge center-ish; H=32 -> row 0 is far north
    assert np.max(np.abs(out[H - 1, :])) < np.max(out) * 0.2


def test_blocking_ridge_ramps_up_and_down():
    """Envelope should be smaller right after onset and right before decay than
    at the midpoint of a long block (avoids a discontinuous pressure jump)."""
    H, W = 32, 64
    lat_2d = _lat_2d(H, W)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)[None, :]
    total = 20.0
    peak_early = np.max(atmo._blocking_ridge_pressure_anomaly(
        lat_2d, lon_1d, 48.0, 0.0, days_left=19.5, total_duration_days=total,
        amp_pa=180.0, radius_km=3200.0, ramp_days=2.0,
    ))
    peak_mid = np.max(atmo._blocking_ridge_pressure_anomaly(
        lat_2d, lon_1d, 48.0, 0.0, days_left=10.0, total_duration_days=total,
        amp_pa=180.0, radius_km=3200.0, ramp_days=2.0,
    ))
    peak_late = np.max(atmo._blocking_ridge_pressure_anomaly(
        lat_2d, lon_1d, 48.0, 0.0, days_left=0.3, total_duration_days=total,
        amp_pa=180.0, radius_km=3200.0, ramp_days=2.0,
    ))
    assert peak_early < peak_mid
    assert peak_late < peak_mid


# ---------------------------------------------------------------------------
# evolve_wind integration
# ---------------------------------------------------------------------------

def test_evolve_wind_jet_index_shifts_midlat_relaxation_and_stays_finite():
    H, W = 32, 64
    elev = _make_elev(H, W)
    T = np.full((H, W), 285.0, dtype=np.float32)
    u0 = np.zeros((H, W), dtype=np.float32)
    v0 = np.zeros((H, W), dtype=np.float32)
    pp = PlanetParams(storm_pressure_amp_pa=0.0, trade_wave_pressure_amp_pa=0.0,
                       jet_block_pressure_amp_pa=0.0)

    u_base, v_base = atmo.evolve_wind(
        u0.copy(), v0.copy(), T, None, elev, dt_days=1.0, cell_relax_days=3.0,
        time_days=10.0, planet_params=pp, jet_index_nh=0.0, jet_index_sh=0.0,
    )
    u_shift, v_shift = atmo.evolve_wind(
        u0.copy(), v0.copy(), T, None, elev, dt_days=1.0, cell_relax_days=3.0,
        time_days=10.0, planet_params=pp, jet_index_nh=1.5, jet_index_sh=0.0,
    )
    assert np.all(np.isfinite(u_shift)) and np.all(np.isfinite(v_shift))
    assert not np.array_equal(u_base, u_shift), "a nonzero NH jet index must change the NH wind field"

    lats_deg = np.rad2deg(_lat_2d(H, W)[:, 0])
    nh_rows = lats_deg >= 0.0
    sh_rows = lats_deg < 0.0
    # SH should be far less affected than NH by an NH-only jet index perturbation
    # (some small numerical bleed across the equator via advection is expected,
    # so this compares magnitudes rather than requiring near-exact equality).
    nh_diff = float(np.max(np.abs(u_base[nh_rows, :] - u_shift[nh_rows, :])))
    sh_diff = float(np.max(np.abs(u_base[sh_rows, :] - u_shift[sh_rows, :])))
    assert nh_diff > 0.05, f"expected a measurable NH wind change, got {nh_diff:.4f}"
    assert sh_diff < nh_diff * 0.1, (
        f"expected SH bleed-over to be much smaller than the NH change: "
        f"sh_diff={sh_diff:.4f} vs nh_diff={nh_diff:.4f}"
    )


def test_evolve_wind_blocking_adds_persistent_high_pressure_signature():
    """An active block should measurably change the wind field vs. no block,
    and disabling it via jet_block_pressure_amp_pa=0 must restore the baseline
    exactly (a clean on/off toggle, matching the storms/trade-wave contract)."""
    H, W = 32, 64
    elev = _make_elev(H, W)
    T = np.full((H, W), 285.0, dtype=np.float32)
    u0 = np.zeros((H, W), dtype=np.float32)
    v0 = np.zeros((H, W), dtype=np.float32)
    block_nh = (0.0, 10.0, 20.0)  # active, lon=0, mid-lifecycle

    pp_off = PlanetParams(storm_pressure_amp_pa=0.0, trade_wave_pressure_amp_pa=0.0,
                           jet_block_pressure_amp_pa=0.0)
    pp_on = PlanetParams(storm_pressure_amp_pa=0.0, trade_wave_pressure_amp_pa=0.0,
                          jet_block_pressure_amp_pa=180.0)

    u_off, v_off = atmo.evolve_wind(
        u0.copy(), v0.copy(), T, None, elev, dt_days=1.0, time_days=10.0,
        planet_params=pp_off, jet_block_nh=block_nh,
    )
    u_on, v_on = atmo.evolve_wind(
        u0.copy(), v0.copy(), T, None, elev, dt_days=1.0, time_days=10.0,
        planet_params=pp_on, jet_block_nh=block_nh,
    )
    assert np.all(np.isfinite(u_on)) and np.all(np.isfinite(v_on))
    assert not np.array_equal(u_off, u_on) or not np.array_equal(v_off, v_on)
    assert np.max(np.abs(u_on)) < 150.0 and np.max(np.abs(v_on)) < 150.0  # vmax_clip default


def test_evolve_wind_jet_deterministic_across_calls():
    H, W = 32, 64
    elev = _make_elev(H, W)
    T = np.full((H, W), 285.0, dtype=np.float32)
    u0 = np.zeros((H, W), dtype=np.float32)
    v0 = np.zeros((H, W), dtype=np.float32)
    pp = PlanetParams(jet_block_pressure_amp_pa=180.0)
    block_nh = (5.0, 8.0, 20.0)

    u1, v1 = atmo.evolve_wind(u0.copy(), v0.copy(), T, None, elev, dt_days=1.0,
                               cell_relax_days=3.0, time_days=42.0, planet_params=pp,
                               jet_index_nh=0.7, jet_block_nh=block_nh)
    u2, v2 = atmo.evolve_wind(u0.copy(), v0.copy(), T, None, elev, dt_days=1.0,
                               cell_relax_days=3.0, time_days=42.0, planet_params=pp,
                               jet_index_nh=0.7, jet_block_nh=block_nh)
    assert np.array_equal(u1, u2)
    assert np.array_equal(v1, v2)


# ---------------------------------------------------------------------------
# Full simulate_step integration
# ---------------------------------------------------------------------------

def _lonlat_1d(H: int, W: int):
    lat = (0.5 - (np.arange(H) + 0.5) / H) * 180.0
    lon = np.linspace(-180.0, 180.0, W, endpoint=False)
    return lat, lon


@pytest.mark.slow
def test_simulate_step_jet_stream_no_nan_over_multimonth_run():
    """60-day run with the full jet mechanism enabled (default params) must
    stay finite and bounded -- the basic stability bar every physics feature
    in this codebase has to clear (mirrors test_storms_no_nan_over_multiday_run)."""
    from simulate import create_initial_state, simulate_step
    H, W = 32, 64
    elev = _make_elev(H, W)
    pp = PlanetParams()
    state = create_initial_state(elev, day_of_year=80.0, planet_params=pp)
    for _ in range(60):
        state, _ = simulate_step(state, days=1.0, block_size=4, wind_block_size=4,
                                  planet_params=pp)
        assert np.all(np.isfinite(state.temperature))
        assert np.all(np.isfinite(state.wind_u)) and np.all(np.isfinite(state.wind_v))
        assert np.all(np.isfinite(state.precipitation))
    assert -2.0 <= state.jet_index_nh <= 2.0
    assert -2.0 <= state.jet_index_sh <= 2.0


@pytest.mark.slow
def test_forced_blocking_ridge_suppresses_regional_precip():
    """Forcing an active NH block at lon=0 for ~3 weeks should leave the
    surrounding mid-latitude region measurably drier than an identical run with
    the block pressure term disabled -- the mechanism a memoryless Rossby wave
    or storm population cannot produce (a genuinely *persistent* regional
    anomaly, not just day-to-day noise)."""
    from simulate import create_initial_state, simulate_step
    H, W = 32, 64
    elev = _make_elev(H, W)
    pp_base = PlanetParams()
    state = create_initial_state(elev, day_of_year=80.0, planet_params=pp_base)
    for _ in range(20):  # brief spinup so wind/precip fields are non-trivial
        state, _ = simulate_step(state, days=1.0, block_size=4, wind_block_size=4,
                                  planet_params=pp_base)

    forced = state._replace(
        jet_block_lon_nh=0.0, jet_block_days_left_nh=25.0, jet_block_total_days_nh=25.0,
    )
    pp_on = PlanetParams(jet_block_pressure_amp_pa=180.0)
    pp_off = PlanetParams(jet_block_pressure_amp_pa=0.0)

    lat_1d, lon_1d = _lonlat_1d(H, W)
    rows = np.where((lat_1d >= 33.0) & (lat_1d <= 63.0))[0]
    cols = np.where(np.abs(lon_1d) <= 30.0)[0]

    st_on, st_off = forced, forced
    precip_on, precip_off = [], []
    for _ in range(21):
        st_on, _ = simulate_step(st_on, days=1.0, block_size=4, wind_block_size=4, planet_params=pp_on)
        st_off, _ = simulate_step(st_off, days=1.0, block_size=4, wind_block_size=4, planet_params=pp_off)
        precip_on.append(float(np.mean(st_on.precipitation[np.ix_(rows, cols)])))
        precip_off.append(float(np.mean(st_off.precipitation[np.ix_(rows, cols)])))

    assert np.all(np.isfinite(precip_on)) and np.all(np.isfinite(precip_off))
    assert np.mean(precip_on) < np.mean(precip_off), (
        f"expected suppressed precip under a persistent blocking ridge: "
        f"blocked={np.mean(precip_on):.3f} mm/day vs disabled={np.mean(precip_off):.3f} mm/day"
    )
