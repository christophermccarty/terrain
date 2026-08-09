"""test_optimizer_scoring.py — Integration tests for the optimizer package.

Verifies that:
1. A proper Earth simulation scores at or above the minimum quality threshold.
2. An intentionally wrong planet configuration scores significantly lower.
3. The headless runner produces consistent metrics on repeated calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _earth_metrics(spinup_years: float = 1.0, eval_years: float = 0.5, H: int = 32, W: int = 64):
    from optimizer.headless import run_simulation
    from planet_params import EARTH
    from simulate import TimeScaleMode
    return run_simulation(
        EARTH,
        spinup_years=spinup_years,
        eval_years=eval_years,
        H=H,
        W=W,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
    )


# ---------------------------------------------------------------------------
# Scoring smoke test
# ---------------------------------------------------------------------------

def test_scoring_perfect_metrics_gives_100():
    """Metrics exactly on target should yield a perfect score."""
    from optimizer.scoring import ClimateScore, ClimateMetrics, EARTH_REFERENCE

    perfect = ClimateMetrics(
        global_mean_t=288.0,
        gradient_nh=52.0,
        gradient_sh=50.0,
        ice_frac_nh=0.05,
        ice_frac_sh=0.07,
        mean_precip=2.7,
        wind_trade_mean=6.5,
        wind_midlat_mean=8.0,
        wind_itcz_conv=0.5,
        seasonal_amplitude_nh=40.0,
    )
    score = ClimateScore(EARTH_REFERENCE).score(perfect)
    assert score == pytest.approx(100.0), f"Perfect metrics should score 100, got {score:.1f}"


def test_scoring_nan_state_gives_zero():
    """NaN-containing state should immediately score zero."""
    from optimizer.scoring import ClimateScore, ClimateMetrics, EARTH_REFERENCE
    nan_m = ClimateMetrics(has_nan=True)
    score = ClimateScore(EARTH_REFERENCE).score(nan_m)
    assert score == 0.0


def test_scoring_breakdown_sums_to_total():
    """Sum of per-metric contributions should equal the reported total."""
    from optimizer.scoring import ClimateScore, ClimateMetrics, EARTH_REFERENCE
    metrics = ClimateMetrics(
        global_mean_t=285.0, gradient_nh=48.0, gradient_sh=45.0,
        ice_frac_nh=0.08, ice_frac_sh=0.06, mean_precip=2.9,
        wind_trade_mean=5.5, wind_midlat_mean=7.0, wind_itcz_conv=0.3,
        seasonal_amplitude_nh=35.0,
    )
    score_fn = ClimateScore(EARTH_REFERENCE)
    bd = score_fn.breakdown(metrics)
    metric_sum = sum(v for k, v in bd.items() if k != "total")
    total_weight = sum(w for _, w in score_fn._metrics)
    expected_total = 100.0 * metric_sum / total_weight
    assert bd["total"] == pytest.approx(expected_total, abs=0.01)


# ---------------------------------------------------------------------------
# Earth benchmark
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_earth_baseline_scores_above_threshold():
    """Earth simulation (1yr spinup, 0.5yr eval) should score at least 55/100.

    The threshold is conservative because:
    - Only a 32x64 grid (low resolution)
    - Only 1yr MONTHLY spinup (not fully equilibrated)
    - Temperature gradient and wind take 2+ years to develop fully
    A score ≥ 55 confirms the scoring function is correctly calibrated and the
    physics produce at least a recognisably Earth-like climate.
    """
    from optimizer.scoring import ClimateScore, EARTH_REFERENCE

    _, metrics = _earth_metrics(spinup_years=1.0, eval_years=0.5)

    assert not metrics.has_nan, "Earth simulation produced NaN"
    assert not metrics.has_inf, "Earth simulation produced Inf"

    score = ClimateScore(EARTH_REFERENCE).score(metrics)
    assert score >= 55.0, (
        f"Earth baseline scored {score:.1f}/100 (expected ≥ 55). "
        f"global_mean_t={metrics.global_mean_t:.1f}K, "
        f"gradient_nh={metrics.gradient_nh:.1f}K, "
        f"ice_nh={metrics.ice_frac_nh*100:.1f}%"
    )


@pytest.mark.slow
def test_dim_star_scores_lower_than_earth():
    """A dim-star planet (S0=400 W/m²) should score lower than Earth on Earth's scoring function."""
    from optimizer.scoring import ClimateScore, EARTH_REFERENCE
    from optimizer.headless import run_simulation
    from planet_params import PlanetParams
    from simulate import TimeScaleMode

    _, earth_metrics = _earth_metrics(spinup_years=1.0, eval_years=0.5)
    earth_score = ClimateScore(EARTH_REFERENCE).score(earth_metrics)

    dim_pp = PlanetParams(solar_constant=400.0)
    _, dim_metrics = run_simulation(
        dim_pp,
        spinup_years=1.0,
        eval_years=0.5,
        H=32, W=64,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
    )
    dim_score = ClimateScore(EARTH_REFERENCE).score(dim_metrics)

    assert dim_score < earth_score - 5.0, (
        f"Dim-star ({dim_score:.1f}) should score at least 5 pts below Earth ({earth_score:.1f})"
    )
    # The dim planet should be significantly colder than Earth
    assert dim_metrics.global_mean_t < earth_metrics.global_mean_t - 10.0, (
        f"Dim planet mean T {dim_metrics.global_mean_t:.1f}K not significantly colder than "
        f"Earth {earth_metrics.global_mean_t:.1f}K"
    )


# ---------------------------------------------------------------------------
# Headless consistency
# ---------------------------------------------------------------------------

def test_headless_no_nan_short_run():
    """Short headless run (32×64, 0.1yr) must not produce NaN or Inf."""
    from optimizer.headless import run_simulation
    from planet_params import EARTH
    from simulate import TimeScaleMode

    _, metrics = run_simulation(
        EARTH,
        spinup_years=0.1,
        eval_years=0.05,
        H=32, W=64,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
    )
    assert not metrics.has_nan, "Short run produced NaN"
    assert not metrics.has_inf, "Short run produced Inf"
    # Temperature should be in a physically plausible range
    assert 220.0 < metrics.global_mean_t < 340.0, (
        f"Global mean T {metrics.global_mean_t:.1f}K outside plausible range [220, 340]"
    )


def test_headless_mars_no_nan():
    """Mars simulation should run without NaN."""
    from optimizer.headless import run_simulation
    from planet_params import MARS
    from simulate import TimeScaleMode

    _, metrics = run_simulation(
        MARS,
        spinup_years=0.1,
        eval_years=0.05,
        H=32, W=64,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
    )
    assert not metrics.has_nan, "Mars short run produced NaN"
    assert not metrics.has_inf, "Mars short run produced Inf"


def test_headless_metrics_plausible_after_spinup():
    """After 0.5yr MONTHLY spinup, metrics should be at least partially developed."""
    from optimizer.headless import run_simulation
    from planet_params import EARTH
    from simulate import TimeScaleMode

    _, metrics = run_simulation(
        EARTH,
        spinup_years=0.5,
        eval_years=0.1,
        H=32, W=64,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
    )
    # Temperature should be close to Earth-like (within 25K of 288K)
    assert abs(metrics.global_mean_t - 288.0) < 25.0, (
        f"Global mean T {metrics.global_mean_t:.1f}K too far from 288K"
    )
    # Precipitation should be non-zero
    assert metrics.mean_precip > 0.0, "Precipitation is zero after spinup"


# ---------------------------------------------------------------------------
# Results utilities
# ---------------------------------------------------------------------------

def test_results_save_load_roundtrip(tmp_path):
    """Save and reload a results list; verify round-trip integrity."""
    from optimizer.results import save_results, load_results

    rows = [
        {"trial_id": 0, "score": 75.3, "param_thermal_diffusion": 0.04},
        {"trial_id": 1, "score": 68.1, "param_thermal_diffusion": 0.06},
    ]
    path = tmp_path / "test_results.csv"
    save_results(rows, path)

    loaded = load_results(path)
    try:
        import pandas as pd
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 2
        scores = loaded["score"].tolist()
    except ImportError:
        assert len(loaded) == 2
        scores = [float(r["score"]) for r in loaded]

    assert pytest.approx(scores[0], abs=0.01) == 75.3
    assert pytest.approx(scores[1], abs=0.01) == 68.1


# ---------------------------------------------------------------------------
# CRU TS v4.10 real-reanalysis integration (ReferenceClimate.cru_* / run_simulation)
# ---------------------------------------------------------------------------

CRU_REFERENCE_PATH = ROOT / "testing" / "reference_data" / "cru_ts_v4.10_1991_2020.npz"
_cru_reference_missing = not CRU_REFERENCE_PATH.exists()
_cru_skip_reason = (
    f"CRU TS reference not built locally: {CRU_REFERENCE_PATH} is missing. "
    "Run `python scripts/build_cru_ts_reference.py` to fetch and build it."
)


def test_cru_reference_weights_default_to_zero_no_op():
    """ReferenceClimate's new cru_* fields must not change any existing score."""
    from optimizer.scoring import ClimateScore, ClimateMetrics, EARTH_REFERENCE

    baseline = ClimateMetrics(
        global_mean_t=288.0, gradient_nh=52.0, gradient_sh=50.0,
        ice_frac_nh=0.05, ice_frac_sh=0.07, mean_precip=2.7,
        wind_trade_mean=6.5, wind_midlat_mean=8.0, wind_itcz_conv=0.5,
        seasonal_amplitude_nh=40.0,
    )
    score = ClimateScore(EARTH_REFERENCE).score(baseline)
    assert score == pytest.approx(100.0)

    # Wildly "bad" cru_* values must not move the score while weight stays 0.0.
    perturbed = ClimateMetrics(
        global_mean_t=288.0, gradient_nh=52.0, gradient_sh=50.0,
        ice_frac_nh=0.05, ice_frac_sh=0.07, mean_precip=2.7,
        wind_trade_mean=6.5, wind_midlat_mean=8.0, wind_itcz_conv=0.5,
        seasonal_amplitude_nh=40.0,
        cru_temp_correlation=-1.0, cru_temp_rmse=50.0,
        cru_precip_log_correlation=-1.0, cru_precip_log_rmse=50.0,
    )
    assert ClimateScore(EARTH_REFERENCE).score(perturbed) == pytest.approx(100.0)


def test_cru_reference_weight_opt_in_changes_score():
    """Raising a cru_* weight makes bad cru_* values actually cost points."""
    import dataclasses
    from optimizer.scoring import ClimateScore, ClimateMetrics, EARTH_REFERENCE

    ref = dataclasses.replace(EARTH_REFERENCE, cru_temp_correlation=(0.88, 1.0, 2.0))
    perfect = ClimateMetrics(
        global_mean_t=288.0, gradient_nh=52.0, gradient_sh=50.0,
        ice_frac_nh=0.05, ice_frac_sh=0.07, mean_precip=2.7,
        wind_trade_mean=6.5, wind_midlat_mean=8.0, wind_itcz_conv=0.5,
        seasonal_amplitude_nh=40.0, cru_temp_correlation=0.95,
    )
    bad = dataclasses.replace(perfect, cru_temp_correlation=-1.0)
    assert ClimateScore(ref).score(perfect) == pytest.approx(100.0)
    assert ClimateScore(ref).score(bad) < 100.0


@pytest.mark.slow
@pytest.mark.skipif(_cru_reference_missing, reason=_cru_skip_reason)
def test_run_simulation_populates_cru_metrics_on_real_terrain():
    """monthly_climatology_path on the real bundled Earth DEM populates cru_* fields."""
    from optimizer.headless import run_simulation
    from planet_params import EARTH
    from simulate import TimeScaleMode
    from real_terrain_validation import load_bundled_earth_dem

    elevation = load_bundled_earth_dem(32, 64)
    _, metrics = run_simulation(
        EARTH,
        spinup_years=1.0,
        eval_years=1.0,
        H=32, W=64,
        elevation=elevation,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.MONTHLY,
        monthly_climatology_path=CRU_REFERENCE_PATH,
    )
    assert not metrics.has_nan and not metrics.has_inf
    assert -1.0 <= metrics.cru_temp_correlation <= 1.0
    assert metrics.cru_temp_correlation != 0.0, "cru_temp_correlation left at its inert default"
    assert metrics.cru_temp_rmse > 0.0
    assert -1.0 <= metrics.cru_precip_log_correlation <= 1.0
    assert metrics.cru_precip_log_rmse > 0.0


def test_run_simulation_without_climatology_path_leaves_cru_fields_inert():
    """Not passing monthly_climatology_path must not touch the cru_* fields at all."""
    from optimizer.headless import run_simulation
    from planet_params import EARTH
    from simulate import TimeScaleMode

    _, metrics = run_simulation(
        EARTH,
        spinup_years=0.1,
        eval_years=0.05,
        H=32, W=64,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
    )
    assert metrics.cru_temp_correlation == 0.0
    assert metrics.cru_temp_rmse == 0.0
    assert metrics.cru_precip_log_correlation == 0.0
    assert metrics.cru_precip_log_rmse == 0.0


def test_results_top_n():
    """top_n returns the highest-scoring entries."""
    from optimizer.results import top_n

    rows = [{"score": s} for s in [50.0, 90.0, 70.0, 30.0, 85.0]]
    best = top_n(rows, n=3)
    if hasattr(best, "to_dict"):
        best = best.to_dict(orient="records")
    scores = [float(r["score"]) for r in best]
    assert scores[0] == pytest.approx(90.0)
    assert len(scores) == 3
