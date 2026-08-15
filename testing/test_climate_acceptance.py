from __future__ import annotations

from climate_acceptance import (
    evaluate_land_candidate,
    evaluate_phase3_candidate,
    evaluate_precipitation_candidate,
    scorecard,
)


def _report(*, precip_rmse=1.4, precip_corr=0.46, group=0.67, klass=0.39, temp=6.2):
    return {
        "metrics": {
            "monthly_climatology": {
                "temperature_c": {"monthly_rmse": temp},
                "precipitation_mm_day": {
                    "monthly_log_rmse": precip_rmse,
                    "monthly_log_correlation": precip_corr,
                },
            },
            "koppen_map_skill": {"group_accuracy": group, "class_accuracy": klass},
        }
    }


def test_precipitation_gate_accepts_joint_improvement():
    decision = evaluate_precipitation_candidate(
        _report(precip_rmse=1.3, precip_corr=0.461, group=0.671, klass=0.391),
        _report(),
    )
    assert decision["accepted"]


def test_precipitation_gate_rejects_koppen_tradeoff():
    decision = evaluate_precipitation_candidate(
        _report(precip_rmse=1.3, group=0.66), _report()
    )
    assert not decision["accepted"]
    assert not decision["gates"]["koppen_group_preserved"]


def test_precipitation_gate_rejects_hidden_regional_regression():
    baseline = _report(precip_rmse=1.4)
    candidate = _report(precip_rmse=1.3, precip_corr=0.461, group=0.671, klass=0.391)
    baseline["metrics"].update(
        {"reference_error_score": 0.10, "regional_target_error_fraction": {"Atacama": 0.2}}
    )
    candidate["metrics"].update(
        {"reference_error_score": 0.11, "regional_target_error_fraction": {"Atacama": 0.3}}
    )
    decision = evaluate_precipitation_candidate(candidate, baseline)
    assert not decision["accepted"]
    assert not decision["gates"]["regional_target_errors_nonregressing"]
    assert not decision["gates"]["reference_error_score_nonregressing"]


def test_regional_gate_ignores_unscored_none_values():
    baseline = _report()
    candidate = _report(precip_rmse=1.3, precip_corr=0.461, group=0.671, klass=0.391)
    baseline["metrics"]["regional_target_error_fraction"] = {"Sparse region": None}
    candidate["metrics"]["regional_target_error_fraction"] = {"Sparse region": None}
    assert evaluate_precipitation_candidate(candidate, baseline)["accepted"]


def test_scorecard_requires_monthly_reference_metrics():
    try:
        scorecard({"metrics": {}})
    except ValueError as exc:
        assert "acceptance metrics" in str(exc)
    else:
        raise AssertionError("missing metrics should be rejected")


def test_land_gate_requires_temperature_improvement_without_koppen_loss():
    decision = evaluate_land_candidate(
        _report(temp=6.1, precip_rmse=1.41, group=0.671, klass=0.391), _report()
    )
    assert decision["accepted"]


def _phase3_report(**kwargs):
    report = _report(**{key: value for key, value in kwargs.items() if key in {
        "precip_rmse", "precip_corr", "group", "klass", "temp"
    }})
    report["metrics"].update({
        "koppen_temperature_thresholds": {
            "coldest_month": {"accuracy": kwargs.get("cold", 0.8)},
            "warmest_month": {"accuracy": kwargs.get("warm", 0.7)},
        },
        "land_seasonal_cycle": {"cycle_error_score": kwargs.get("cycle", 5.0)},
    })
    if kwargs.get("convergence", False):
        report["metrics"]["atmospheric_heat_convergence"] = {
            "max_abs_global_area_mean_w_m2": kwargs.get("closure", 0.0)
        }
    return report


def test_phase3_gate_requires_thresholds_shape_and_closed_forcing():
    baseline = _phase3_report()
    candidate = _phase3_report(
        temp=6.1,
        precip_rmse=1.39,
        group=0.68,
        klass=0.40,
        cold=0.81,
        warm=0.71,
        cycle=4.0,
        convergence=True,
    )
    assert evaluate_phase3_candidate(candidate, baseline)["accepted"]


def test_phase3_gate_rejects_nonconservative_forcing():
    baseline = _phase3_report()
    candidate = _phase3_report(
        temp=6.1, group=0.68, klass=0.40, cycle=4.0,
        convergence=True, closure=0.1,
    )
    decision = evaluate_phase3_candidate(candidate, baseline)
    assert not decision["accepted"]
    assert not decision["gates"]["heat_convergence_globally_closed"]
