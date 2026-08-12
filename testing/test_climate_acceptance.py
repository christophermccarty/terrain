from __future__ import annotations

from climate_acceptance import evaluate_land_candidate, evaluate_precipitation_candidate, scorecard


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
