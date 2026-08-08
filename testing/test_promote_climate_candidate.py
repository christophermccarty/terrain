from scripts.promote_climate_candidate import promotion_decision


def _report(*, height=128, width=256, spinup=5.0, evaluation=5.0, precip=1.3, group=0.68, klass=0.40):
    return {
        "config": {"height": height, "width": width, "spinup_years": spinup, "evaluation_years": evaluation},
        "metrics": {
            "monthly_climatology": {
                "temperature_c": {"monthly_rmse": 6.0},
                "precipitation_mm_day": {"monthly_log_rmse": precip, "monthly_log_correlation": 0.47},
            },
            "koppen_map_skill": {"group_accuracy": group, "class_accuracy": klass},
        },
    }


def test_promotion_requires_long_configuration_and_joint_skill():
    decision = promotion_decision(_report(), _report(precip=1.4, group=0.67, klass=0.39))
    assert decision["promoted"]


def test_promotion_rejects_short_run_even_with_good_skill():
    baseline = _report(height=64, width=128, spinup=1.0, evaluation=1.0, precip=1.4, group=0.67, klass=0.39)
    decision = promotion_decision(_report(height=64, width=128, spinup=1.0, evaluation=1.0), baseline)
    assert not decision["promoted"]
    assert not decision["configuration_gates"]["resolution_at_least_128x256"]
