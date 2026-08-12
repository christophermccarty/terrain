from __future__ import annotations

from scripts.summarize_current_baseline import render_current_baseline


def test_current_baseline_summary_renders_headline_metrics():
    report = {
        "config": {"height": 64, "width": 128, "time_scale": "MONTHLY", "spinup_years": 1.0, "evaluation_years": 1.0},
        "metrics": {
            "global": {"temperature_k": 288.0, "precip_mm_day": 3.0, "cloud_fraction": 0.2},
            "koppen_map_skill": {"group_accuracy": 0.7, "class_accuracy": 0.4},
            "koppen_temperature_thresholds": {
                "coldest_month": {"accuracy": 0.9}, "warmest_month": {"accuracy": 0.8}
            },
            "precip_rescale": {"closure": {"raw_to_final_ratio": 1.2, "target_adjustment_mm_day": -0.5}},
            "regional_precip_mm_year": {"Sahara": 100.0, "Atacama": 80.0},
            "regional_target_error_fraction": {"Sahara": 0.0, "Atacama": 0.6},
        },
    }
    page = render_current_baseline(report)
    assert "64×128" in page
    assert "Köppen group accuracy: 0.700" in page
    assert "Raw-to-final precipitation ratio: 1.200" in page
    assert "| Sahara | 100 | < 200 | within target |" in page
    assert "| Atacama | 80 | < 50 | outside target |" in page
    assert "| Kalahari | — | < 200 | not scored |" in page
