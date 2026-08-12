from __future__ import annotations

import copy

import numpy as np
import pytest

from monthly_climatology import MonthlyClimatology
from planet_params import EARTH
from real_terrain_validation import (
    DEFAULT_BASELINE_PATH,
    RealTerrainValidationConfig,
    compare_validation_reports,
    load_bundled_earth_dem,
    load_validation_report,
    run_real_terrain_validation,
    summarize_real_terrain_climate,
)
from regional_validation import EARTH_PRECIP_REGIONS, region_mask
from simulate import PlanetState


def test_bundled_dem_downsampling_is_deterministic_and_geographic():
    first = load_bundled_earth_dem(64, 128)
    second = load_bundled_earth_dem(64, 128)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (64, 128)
    assert first.dtype == np.float32
    assert 0.0 < float(np.mean(first == 0.0)) < 0.9

    land = first > 0.0
    missing = [
        region.name
        for region in EARTH_PRECIP_REGIONS
        if not np.any(region_mask(first.shape, region, cell_mask=land))
    ]
    assert not missing


def test_climate_summary_contains_global_regional_and_zonal_metrics():
    shape = (180, 360)
    elevation = np.ones(shape, dtype=np.float32)
    elevation[:12] = 0.0
    state = PlanetState(day_of_year=80.0, elevation=elevation, planet_params=EARTH)
    temperature = np.full(shape, 288.0, dtype=np.float32)
    precipitation = np.full(shape, 1.0, dtype=np.float32)
    cloud = np.full(shape, 0.5, dtype=np.float32)
    soil = np.full(shape, 0.4, dtype=np.float32)
    lower_v = np.full(shape, 1.5, dtype=np.float32)
    upper_v = np.full(shape, -1.0, dtype=np.float32)

    metrics = summarize_real_terrain_climate(
        state,
        mean_temperature_k=temperature,
        mean_precipitation_mm_day=precipitation,
        mean_cloud_fraction=cloud,
        mean_soil_moisture=soil,
        planet_params=EARTH,
        mean_surface_temperature_k=temperature + 1.0,
        mean_lower_meridional_wind_m_s=lower_v,
        mean_upper_meridional_wind_m_s=upper_v,
    )

    assert metrics["global"]["temperature_k"] == pytest.approx(288.0)
    assert metrics["global"]["precip_mm_day"] == pytest.approx(1.0)
    assert set(metrics["regional_precip_mm_year"]) == {
        region.name for region in EARTH_PRECIP_REGIONS
    }
    assert metrics["regional_precip_mm_year"]["US Midwest"] == pytest.approx(
        EARTH.orbital_period_days
    )
    assert "40-50N" in metrics["zonal"]
    assert np.isfinite(metrics["reference_error_score"])
    assert metrics["two_layer_overturning"]["diagnosed_mse_strength_available"] is False
    assert metrics["two_layer_overturning"]["two_layer_mass_flux_available"] is True


def test_climate_summary_scores_optional_monthly_climatology():
    shape = (16, 32)
    elevation = np.zeros(shape, dtype=np.float32)
    elevation[4:12, 4:28] = 0.5
    monthly_temperature = np.full((12, *shape), 288.0, dtype=np.float32)
    monthly_precipitation = np.full((12, *shape), 2.0, dtype=np.float32)
    state = PlanetState(
        day_of_year=80.0,
        elevation=elevation,
        planet_params=EARTH,
        monthly_temp=monthly_temperature,
        monthly_precip=monthly_precipitation,
    )
    reference = MonthlyClimatology(
        temperature_k=monthly_temperature,
        precipitation_mm_day=monthly_precipitation,
        land_fraction=np.ones(shape, dtype=np.float32),
        metadata={"source": "synthetic", "period": "1991-2020", "license": "test-only"},
    )
    metrics = summarize_real_terrain_climate(
        state,
        mean_temperature_k=monthly_temperature.mean(axis=0),
        mean_precipitation_mm_day=monthly_precipitation.mean(axis=0),
        mean_cloud_fraction=np.full(shape, 0.5, dtype=np.float32),
        mean_soil_moisture=np.full(shape, 0.4, dtype=np.float32),
        planet_params=EARTH,
        monthly_climatology=reference,
    )
    assert metrics["monthly_climatology"]["temperature_c"]["monthly_rmse"] == pytest.approx(0.0)
    assert metrics["monthly_climatology"]["precipitation_mm_day"]["monthly_log_rmse"] == pytest.approx(0.0)
    assert metrics["regional_land_temperature"]["Sahara"]["annual_bias_c"] == pytest.approx(0.0)
    assert metrics["regional_land_temperature"]["US Midwest"]["monthly_rmse_c"] == pytest.approx(0.0)


def test_baseline_comparison_reports_material_regression():
    baseline = {
        "config": {"height": 64},
        "metrics": {
            "global": {
                "temperature_k": 288.0,
                "precip_mm_day": 2.7,
                "cloud_fraction": 0.5,
                "land_soil_moisture": 0.4,
                "land_soil_floor_fraction": 0.0,
                "nh_midlat_soil_floor_fraction": 0.0,
            },
            "regional_precip_mm_year": {"Sahara": 150.0},
            "regional_soil_moisture": {"Sahara": 0.2},
            "zonal": {
                "0-10N": {"temperature_c": 26.0, "precip_mm_year": 1900.0}
            },
        },
    }
    current = copy.deepcopy(baseline)
    current["metrics"]["global"]["temperature_k"] = 294.0
    current["metrics"]["regional_precip_mm_year"]["Sahara"] = 400.0

    failures = compare_validation_reports(current, baseline)
    assert any("temperature_k" in failure for failure in failures)
    assert any("Sahara" in failure for failure in failures)


def test_tracked_real_terrain_baseline_is_valid():
    report = load_validation_report(DEFAULT_BASELINE_PATH)
    assert report["config"] == {
        "block_size": 4,
        "evaluation_years": 1.0,
        "height": 64,
        "precip_block_size": 1,
        "spinup_years": 1.0,
        "start_day": 80.0,
        "time_scale": "MONTHLY",
        "width": 128,
        "wind_block_size": 4,
    }
    assert np.isfinite(report["metrics"]["reference_error_score"])


@pytest.mark.slow
def test_compact_real_terrain_report_contains_regional_moisture_budget():
    _, report = run_real_terrain_validation(
        RealTerrainValidationConfig(spinup_years=0.0, evaluation_years=0.25)
    )
    budget = report["metrics"]["regional_moisture_budget"]
    assert set(budget) == {region.name for region in EARTH_PRECIP_REGIONS}
    assert budget["Atacama"]["precipitation_final_mm_day"] is not None
    assert budget["Atacama"]["post_raw_precip_adjustment_mm_day"] is not None
    seasonal_budget = report["metrics"]["seasonal_regional_moisture_budget"]
    assert seasonal_budget["season_order"] == ["DJF", "MAM", "JJA", "SON"]
    assert seasonal_budget["seasons"]["MAM"]["regions"]["Atacama"]["lower_wind_speed_m_s"] is not None
    seasonal_jet = report["metrics"]["seasonal_jet"]
    assert seasonal_jet["sample_count"] == 3
    assert seasonal_jet["upper"]["nh"]["mean_core_speed_m_s"] > 0.0


@pytest.mark.slow
def test_compact_real_terrain_run_matches_tracked_baseline():
    baseline = load_validation_report(DEFAULT_BASELINE_PATH)
    config = RealTerrainValidationConfig(**baseline["config"])
    _, current = run_real_terrain_validation(config)
    failures = compare_validation_reports(current, baseline)
    assert not failures, "\n".join(failures)
