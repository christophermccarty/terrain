from __future__ import annotations

import numpy as np
import pytest

from regional_moisture_budget import (
    REGIONAL_MOISTURE_SOURCE_REGIONS,
    regional_moisture_budget_snapshot,
    season_for_day,
    seasonal_regional_moisture_budget,
    time_average_regional_moisture_budget,
)
from regional_validation import region_mask


def _fields(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "precipitation_final_mm_day": np.full(shape, 3.0, dtype=np.float32),
        "precipitation_raw_mm_day": np.full(shape, 1.0, dtype=np.float32),
        "land_evaporation_mm_day": np.full(shape, 0.4, dtype=np.float32),
        "ocean_evaporation_mm_day": np.zeros(shape, dtype=np.float32),
        "div": np.full(shape, -2.0e-6, dtype=np.float32),
        "conv": np.full(shape, 0.3, dtype=np.float32),
        "humidity_before_rainout": np.full(shape, 0.01, dtype=np.float32),
        "ascent_driver": np.full(shape, 0.6, dtype=np.float32),
        "conv_driver": np.full(shape, 0.7, dtype=np.float32),
        "orog": np.full(shape, 0.8, dtype=np.float32),
        "rain_shadow_suppression": np.full(shape, 0.9, dtype=np.float32),
        "precip_target_achieved_fraction": np.full(shape[0], 0.95, dtype=np.float32),
    }


def test_snapshot_reports_units_and_distinguishes_allocator_adjustment():
    shape = (180, 360)
    snapshot = regional_moisture_budget_snapshot(
        np.ones(shape, dtype=np.float32), _fields(shape)
    )
    atacama = snapshot["Atacama"]
    assert atacama["precipitation_final_mm_day"] == pytest.approx(3.0)
    assert atacama["post_raw_precip_adjustment_mm_day"] == pytest.approx(2.0)
    assert atacama["land_evaporation_mm_day"] == pytest.approx(0.4)
    assert atacama["lower_wind_convergence_proxy"] == pytest.approx(2.0e-6)
    assert atacama["moisture_flux_convergence_driver"] == pytest.approx(0.3)
    assert atacama["row_target_achieved_fraction"] == pytest.approx(0.95)


def test_time_average_uses_sample_duration_and_preserves_missing_values():
    shape = (180, 360)
    early = regional_moisture_budget_snapshot(np.ones(shape), _fields(shape))
    late_fields = _fields(shape)
    late_fields["precipitation_final_mm_day"][:] = 5.0
    late = regional_moisture_budget_snapshot(np.ones(shape), late_fields)
    averaged = time_average_regional_moisture_budget(((1.0, early), (3.0, late)))
    assert averaged["S Japan"]["precipitation_final_mm_day"] == pytest.approx(4.5)
    assert averaged["S Japan"]["ocean_evaporation_mm_day"] == pytest.approx(0.0)


def test_pathway_snapshot_reports_a_real_flux_measure_and_ocean_source_context():
    shape = (180, 360)
    elevation = np.ones(shape, dtype=np.float32)
    source_mask = region_mask(shape, REGIONAL_MOISTURE_SOURCE_REGIONS["Atacama"])
    elevation[source_mask] = 0.0
    fields = _fields(shape)
    fields["storm_window"] = np.full(shape[0], 0.2, dtype=np.float32)
    fields["subsidence_suppression"] = np.full(shape, 0.8, dtype=np.float32)
    fields["upwind_sst_anomaly"] = np.full(shape, -1.5, dtype=np.float32)
    surface = np.full(shape, 302.0, dtype=np.float32)
    air = np.full(shape, 303.0, dtype=np.float32)
    humidity = fields["humidity_before_rainout"]
    surface[source_mask] = 285.0
    air[source_mask] = 287.0
    humidity[source_mask] = 0.02
    snapshot = regional_moisture_budget_snapshot(
        elevation,
        fields,
        pathway_fields={
            "lower_wind_u_m_s": np.full(shape, 5.0, dtype=np.float32),
            "lower_wind_v_m_s": np.zeros(shape, dtype=np.float32),
            "surface_temperature_k": surface,
            "air_temperature_k": air,
            "cloud_fraction": np.full(shape, 0.7, dtype=np.float32),
        },
    )
    atacama = snapshot["Atacama"]
    assert np.isfinite(atacama["physical_moisture_flux_convergence_q_s"])
    assert atacama["lower_wind_speed_m_s"] == pytest.approx(5.0)
    assert atacama["storm_track_window"] == pytest.approx(0.2)
    assert atacama["source_ocean_humidity_q"] == pytest.approx(0.02)
    assert atacama["source_ocean_surface_temperature_k"] == pytest.approx(285.0)
    assert atacama["source_ocean_air_surface_inversion_k"] == pytest.approx(2.0)
    assert atacama["land_minus_source_ocean_surface_temperature_k"] == pytest.approx(17.0)
    assert atacama["source_to_region_lower_wind_m_s"] > 4.9
    assert atacama["upwind_sst_anomaly_k"] == pytest.approx(-1.5)


def test_astronomical_season_buckets_are_explicit_and_duration_weighted():
    early = regional_moisture_budget_snapshot(np.ones((180, 360)), _fields((180, 360)))
    late_fields = _fields((180, 360))
    late_fields["precipitation_final_mm_day"][:] = 5.0
    late = regional_moisture_budget_snapshot(np.ones((180, 360)), late_fields)
    report = seasonal_regional_moisture_budget((("MAM", 1.0, early), ("MAM", 3.0, late)))
    assert report["season_order"] == ["DJF", "MAM", "JJA", "SON"]
    assert report["seasons"]["MAM"]["sampled_days"] == pytest.approx(4.0)
    assert report["seasons"]["MAM"]["regions"]["S Japan"]["precipitation_final_mm_day"] == pytest.approx(4.5)
    assert report["seasons"]["JJA"]["sampled_days"] == 0.0
    assert season_for_day(80.0, orbital_period_days=365.2422, vernal_equinox_day=80.0) == "MAM"
    assert season_for_day(172.0, orbital_period_days=365.2422, vernal_equinox_day=80.0) == "JJA"
