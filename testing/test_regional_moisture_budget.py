from __future__ import annotations

import numpy as np
import pytest

from regional_moisture_budget import (
    regional_moisture_budget_snapshot,
    time_average_regional_moisture_budget,
)


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
