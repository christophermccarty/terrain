"""Tests for the provider-neutral monthly T/P validation contract."""
from __future__ import annotations

import numpy as np
import pytest

from monthly_climatology import (
    MonthlyClimatology,
    load_monthly_climatology,
    regrid_monthly_climatology,
    save_monthly_climatology,
    score_monthly_climatology,
)


def _reference(height: int = 4) -> MonthlyClimatology:
    width = 2 * height
    month = np.arange(12, dtype=np.float64)[:, None, None]
    row = np.arange(height, dtype=np.float64)[None, :, None]
    col = np.arange(width, dtype=np.float64)[None, None, :]
    temperature = 270.0 + month + row + col * 0.01
    precipitation = 0.1 + month * 0.02 + row * 0.03 + col * 0.01
    return MonthlyClimatology(
        temperature_k=temperature,
        precipitation_mm_day=precipitation,
        land_fraction=np.ones((height, width), dtype=np.float64),
        metadata={
            "source": "synthetic test normal",
            "period": "1991-2020",
            "license": "test-only",
        },
    )


def test_exact_monthly_reference_scores_perfectly():
    reference = _reference()
    score = score_monthly_climatology(
        reference.temperature_k,
        reference.precipitation_mm_day,
        reference,
        model_land_mask=np.ones(reference.shape, dtype=bool),
    )
    assert score["temperature_c"]["monthly_bias"] == pytest.approx(0.0)
    assert score["temperature_c"]["monthly_rmse"] == pytest.approx(0.0)
    assert score["temperature_c"]["monthly_correlation"] == pytest.approx(1.0)
    assert score["precipitation_mm_day"]["monthly_log_rmse"] == pytest.approx(0.0)
    assert score["precipitation_mm_day"]["annual_log_correlation"] == pytest.approx(1.0)


def test_coastal_reference_mask_excludes_unscored_cells():
    reference = _reference()
    masked = MonthlyClimatology(
        temperature_k=reference.temperature_k,
        precipitation_mm_day=reference.precipitation_mm_day,
        land_fraction=np.pad(np.ones((2, 4)), ((0, 2), (0, 4))),
        metadata=reference.metadata,
    )
    model_temperature = reference.temperature_k.copy()
    model_temperature[:, 2:, 4:] += 100.0
    score = score_monthly_climatology(
        model_temperature,
        reference.precipitation_mm_day,
        masked,
        minimum_land_fraction=0.5,
    )
    assert score["temperature_c"]["monthly_rmse"] == pytest.approx(0.0)
    assert score["scored_area_fraction"] < 1.0


def test_regrid_preserves_constant_fields_and_metadata():
    source = MonthlyClimatology(
        temperature_k=np.full((12, 4, 8), 280.0),
        precipitation_mm_day=np.full((12, 4, 8), 2.5),
        land_fraction=np.full((4, 8), 0.6),
        metadata={"source": "constant", "period": "1991-2020", "license": "test-only"},
    )
    target = regrid_monthly_climatology(source, 8, 16)
    assert target.shape == (8, 16)
    assert np.allclose(target.temperature_k, 280.0)
    assert np.allclose(target.precipitation_mm_day, 2.5)
    assert np.allclose(target.land_fraction, 0.6)
    assert np.all((target.land_fraction >= 0.0) & (target.land_fraction <= 1.0))
    assert target.metadata["regridded_from"] == {"height": 4, "width": 8}


def test_npz_round_trip_is_safe_and_preserves_metadata(tmp_path):
    path = save_monthly_climatology(_reference(), tmp_path / "reference.npz")
    loaded = load_monthly_climatology(path)
    assert loaded.metadata["source"] == "synthetic test normal"
    assert np.array_equal(loaded.temperature_k, _reference().temperature_k.astype(np.float32))
    assert np.array_equal(
        loaded.precipitation_mm_day, _reference().precipitation_mm_day.astype(np.float32)
    )


def test_reference_requires_provenance_and_matching_grid():
    with pytest.raises(ValueError, match="source"):
        MonthlyClimatology(
            temperature_k=np.ones((12, 4, 8)),
            precipitation_mm_day=np.ones((12, 4, 8)),
            metadata={"period": "1991-2020"},
        )
    with pytest.raises(ValueError, match="does not match"):
        score_monthly_climatology(
            np.ones((12, 8, 16)), np.ones((12, 8, 16)), _reference()
        )
