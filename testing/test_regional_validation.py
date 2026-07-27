from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from regional_validation import (
    EARTH_PRECIP_REGIONS,
    ClimateRegion,
    precipitation_by_region_mm_year,
    region_mask,
    target_error_fraction,
)


def test_region_mask_uses_cell_centers_and_supports_dateline():
    region = ClimateRegion(
        "Dateline", 10.0, -10.0, 170.0, -170.0, "control", 0.0, 1.0
    )
    mask = region_mask((18, 36), region)
    assert np.any(mask)
    assert np.all(mask[:, 18] == 0)  # Greenwich-side cell is excluded.
    assert np.any(mask[:, 0])
    assert np.any(mask[:, -1])


def test_precipitation_regions_apply_land_mask_and_planet_year():
    shape = (180, 360)
    precipitation = np.ones(shape, dtype=np.float32)
    land = np.ones(shape, dtype=bool)
    values = precipitation_by_region_mm_year(
        precipitation,
        land_mask=land,
        days_per_year=400.0,
    )
    assert set(values) == {region.name for region in EARTH_PRECIP_REGIONS}
    assert all(value == pytest.approx(400.0) for value in values.values())


@pytest.mark.parametrize(
    "value,expected",
    [
        (900.0, 0.0),
        (0.0, 0.8),
        (1200.0, 0.2),
    ],
)
def test_target_error_fraction(value, expected):
    midwest = next(r for r in EARTH_PRECIP_REGIONS if r.name == "US Midwest")
    assert target_error_fraction(value, midwest) == pytest.approx(expected)


def test_bundled_real_dem_covers_every_validation_region():
    """Guard the geography used by future regional climate gates."""
    dem_path = Path(__file__).resolve().parents[1] / "images" / "16_bit_dem_small_512.tif"
    raw = np.asarray(Image.open(dem_path), dtype=np.float32)
    ocean_threshold = 8070.0
    elevation = np.maximum(
        0.0, (raw - ocean_threshold) / (float(np.max(raw)) - ocean_threshold)
    )
    land = elevation > 0.0

    missing = [
        region.name
        for region in EARTH_PRECIP_REGIONS
        if not np.any(region_mask(elevation.shape, region, cell_mask=land))
    ]
    assert not missing, f"Bundled DEM has no land cells in: {missing}"
