from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from atmosphere import _moisture_budget_precip_rescale, generate_precipitation
from planet_params import EARTH
from testing.conftest import make_mixed_elev


def test_budget_rescale_prefers_existing_condensation_and_obeys_caps():
    q = np.full((1, 4), 0.01, dtype=np.float32)
    dq = np.array([[0.0001, 0.0002, 0.0010, 0.0020]], dtype=np.float32)
    target = np.array([3.0], dtype=np.float32)

    result, diagnostics = _moisture_budget_precip_rescale(
        dq,
        q,
        target,
        dt_days=1.0,
        column_mm_per_q=2000.0,
    )

    added = result - dq
    assert added[0, 3] > added[0, 0]
    assert np.all(result <= 0.85 * q + 1e-9)
    assert np.all(added <= 0.15 * q + 1e-9)
    assert not diagnostics["capacity_limited"][0]


def test_budget_rescale_leaves_unmet_target_when_moisture_limited():
    q = np.full((2, 8), 1e-5, dtype=np.float32)
    dq = np.zeros_like(q)
    target = np.full(2, 20.0, dtype=np.float32)

    result, diagnostics = _moisture_budget_precip_rescale(
        dq,
        q,
        target,
        dt_days=1.0,
        column_mm_per_q=2000.0,
    )

    assert np.all(result <= 0.15 * q + 1e-12)
    assert np.all(diagnostics["capacity_limited"])
    assert np.all(diagnostics["unmet_row_mm_day"] > 0.0)


def test_budget_rescale_scales_excess_down_to_target():
    q = np.full((1, 6), 0.02, dtype=np.float32)
    dq = np.full((1, 6), 0.005, dtype=np.float32)
    target = np.array([2.0], dtype=np.float32)

    result, _ = _moisture_budget_precip_rescale(
        dq,
        q,
        target,
        dt_days=1.0,
        column_mm_per_q=2000.0,
    )

    precipitation = result * 2000.0
    assert float(np.mean(precipitation)) == pytest.approx(2.0, rel=1e-6)


def test_generate_precipitation_exposes_budget_diagnostics():
    H, W = 16, 32
    elevation = make_mixed_elev(H, W)
    temperature = np.full((H, W), 288.0, dtype=np.float32)
    wind_u = np.full((H, W), 4.0, dtype=np.float32)
    wind_v = np.zeros((H, W), dtype=np.float32)
    pp = dataclasses.replace(EARTH, moisture_budget_precip_rescale=True)
    debug: dict = {}

    precipitation, humidity, *_ = generate_precipitation(
        H,
        W,
        elevation,
        temperature=temperature,
        wind_u=wind_u,
        wind_v=wind_v,
        day_of_year=80.0,
        dt_days=1.0,
        planet_params=pp,
        debug_fields=debug,
    )

    assert np.all(np.isfinite(precipitation))
    assert np.all(precipitation >= 0.0)
    assert np.all(humidity >= 0.0)
    assert "precip_rescale_capacity_limited" in debug
    assert "precip_rescale_unmet_mm_day" in debug
    assert "precipitation_raw_mm_day" in debug
    assert "precipitation_final_mm_day" in debug
    assert "rainout_raw_dq" in debug
    assert debug["precip_rescale_capacity_limited"].shape == (H,)
    assert debug["precipitation_raw_mm_day"].shape == (H, W)
    assert debug["precipitation_final_mm_day"].shape == (H, W)
