"""Tests for Feature 4: CH4 radiative forcing and permafrost carbon."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_ch4_forcing_magnitude():
    """Modern CH4 (1900 ppb) vs pre-industrial (700 ppb) forcing should be ~0.45–0.70 W/m²."""
    from carbon_cycle import ch4_radiative_forcing
    forcing = ch4_radiative_forcing(1900.0, 700.0)
    assert 0.45 <= forcing <= 0.70, f"CH4 forcing: {forcing:.3f} W/m²"


def test_ch4_lifetime():
    """CH4 should decay by ~1/e in one lifetime (9yr = 3287 days)."""
    from carbon_cycle import ch4_oxidation_step
    initial = 1900.0
    after = ch4_oxidation_step(initial, dt_days=3287.0)
    ratio = after / initial
    # 1/e ≈ 0.368; allow ±5% tolerance
    assert 0.33 <= ratio <= 0.41, f"CH4 1-lifetime ratio: {ratio:.3f} (expected ~0.368)"


def test_permafrost_stable_at_cold():
    """Cold Arctic permafrost (T < 273K) should release < 0.5% carbon over 10yr."""
    from carbon_cycle import permafrost_thaw_step
    from testing.conftest import make_mixed_elev

    H, W = 32, 64
    elev = make_mixed_elev(H, W)
    T_cold = np.full((H, W), 260.0, dtype=np.float32)  # Well below freezing

    # Initialize permafrost everywhere (simplified test)
    pfc = np.full((H, W), 30.0, dtype=np.float32)
    total_initial = float(np.sum(pfc))

    snow = np.full((H, W), 0.5, dtype=np.float32)
    dt = 365.0 * 10  # 10 years

    pfc_new, d_co2, d_ch4 = permafrost_thaw_step(pfc, T_cold, snow, dt)
    total_released = total_initial - float(np.sum(pfc_new))
    frac_released = total_released / total_initial
    assert frac_released < 0.005, f"Cold permafrost released {frac_released*100:.2f}% (should be < 0.5%)"


def test_permafrost_releases_on_warming():
    """Warm Arctic (T > 275K) should release measurable carbon within 20yr."""
    from carbon_cycle import permafrost_thaw_step
    from testing.conftest import make_mixed_elev

    H, W = 32, 64
    pfc = np.full((H, W), 30.0, dtype=np.float32)
    T_warm = np.full((H, W), 278.0, dtype=np.float32)  # 5°C above freezing
    snow_none = np.zeros((H, W), dtype=np.float32)

    d_co2_total = 0.0
    d_ch4_total = 0.0
    for _ in range(20):
        pfc, d_co2, d_ch4 = permafrost_thaw_step(pfc, T_warm, snow_none, dt_days=365.0)
        d_co2_total += d_co2
        d_ch4_total += d_ch4

    assert d_co2_total > 0.001, f"CO2 release should be > 0 after 20yr warming: {d_co2_total:.4f} ppm"
    assert d_ch4_total > 0.0, f"CH4 release should be > 0 after 20yr warming"


def test_wetland_emissions_warm_wet():
    """Warm, wet land cells should emit CH4; cold/dry should emit near zero."""
    from carbon_cycle import wetland_ch4_emissions

    H, W = 32, 64
    land_mask = np.ones((H, W), dtype=bool)

    T_warm = np.full((H, W), 303.0, dtype=np.float32)  # 30°C
    T_cold = np.full((H, W), 255.0, dtype=np.float32)  # -18°C
    sm_wet = np.full((H, W), 0.9, dtype=np.float32)
    sm_dry = np.zeros((H, W), dtype=np.float32)

    em_warm_wet = wetland_ch4_emissions(T_warm, sm_wet, land_mask, dt_days=365.0)
    em_cold_dry = wetland_ch4_emissions(T_cold, sm_dry, land_mask, dt_days=365.0)

    assert em_warm_wet > 0.01, f"Warm/wet wetland emission too small: {em_warm_wet:.4f} ppb/yr"
    assert em_cold_dry < 0.001, f"Cold/dry should emit ~0: {em_cold_dry:.6f} ppb/yr"
