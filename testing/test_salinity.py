"""Tests for Feature 3: Ocean salinity and AMOC freshwater coupling."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_salinity_init_in_state(mixed_initial_state):
    """After first step, salinity should be initialized: ocean ~35 PSU, land = 0."""
    from simulate import simulate_step
    state, _ = simulate_step(mixed_initial_state, days=1.0)
    sal = state.salinity
    assert sal is not None, "salinity field must be present after step"

    from masks import get_masks
    sea_mask, land_mask = get_masks(state.elevation)
    ocean_mean = float(np.mean(sal[sea_mask]))
    land_vals = sal[land_mask]
    assert 30.0 <= ocean_mean <= 40.0, f"Ocean salinity init: {ocean_mean:.2f} PSU"
    assert float(np.max(land_vals)) < 0.01, f"Land salinity should be 0, got {float(np.max(land_vals)):.3f}"


def test_salinity_ep_balance(mixed_initial_state):
    """evolve_salinity: high evaporation should raise salinity; heavy rain should lower it."""
    from ocean import evolve_salinity
    from planet_params import EARTH
    from masks import get_masks

    H, W = mixed_initial_state.elevation.shape
    sea_mask, _ = get_masks(mixed_initial_state.elevation)

    sal = np.where(sea_mask, 35.0, 0.0).astype(np.float32)
    T_warm = np.full((H, W), 295.0, dtype=np.float32)
    T_cold = np.full((H, W), 273.0, dtype=np.float32)
    P_dry = np.zeros((H, W), dtype=np.float32)
    P_wet = np.full((H, W), 20.0, dtype=np.float32)
    ice_zero = np.zeros((H, W), dtype=np.float32)

    # Warm + dry → evaporation → salinity rises
    sal_evap = evolve_salinity(sal, T_warm, mixed_initial_state.elevation, P_dry, ice_zero, dt_days=30.0, pp=EARTH)
    # Cold + wet → precipitation dominates → salinity falls
    sal_rain = evolve_salinity(sal, T_cold, mixed_initial_state.elevation, P_wet, ice_zero, dt_days=30.0, pp=EARTH)

    ocean_evap = float(np.mean(sal_evap[sea_mask]))
    ocean_rain = float(np.mean(sal_rain[sea_mask]))
    assert ocean_evap > 35.0, f"Warm/dry should raise salinity above 35: {ocean_evap:.2f}"
    assert ocean_rain < 35.0, f"Cold/wet should lower salinity below 35: {ocean_rain:.2f}"


def test_salinity_brine_rejection():
    """Freezing (positive ice_delta) should export salt to adjacent ocean cells."""
    from ocean import evolve_salinity
    from planet_params import EARTH
    from testing.conftest import make_mixed_elev

    H, W = 32, 64
    elev = make_mixed_elev(H, W)

    from masks import get_masks
    sea_mask, _ = get_masks(elev)

    sal = np.where(sea_mask, 35.0, 0.0).astype(np.float32)
    T_freeze = np.full((H, W), 271.0, dtype=np.float32)
    P_zero = np.zeros((H, W), dtype=np.float32)

    # Simulate freezing: ice_delta > 0 over ocean
    ice_delta = np.where(sea_mask, 0.3, 0.0).astype(np.float32)
    sal_after = evolve_salinity(sal, T_freeze, elev, P_zero, ice_delta, dt_days=30.0, pp=EARTH)

    ocean_sal_before = float(np.mean(sal[sea_mask]))
    ocean_sal_after = float(np.mean(sal_after[sea_mask]))
    # Brine rejection raises salinity
    assert ocean_sal_after > ocean_sal_before, (
        f"Brine rejection should raise salinity: before={ocean_sal_before:.2f}, after={ocean_sal_after:.2f}"
    )


def test_salinity_stays_in_bounds():
    """Salinity must remain in [0, 45] PSU under all tested conditions."""
    from ocean import evolve_salinity
    from planet_params import EARTH
    from testing.conftest import make_mixed_elev

    H, W = 32, 64
    elev = make_mixed_elev(H, W)
    from masks import get_masks
    sea_mask, _ = get_masks(elev)

    sal = np.where(sea_mask, 35.0, 0.0).astype(np.float32)
    T = np.full((H, W), 300.0, dtype=np.float32)  # extreme evaporation
    P = np.full((H, W), 100.0, dtype=np.float32)  # extreme rain too
    ice_delta = np.where(sea_mask, 1.0, 0.0).astype(np.float32)  # extreme brine

    for _ in range(10):
        sal = evolve_salinity(sal, T, elev, P, ice_delta, dt_days=30.0, pp=EARTH)

    assert float(np.min(sal)) >= 0.0
    assert float(np.max(sal)) <= 45.0
