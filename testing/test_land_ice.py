"""test_land_ice.py — Phase 5 canvas item: land-ice mass balance, thickness,
flow, and the derived sea-level diagnostic.

Mirrors testing/test_ice_thickness.py's structure (that file covers *sea*
ice thickness/Stefan's-law growth; this covers *land* ice, which previously
had no mass, thickness, or flow at all -- only `ice_sheet_age`, a Koppen-EF
classification counter). See PlanetParams.enable_land_ice_dynamics for the
full design writeup.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import EARTH


def _step(state, days: float = 1.0, planet_params=None):
    from simulate import simulate_step
    return simulate_step(state, days=days, block_size=4, planet_params=planet_params)[0]


# ---------------------------------------------------------------------------
# Disabled by default: exact no-op
# ---------------------------------------------------------------------------

def test_disabled_by_default():
    assert EARTH.enable_land_ice_dynamics is False


def test_noop_when_disabled(mixed_initial_state):
    """With the master gate off, land_ice_thickness stays at zero and
    sea_level_change_m stays at 0.0, regardless of climate."""
    state = mixed_initial_state
    for _ in range(20):
        state = _step(state)
    assert state.land_ice_thickness is not None
    assert np.all(state.land_ice_thickness == 0.0)
    assert state.sea_level_change_m == 0.0


# ---------------------------------------------------------------------------
# Enabled: field present, finite, bounded
# ---------------------------------------------------------------------------

def test_field_present_and_finite_when_enabled(mixed_initial_state):
    pp = dataclasses.replace(EARTH, enable_land_ice_dynamics=True)
    state = mixed_initial_state
    for _ in range(20):
        state = _step(state, planet_params=pp)
    assert state.land_ice_thickness is not None
    assert state.land_ice_thickness.shape == state.elevation.shape
    assert np.all(np.isfinite(state.land_ice_thickness))
    assert np.all(state.land_ice_thickness >= 0.0)
    assert np.isfinite(state.sea_level_change_m)


def test_thickness_capped_at_max(mixed_initial_state):
    """A pre-seeded thickness above land_ice_max_thickness_m must be clipped
    down (not left over ceiling, and not grown further from an absurd
    starting point) after one step."""
    pp = dataclasses.replace(
        EARTH, enable_land_ice_dynamics=True, land_ice_max_thickness_m=100.0,
    )
    state = mixed_initial_state
    H, W = state.elevation.shape
    seeded = np.full((H, W), 500.0, dtype=np.float32)
    state = state._replace(land_ice_thickness=seeded)
    state = _step(state, planet_params=pp)
    assert np.all(state.land_ice_thickness <= 100.0 + 1e-3)


def test_flow_stable_at_high_thickness_and_monthly_dt(mixed_initial_state):
    """Regression guard for a real bug found via a real-terrain (512x1024)
    check: seeding an Antarctic-scale reservoir (~2000 m) and stepping at
    MONTHLY cadence (dt=30.44) with the default flow diffusivity drove the
    substep count past its old hard cap without shrinking the effective
    diffusivity to match, silently violating the diffusion scheme's CFL
    stability limit and overflowing thickness to NaN within one step."""
    pp = dataclasses.replace(EARTH, enable_land_ice_dynamics=True)
    H, W = mixed_initial_state.elevation.shape
    seeded = np.full((H, W), 2000.0, dtype=np.float32)
    state = mixed_initial_state._replace(land_ice_thickness=seeded)
    state = _step(state, days=30.44, planet_params=pp)
    assert np.all(np.isfinite(state.land_ice_thickness))
    assert np.isfinite(state.sea_level_change_m)


def test_zero_over_ocean(mixed_initial_state):
    from masks import get_masks
    pp = dataclasses.replace(EARTH, enable_land_ice_dynamics=True)
    state = mixed_initial_state
    H, W = state.elevation.shape
    seeded = np.full((H, W), 50.0, dtype=np.float32)
    state = state._replace(land_ice_thickness=seeded)
    for _ in range(5):
        state = _step(state, planet_params=pp)
    sea_mask, _ = get_masks(state.elevation)
    assert np.all(state.land_ice_thickness[sea_mask] == 0.0)


# ---------------------------------------------------------------------------
# Wiring: enabling the gate must change the resulting state
# ---------------------------------------------------------------------------

def test_enabling_changes_state_from_seeded_ice(mixed_initial_state):
    """A fresh spin-up never approaches the 10 m snow-depth overflow cap in
    a short run, so this seeds a nonzero starting reservoir directly (the
    same reason test_param_wiring.py's shared 15-day fresh-start harness
    can't exercise this parameter either -- it needs bootstrapped state)."""
    H, W = mixed_initial_state.elevation.shape
    seeded = np.full((H, W), 20.0, dtype=np.float32)
    base_state = mixed_initial_state._replace(land_ice_thickness=seeded)

    pp_off = EARTH
    pp_on = dataclasses.replace(EARTH, enable_land_ice_dynamics=True)

    state_off = base_state
    state_on = base_state
    for _ in range(10):
        state_off = _step(state_off, planet_params=pp_off)
        state_on = _step(state_on, planet_params=pp_on)

    assert not np.allclose(state_off.land_ice_thickness, state_on.land_ice_thickness), (
        "enable_land_ice_dynamics=True vs False produced identical thickness "
        "from a seeded reservoir -- parameter may be unwired"
    )


def test_ice_melts_under_warm_conditions(mixed_initial_state):
    """Seed a mid-latitude land bump and step forward; a warmer melt factor
    should remove more mass than a colder one, all else equal."""
    H, W = mixed_initial_state.elevation.shape
    seeded = np.full((H, W), 5.0, dtype=np.float32)
    base_state = mixed_initial_state._replace(land_ice_thickness=seeded)

    pp_low_melt = dataclasses.replace(
        EARTH, enable_land_ice_dynamics=True, ice_melt_degree_day_mm=1.0,
        ice_flow_diffusivity=0.0,
    )
    pp_high_melt = dataclasses.replace(
        EARTH, enable_land_ice_dynamics=True, ice_melt_degree_day_mm=20.0,
        ice_flow_diffusivity=0.0,
    )

    state_low = base_state
    state_high = base_state
    for _ in range(10):
        state_low = _step(state_low, planet_params=pp_low_melt)
        state_high = _step(state_high, planet_params=pp_high_melt)

    assert float(np.sum(state_high.land_ice_thickness)) <= float(np.sum(state_low.land_ice_thickness)), (
        "a higher degree-day melt factor did not remove more ice mass"
    )


# ---------------------------------------------------------------------------
# Flow kernel: direct unit tests (mirrors test_ice_thickness.py calling
# ocean.update_sea_ice directly instead of going through simulate_step)
# ---------------------------------------------------------------------------

def test_flow_spreads_a_thickness_bump():
    from simulate import _land_ice_flow_step

    H, W = 16, 32
    land_mask = np.ones((H, W), dtype=bool)
    thickness = np.zeros((H, W), dtype=np.float32)
    thickness[8, 16] = 100.0

    result = thickness.copy()
    for _ in range(20):
        result = _land_ice_flow_step(result, land_mask, k=1.0e-2, dt=0.1)

    assert result[8, 16] < thickness[8, 16], "peak did not decrease under diffusion"
    assert result[8, 17] > 0.0 and result[7, 16] > 0.0, "ice did not spread to neighbors"
    assert np.all(result >= 0.0), "flow produced negative thickness"


def test_flow_conserves_mass_in_the_interior():
    """Away from the grid poles (which use a clamped/mirrored boundary, not
    periodic), thickness-weighted diffusion should conserve total mass."""
    from simulate import _land_ice_flow_step

    H, W = 32, 32
    land_mask = np.ones((H, W), dtype=bool)
    thickness = np.zeros((H, W), dtype=np.float32)
    thickness[H // 2, W // 2] = 100.0

    total_before = float(np.sum(thickness))
    result = _land_ice_flow_step(thickness, land_mask, k=1.0e-3, dt=1.0)
    total_after = float(np.sum(result))
    assert total_after == pytest.approx(total_before, rel=1e-4)


def test_flow_discards_mass_at_coastline():
    """Ice diffusing toward an ocean neighbor should lose mass (calving
    proxy), not pile up or leak into the ocean cell's own thickness."""
    from simulate import _land_ice_flow_step

    H, W = 8, 8
    land_mask = np.ones((H, W), dtype=bool)
    land_mask[:, 4:] = False  # right half is "ocean"
    thickness = np.zeros((H, W), dtype=np.float32)
    thickness[:, 3] = 50.0  # ice right at the coast

    total_before = float(np.sum(thickness))
    result = thickness.copy()
    for _ in range(10):
        result = _land_ice_flow_step(result, land_mask, k=5.0e-3, dt=1.0)

    assert np.all(result[:, 4:] == 0.0), "ocean cells must stay at exactly zero thickness"
    assert float(np.sum(result)) < total_before, "coastal mass loss (calving proxy) did not occur"


def test_zero_diffusivity_leaves_thickness_field_flat():
    from simulate import _land_ice_flow_step

    H, W = 8, 8
    land_mask = np.ones((H, W), dtype=bool)
    thickness = np.zeros((H, W), dtype=np.float32)
    thickness[4, 4] = 10.0

    result = _land_ice_flow_step(thickness, land_mask, k=0.0, dt=1.0)
    assert np.array_equal(result, thickness)


# ---------------------------------------------------------------------------
# Sea-level diagnostic
# ---------------------------------------------------------------------------

def test_sea_level_falls_as_land_ice_grows(mixed_initial_state):
    """Growing land ice locks up water that would otherwise be ocean, so
    sea_level_change_m (defined negative = sea level fell) should trend
    negative as the reservoir accumulates from zero."""
    pp = dataclasses.replace(
        EARTH, enable_land_ice_dynamics=True, ice_flow_diffusivity=0.0,
        ice_melt_degree_day_mm=0.0,
    )
    H, W = mixed_initial_state.elevation.shape
    # Force a large accumulation directly via a pre-saturated snowpack so the
    # 10 m overflow mechanism actually triggers within a short run.
    from masks import get_masks
    _, land_mask = get_masks(mixed_initial_state.elevation)
    seeded_ice = np.where(land_mask, 30.0, 0.0).astype(np.float32)
    state = mixed_initial_state._replace(land_ice_thickness=seeded_ice)
    state = _step(state, planet_params=pp)
    assert state.sea_level_change_m <= 0.0
    assert state.sea_level_change_m < -1e-9, "sea_level_change_m did not respond to a large seeded ice reservoir"


def test_sea_level_zero_with_zero_ice(mixed_initial_state):
    pp = dataclasses.replace(EARTH, enable_land_ice_dynamics=True)
    state = mixed_initial_state
    for _ in range(3):
        state = _step(state, planet_params=pp)
    # A fresh spin-up never reaches the snow overflow threshold this fast.
    assert state.sea_level_change_m == pytest.approx(0.0, abs=1e-6)
