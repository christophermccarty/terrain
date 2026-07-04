"""test_param_wiring.py -- catch parameters that are accepted but never read.

The 2026-07-03 audit found several parameters that looked tunable but did
nothing: `update_wind` was silently ignored in MONTHLY/ANNUAL mode,
`ocean_exchange_coeff` was hardcoded past (the optimizer thought it was
sweeping 0.08 while the real value stayed 0.03), and `co2_initial_ppm`/
`ch4_initial_ppb` were only applied by the optimizer's headless runner, never
by `create_initial_state` itself. None of these were caught by existing tests
because those tests check aggregate climate metrics, not "does this parameter
change anything at all."

Pattern: run a short simulation at a parameter's default vs. a substantially
perturbed value (same seed/elevation/day count) and assert the resulting
`PlanetState` differs in at least one field beyond a noise floor. A parameter
that produces a byte-identical state across a large perturbation is either
dead code or not wired through -- both worth failing loudly on.
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
from simulate import create_initial_state, simulate_step
from optimizer.headless import _make_default_elevation

H, W = 24, 48
N_STEPS = 15  # long enough for transient mechanisms (storms, trade waves) to be active

_COMPARE_FIELDS = [
    "temperature", "air_temperature", "wind_u", "wind_v", "precipitation",
    "humidity", "cloud_cover", "soil_moisture", "salinity", "ice_cover",
    "co2_atmosphere", "ch4_atmosphere", "co2_ocean",
]


def _run(planet_params, **step_kwargs):
    elev = _make_default_elevation(H, W)
    state = create_initial_state(elev, planet_params=planet_params)
    for _ in range(N_STEPS):
        state, _ = simulate_step(state, days=1.0, planet_params=planet_params, **step_kwargs)
    return state


def _states_differ(a, b, atol: float = 1e-5) -> bool:
    for field in _COMPARE_FIELDS:
        va, vb = getattr(a, field), getattr(b, field)
        if va is None or vb is None:
            continue
        if not np.allclose(np.asarray(va), np.asarray(vb), atol=atol, rtol=0):
            return True
    return False


# ---------------------------------------------------------------------------
# simulate_step kwargs (forwarded directly, not via PlanetParams)
# ---------------------------------------------------------------------------

STEP_KWARG_CASES = [
    ("ocean_exchange_coeff", 0.30),
    ("ocean_exchange_inertia", 0.0),
    ("wind_damping", 0.15),
    ("wind_baroclinic_jet_amp", 5.0e6),
    ("ice_freeze_rate", 0.20),
    ("ice_melt_rate", 0.60),
    # NOT included: latent_cooling_coeff. This test found it to be a genuine
    # dead parameter (accepted by simulate_step, read nowhere in the
    # codebase) -- see the comment at its definition in simulate.py. Left
    # undocumented-fixed rather than newly wired, per project convention for
    # deprecated no-ops (e.g. ocean_exchange_floor/span).
]


@pytest.mark.parametrize("name,perturbed", STEP_KWARG_CASES)
def test_simulate_step_kwarg_is_wired(name, perturbed):
    baseline = _run(EARTH)
    changed = _run(EARTH, **{name: perturbed})
    assert _states_differ(baseline, changed), (
        f"simulate_step kwarg {name!r}=default vs {name!r}={perturbed} produced "
        f"a byte-identical state after {N_STEPS} days -- parameter may be dead/unwired"
    )


# ---------------------------------------------------------------------------
# PlanetParams fields (need a modified PlanetParams instance)
# ---------------------------------------------------------------------------

PLANET_PARAM_CASES = [
    ("eddy_heat_flux_coeff", 0.05),
    ("storm_pressure_amp_pa", 0.0),
    ("trade_wave_pressure_amp_pa", 0.0),
    ("ekman_strength", 0.0),
    ("cloud_greenhouse_factor", 0.0),
    ("wv_greenhouse_factor", 0.0),
    ("deep_ocean_exchange_rate", 5e-4),
]


@pytest.mark.parametrize("field,perturbed", PLANET_PARAM_CASES)
def test_planet_param_is_wired(field, perturbed):
    changed_pp = dataclasses.replace(EARTH, **{field: perturbed})
    baseline = _run(EARTH)
    changed = _run(changed_pp)
    assert _states_differ(baseline, changed), (
        f"PlanetParams.{field}=default vs {perturbed} produced a byte-identical "
        f"state after {N_STEPS} days -- parameter may be dead/unwired"
    )


# ---------------------------------------------------------------------------
# create_initial_state must actually seed from PlanetParams (2026-07-03 fix)
# ---------------------------------------------------------------------------

def test_co2_ch4_initial_values_applied():
    # create_initial_state also runs one simulate_step to populate derived
    # fields, so a tiny (<1 day of natural CH4 source/sink drift) tolerance
    # is expected -- this is not testing bit-exact seeding, just that the
    # value was actually used as the starting point rather than ignored.
    elev = _make_default_elevation(H, W)
    custom = dataclasses.replace(EARTH, co2_initial_ppm=777.0, ch4_initial_ppb=1234.0)
    state = create_initial_state(elev, planet_params=custom)
    assert state.co2_atmosphere == pytest.approx(777.0, abs=0.5)
    assert state.ch4_atmosphere == pytest.approx(1234.0, abs=0.5)
