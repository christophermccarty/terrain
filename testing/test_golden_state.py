"""test_golden_state.py -- bit-tolerance regression guard on simulate_step's output.

Would have caught the wind-cache and Ekman-scaling bugs found in the
2026-07-03 audit immediately as "something changed," independent of
understanding why -- both bugs were invisible to tests that only check
aggregate climate metrics, since a compensating constant or the bug's own
magnitude could still land within a wide metric tolerance.

If this test fails after a DELIBERATE physics change, regenerate the fixture:

    python scripts/generate_golden_state.py

...then commit the updated testing/fixtures/golden_state_reference.pkl
alongside the code change. Do NOT relax this test's tolerance to make it
pass -- that defeats its purpose.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer.headless import run_simulation
from simulate import load_state
from scripts.generate_golden_state import GOLDEN_STATE_CONFIG, FIXTURE_PATH

_ARRAY_OR_SCALAR_FIELDS = [
    "day_of_year", "total_days",
    "temperature", "air_temperature", "wind_u", "wind_v", "precipitation",
    "humidity", "soil_moisture", "soil_moisture_deep", "cloud_cover", "cloud_water", "snow_depth",
    "ice_cover", "co2_atmosphere", "co2_ocean", "vegetation_biomass",
    "salinity", "ch4_atmosphere", "permafrost_carbon", "T_deep_ocean",
    "ice_thickness", "biome_type", "koppen_type",
]


@pytest.fixture(scope="module")
def fresh_state():
    state, _ = run_simulation(**GOLDEN_STATE_CONFIG)
    return state


def test_golden_state_fixture_exists():
    assert FIXTURE_PATH.exists(), (
        f"{FIXTURE_PATH} is missing -- run `python scripts/generate_golden_state.py` "
        "once and commit the resulting fixture."
    )


def test_golden_state_matches_reference(fresh_state):
    reference = load_state(FIXTURE_PATH)
    mismatches = []
    for field in _ARRAY_OR_SCALAR_FIELDS:
        got = getattr(fresh_state, field)
        want = getattr(reference, field)
        if got is None and want is None:
            continue
        if (got is None) != (want is None):
            mismatches.append(f"{field}: one is None, other is not")
            continue
        try:
            np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-4, rtol=0)
        except AssertionError as e:
            mismatches.append(f"{field}: {e}")

    assert not mismatches, (
        "simulate_step's output drifted from the golden-state reference. If this "
        "is an intentional physics change, run `python scripts/generate_golden_state.py` "
        "and commit the updated fixture. Mismatched fields:\n" + "\n".join(mismatches)
    )


def test_golden_state_run_is_reproducible():
    """Two independent runs of the identical config must match each other, not
    just the committed fixture -- guards against hidden nondeterminism (e.g. an
    unseeded RNG) that a single fixture comparison alone wouldn't reveal."""
    state_a, _ = run_simulation(**GOLDEN_STATE_CONFIG)
    state_b, _ = run_simulation(**GOLDEN_STATE_CONFIG)
    for field in _ARRAY_OR_SCALAR_FIELDS:
        a, b = getattr(state_a, field), getattr(state_b, field)
        if a is None or b is None:
            continue
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=0, rtol=0)
