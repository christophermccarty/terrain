"""Versioned save envelope and cache-reset tests."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_save_load_envelope_roundtrip(tmp_path):
    from planet_params import EARTH
    from simulate import STATE_SCHEMA_VERSION, create_initial_state, load_state, save_state
    from testing.conftest import make_mixed_elev

    state = create_initial_state(make_mixed_elev(8, 16), day_of_year=80.0, planet_params=EARTH)
    state = state._replace(planet_params=EARTH, total_days=42.0)
    path = tmp_path / "test.pkl"
    save_state(state, path)

    with open(path, "rb") as f:
        envelope = pickle.load(f)
    assert envelope["schema_version"] == STATE_SCHEMA_VERSION
    assert envelope["planet_params"].solar_constant == pytest.approx(EARTH.solar_constant)

    loaded = load_state(path)
    assert loaded.total_days == pytest.approx(42.0)
    assert loaded.planet_params is not None


def test_legacy_raw_pickle_still_loads(tmp_path):
    from simulate import create_initial_state, load_state
    from testing.conftest import make_mixed_elev

    state = create_initial_state(make_mixed_elev(8, 16), day_of_year=80.0)
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as f:
        pickle.dump(state, f)
    loaded = load_state(path)
    assert loaded.elevation.shape == state.elevation.shape


def test_clear_simulation_caches_resets_mask_cache():
    from masks import clear_all_caches, get_masks
    from simulate import clear_simulation_caches
    from testing.conftest import make_mixed_elev

    elev = make_mixed_elev(8, 16)
    get_masks(elev)
    clear_simulation_caches()
    clear_all_caches()
    # Should recompute without stale id() hits after explicit clear
    sea1, _ = get_masks(elev)
    elev2 = elev.copy()
    sea2, _ = get_masks(elev2, use_cache=False)
    np.testing.assert_array_equal(sea1, sea2)
