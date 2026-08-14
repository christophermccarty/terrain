"""Versioned save envelope and cache-reset tests."""
from __future__ import annotations

import pickle
import json
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


def test_safe_npz_roundtrip_preserves_extended_state(tmp_path):
    from planet_params import EARTH
    from simulate import create_initial_state, load_state, save_state
    from testing.conftest import make_mixed_elev

    state = create_initial_state(
        make_mixed_elev(8, 16), day_of_year=80.0, planet_params=EARTH
    )._replace(
        total_days=42.5,
        surface_water_mm=np.full((8, 16), 3.0, dtype=np.float32),
        river_discharge_mm_day=np.full((8, 16), 0.5, dtype=np.float32),
        runoff_to_ocean_mm_day=np.full((8, 16), 0.1, dtype=np.float32),
        lower_pressure_depth_pa=np.full((8, 16), 40_000.0, dtype=np.float32),
        midlevel_pressure_depth_pa=np.full((8, 16), 35_000.0, dtype=np.float32),
        upperlevel_pressure_depth_pa=np.full((8, 16), 25_000.0, dtype=np.float32),
        lower_pressure_cloud_condensate=np.full((8, 16), 0.2, dtype=np.float32),
        midlevel_pressure_cloud_condensate=np.full((8, 16), 0.3, dtype=np.float32),
        upperlevel_pressure_cloud_condensate=np.full((8, 16), 0.1, dtype=np.float32),
        lower_pressure_hydrometeors=np.full((8, 16), 0.05, dtype=np.float32),
        midlevel_pressure_hydrometeors=np.full((8, 16), 0.07, dtype=np.float32),
        upperlevel_pressure_hydrometeors=np.full((8, 16), 0.03, dtype=np.float32),
    )
    path = tmp_path / "test.npz"
    save_state(state, path)
    loaded = load_state(path)

    assert loaded.total_days == pytest.approx(42.5)
    assert loaded.planet_params == EARTH
    for field in state._fields:
        expected = getattr(state, field)
        actual = getattr(loaded, field)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def test_safe_npz_rejects_unknown_state_fields(tmp_path):
    from simulate import load_state

    metadata = {
        "format": "planetsim-npz",
        "schema_version": 1,
        "scalars": {"day_of_year": 80.0, "intruder": 1},
        "none_fields": [],
        "planet_params": None,
    }
    path = tmp_path / "unknown.npz"
    np.savez(
        path,
        __metadata__=np.asarray(json.dumps(metadata)),
        elevation=np.zeros((4, 8), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="unknown fields"):
        load_state(path)


def test_safe_npz_never_enables_pickle_loading(tmp_path):
    from simulate import load_state

    metadata = {
        "format": "planetsim-npz",
        "schema_version": 1,
        "scalars": {"day_of_year": 80.0},
        "none_fields": [],
        "planet_params": None,
    }
    path = tmp_path / "object-array.npz"
    np.savez(
        path,
        __metadata__=np.asarray(json.dumps(metadata)),
        elevation=np.asarray([["not-safe"]], dtype=object),
    )
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_state(path)


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
