"""test_upper_layer_wind.py -- 1.5-layer atmosphere: real prognostic upper-level wind.

Replaces the old magnitude-only `baroclinic_jet_amp * jet_window * |dT/dy|`
surface-jet hack with a genuine, independent upper-level momentum budget
(`atmosphere.evolve_wind_aloft`) coupled back to the surface via a real
per-cell relaxation term in `evolve_wind`. Unlike the old hack, this has an
actual sign/direction -- the property these tests are built around, since
`|dT/dy|` structurally could never distinguish a normal pole-equator gradient
from a reversed one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import atmosphere as atmo
from planet_params import EARTH
from simulate import PlanetState, create_initial_state, simulate_step, save_state, load_state


def _lat_deg_1d(H: int) -> np.ndarray:
    return (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * 180.0


def _idealized_temperature(H: int, W: int, reversed_gradient: bool = False) -> np.ndarray:
    lat_deg = _lat_deg_1d(H)
    if reversed_gradient:
        # Unrealistic (warm poles, cold equator) -- purely to test that the
        # jet's sign follows the gradient, not a hardcoded hemisphere convention.
        T_1d = 288.0 - 45.0 * np.cos(np.deg2rad(lat_deg)) ** 2
    else:
        T_1d = 288.0 - 45.0 * np.sin(np.deg2rad(lat_deg)) ** 2
    return np.repeat(T_1d[:, None], W, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# evolve_wind_aloft: directional sign correctness (unit-level, fast)
# ---------------------------------------------------------------------------

def test_upper_layer_jet_is_westerly_for_normal_gradient():
    """A realistic warm-equator/cold-pole profile must produce a westerly
    (positive u) jet core in the mid-latitude band on both hemispheres."""
    H, W = 32, 64
    T = _idealized_temperature(H, W, reversed_gradient=False)
    u2, v2 = np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)
    for _ in range(150):
        u2, v2 = atmo.evolve_wind_aloft(u2, v2, T, dt_days=1.0, planet_params=EARTH)

    lat_deg = _lat_deg_1d(H)
    jet_band = (np.abs(lat_deg) >= 40.0) & (np.abs(lat_deg) <= 55.0)
    assert np.all(np.isfinite(u2)) and np.all(np.isfinite(v2))
    assert u2[jet_band].mean() > 0.5, "expected a clear westerly jet core at 40-55 deg"


def test_upper_layer_jet_sign_flips_with_reversed_gradient():
    """The defining property the old |dT/dy| hack could never have: reversing
    the temperature gradient must reverse the jet's sign, not just its
    magnitude."""
    H, W = 32, 64
    T_normal = _idealized_temperature(H, W, reversed_gradient=False)
    T_reversed = _idealized_temperature(H, W, reversed_gradient=True)

    lat_deg = _lat_deg_1d(H)
    jet_band = (np.abs(lat_deg) >= 40.0) & (np.abs(lat_deg) <= 55.0)

    u2n, v2n = np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)
    u2r, v2r = np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)
    for _ in range(150):
        u2n, v2n = atmo.evolve_wind_aloft(u2n, v2n, T_normal, dt_days=1.0, planet_params=EARTH)
        u2r, v2r = atmo.evolve_wind_aloft(u2r, v2r, T_reversed, dt_days=1.0, planet_params=EARTH)

    assert u2n[jet_band].mean() > 0.5
    assert u2r[jet_band].mean() < -0.5, "reversed gradient should produce an easterly, not just a weaker westerly"


def test_upper_layer_zero_amplitude_is_calm():
    """upper_pgf_amp=0.0 must not spontaneously generate a jet (pure inertia
    from rest, no forcing)."""
    H, W = 24, 48
    T = _idealized_temperature(H, W)
    u2, v2 = np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)
    for _ in range(30):
        u2, v2 = atmo.evolve_wind_aloft(u2, v2, T, dt_days=1.0, upper_pgf_amp=0.0, planet_params=EARTH)
    assert np.allclose(u2, 0.0, atol=1e-6)
    assert np.allclose(v2, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Full-system integration: aloft should be stronger than the surface at jet
# latitudes (real tropospheric jets are stronger aloft), and the coupling
# should measurably affect the surface (on/off check).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_upper_layer_stronger_than_surface_at_jet_band(earth_spinup_state):
    H = earth_spinup_state.temperature.shape[0]
    lat_deg = _lat_deg_1d(H)
    jet_band = (np.abs(lat_deg) >= 40.0) & (np.abs(lat_deg) <= 55.0)

    speed_surface = np.hypot(earth_spinup_state.wind_u, earth_spinup_state.wind_v)
    speed_aloft = np.hypot(earth_spinup_state.wind_u_aloft, earth_spinup_state.wind_v_aloft)

    assert np.all(np.isfinite(speed_aloft))
    assert speed_aloft[jet_band].mean() > speed_surface[jet_band].mean(), (
        f"aloft jet-band speed {speed_aloft[jet_band].mean():.2f} m/s should exceed "
        f"surface {speed_surface[jet_band].mean():.2f} m/s"
    )


@pytest.mark.slow
def test_upper_layer_coupling_strengthens_surface_westerlies(mixed_elev):
    """Disabling the aloft-to-surface mixing (wind_baroclinic_jet_amp=0)
    should leave the surface mid-lat jet weaker than with it enabled --
    the on/off wiring check for the new coupling term."""
    def _spinup(**kwargs):
        state = create_initial_state(mixed_elev, day_of_year=80.0)
        for _ in range(200):
            state, _ = simulate_step(state, days=1.0, block_size=4, wind_block_size=4, **kwargs)
        return state

    state_on = _spinup()
    state_off = _spinup(wind_baroclinic_jet_amp=0.0)

    H = state_on.temperature.shape[0]
    lat_deg = _lat_deg_1d(H)
    jet_band = (np.abs(lat_deg) >= 40.0) & (np.abs(lat_deg) <= 55.0)

    u_on = state_on.wind_u[jet_band].mean()
    u_off = state_off.wind_u[jet_band].mean()
    assert u_on > u_off, (
        f"surface mid-lat westerly should be stronger with aloft coupling on "
        f"(on={u_on:.2f}, off={u_off:.2f})"
    )


# ---------------------------------------------------------------------------
# Save/load round-trip with an old-format state (missing wind_u_aloft/v_aloft)
# ---------------------------------------------------------------------------

def test_old_format_state_loads_and_lazy_inits(tmp_path, mixed_elev):
    """A PlanetState pickled before this field existed (simulated here by
    constructing one without wind_u_aloft/wind_v_aloft, relying on the
    NamedTuple defaults) must load and step without error, lazily
    initializing the new fields on first use."""
    old_state = PlanetState(
        day_of_year=80.0,
        elevation=mixed_elev,
        temperature=None,
        wind_u=None,
        wind_v=None,
        # wind_u_aloft / wind_v_aloft intentionally omitted -> NamedTuple default (None)
    )
    assert old_state.wind_u_aloft is None
    assert old_state.wind_v_aloft is None

    path = tmp_path / "old_format_state.pkl"
    save_state(old_state, path)
    loaded = load_state(path)
    assert loaded.wind_u_aloft is None

    stepped, _ = simulate_step(loaded, days=1.0, block_size=4, wind_block_size=4)
    assert stepped.wind_u_aloft is not None
    assert np.all(np.isfinite(stepped.wind_u_aloft))
    assert np.all(np.isfinite(stepped.wind_v_aloft))
