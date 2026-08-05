"""test_sst_land_coupling.py -- SST -> adjacent-land precipitation coupling.

Covers `atmosphere._upwind_sst_anomaly` and the two consumers it feeds:
`sst_land_coupling_strength` (suppression side) and `sst_land_target_weight`
(moisture-budget target side).  See `docs/ACCURACY_AUDIT.md` D3.

Why the mechanism exists: D5 built real eastern-boundary upwelling and then
measured that adjacent land precipitation does not respond to it at any gyre
strength, establishing that per-cell SST anomalies do not reach land climate in
this model's atmosphere.  D3 records that the missing piece is the coupling, and
that it must be **shown to transmit** before more ocean physics is built.  These
tests are that demonstration, planted-anomaly style: a controlled cold patch of
ocean upwind of a land block must dry that land, and a warm patch must wet it.
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

import masks as masks_module
from atmosphere import _upwind_sst_anomaly, generate_precipitation
from masks import get_masks
from planet_params import EARTH

H, W = 48, 96


@pytest.fixture(autouse=True)
def _clear_mask_caches():
    masks_module.clear_all_caches()
    yield
    masks_module.clear_all_caches()


def _continent() -> np.ndarray:
    """One mid-latitude land block; ocean to its west, which is upwind here."""
    elev = np.zeros((H, W), dtype=np.float32)
    elev[16:32, 40:72] = 0.35
    return elev


def _westerly(speed: float = 8.0):
    u = np.full((H, W), speed, dtype=np.float32)
    v = np.zeros((H, W), dtype=np.float32)
    return u, v


def _uniform_sst(base: float = 291.0) -> np.ndarray:
    return np.full((H, W), base, dtype=np.float32)


def _precip(pp, temperature, u, v, elev):
    P, _, _, _ = generate_precipitation(
        H, W, elev,
        temperature=temperature, wind_u=u, wind_v=v,
        day_of_year=80.0, dt_days=1.0,
        surface_pressure_hpa=pp.surface_pressure_pa / 100.0,
        planet_params=pp,
    )
    return P


# --------------------------------------------------------------------------
# The anomaly field itself
# --------------------------------------------------------------------------

def test_anomaly_is_zero_when_the_ocean_is_uniform():
    """No structure in, no signal out -- the row-mean subtraction must be exact."""
    elev = _continent()
    sea, _ = get_masks(elev, use_cache=False)
    u, v = _westerly()
    dx = np.full((H, W), 3.0e5, dtype=np.float32)
    anomaly = _upwind_sst_anomaly(_uniform_sst(), sea.astype(np.float32), u, v,
                                  dx, 3.0e5, 600.0)
    assert np.allclose(anomaly, 0.0, atol=1e-4)


def test_anomaly_reaches_land_from_upwind_not_downwind():
    """Direction check, and the one a planted reversal must break.

    With westerly flow the ocean west of the continent is upwind, so a cold
    patch there must register on the land.  The same patch placed east of the
    continent -- downwind -- must not.
    """
    elev = _continent()
    sea, land = get_masks(elev, use_cache=False)
    u, v = _westerly()
    dx = np.full((H, W), 3.0e5, dtype=np.float32)
    sea_f = sea.astype(np.float32)

    upwind = _uniform_sst()
    upwind[16:32, 30:40] -= 4.0          # ocean immediately WEST of the land
    downwind = _uniform_sst()
    downwind[16:32, 72:82] -= 4.0        # ocean immediately EAST of the land

    west_edge = np.zeros((H, W), dtype=bool)
    west_edge[16:32, 40:44] = True
    west_edge &= land

    from_upwind = _upwind_sst_anomaly(upwind, sea_f, u, v, dx, 3.0e5, 600.0)
    from_downwind = _upwind_sst_anomaly(downwind, sea_f, u, v, dx, 3.0e5, 600.0)

    assert from_upwind[west_edge].mean() < -0.2, (
        "a cold patch upwind must reach the coastal land cells"
    )
    assert from_downwind[west_edge].mean() > from_upwind[west_edge].mean(), (
        "planted violation: the same patch downwind must not produce the same signal"
    )


def test_anomaly_decays_inland():
    elev = _continent()
    sea, land = get_masks(elev, use_cache=False)
    u, v = _westerly()
    dx = np.full((H, W), 3.0e5, dtype=np.float32)
    temperature = _uniform_sst()
    temperature[16:32, 24:40] -= 4.0
    anomaly = _upwind_sst_anomaly(temperature, sea.astype(np.float32), u, v,
                                  dx, 3.0e5, 900.0)
    row = 24
    assert abs(anomaly[row, 41]) > abs(anomaly[row, 55]), (
        "the ocean's influence must fall off with distance inland"
    )


# --------------------------------------------------------------------------
# Transmission: the property D3 exists to establish
# --------------------------------------------------------------------------

def test_target_side_transmits_with_the_right_sign():
    """Cold water upwind dries the land it feeds; warm water wets it.

    This is the measurement D3 asks for. It runs on the *target* side because
    that is the half that survives the moisture budget's row rescale -- process
    note 9. Its companion on the suppression side is measured inert on Earth and
    is pinned as such below.
    """
    elev = _continent()
    _, land = get_masks(elev, use_cache=False)
    u, v = _westerly()
    pp = dataclasses.replace(EARTH, sst_land_target_weight=1.5)

    cold = _uniform_sst()
    cold[16:32, 20:40] -= 4.0
    warm = _uniform_sst()
    warm[16:32, 20:40] += 4.0

    land_box = np.zeros((H, W), dtype=bool)
    land_box[16:32, 40:56] = True
    land_box &= land

    p_cold = _precip(pp, cold, u, v, elev)[land_box].mean()
    p_warm = _precip(pp, warm, u, v, elev)[land_box].mean()
    assert p_cold < p_warm, (
        f"cold upwind water must dry the land it feeds ({p_cold:.4f} vs "
        f"{p_warm:.4f} mm/day)"
    )


def test_target_weight_zero_is_an_exact_no_op():
    elev = _continent()
    u, v = _westerly()
    temperature = _uniform_sst()
    temperature[16:32, 20:40] -= 4.0
    off = dataclasses.replace(EARTH, sst_land_target_weight=0.0,
                              sst_land_coupling_strength=0.0)
    other_shape = dataclasses.replace(off, sst_land_coupling_km=2400.0)
    assert np.array_equal(
        _precip(off, temperature, u, v, elev),
        _precip(other_shape, temperature, u, v, elev),
    ), "the fetch knob must be inert while both strengths are zero"


def test_response_saturates_rather_than_scaling_with_kelvin():
    """A single outlier cell must not be able to claim the whole clip.

    The linear form shipped first and was withdrawn: the model's Kuroshio
    anomaly reaches +1.9 K where the land median is 0.35 K, and at a weight that
    was otherwise reasonable the S Japan box took 2.7x its row's target share and
    landed at 2809 mm/yr against a 1100-2200 target. `tanh` bounds the tail while
    leaving the sub-kelvin range where most coastal land sits nearly linear.
    """
    elev = _continent()
    _, land = get_masks(elev, use_cache=False)
    u, v = _westerly()
    pp = dataclasses.replace(EARTH, sst_land_target_weight=1.5)

    land_box = np.zeros((H, W), dtype=bool)
    land_box[16:32, 40:56] = True
    land_box &= land

    def dry(delta):
        temperature = _uniform_sst()
        temperature[16:32, 20:40] -= delta
        return float(_precip(pp, temperature, u, v, elev)[land_box].mean())

    baseline = dry(0.0)
    small = baseline - dry(2.0)
    large = baseline - dry(12.0)
    assert large > small > 0.0, "the drying response must still be monotonic"
    assert large < 3.0 * small, (
        f"a 6x colder anomaly must not give a proportionate response "
        f"({large:.4f} vs {small:.4f} mm/day) -- the tanh bound is missing"
    )


def test_warm_anomalies_do_not_boost_the_land_they_feed():
    """Cold side only -- the warm half would double-count two shipped mechanisms.

    Ocean evaporation already responds to SST through `qsat`, and
    `monsoon_east_margin_exemption` is calibrated at 3.0 for exactly the warm
    western-boundary-current margins (SE US, East China, S Japan) a symmetric
    form would boost again.  Measured, that double-count drove S Japan to
    2809 mm/yr against a 1100-2200 target at 256x512.

    Asserted on the mechanism's own factor rather than on final precipitation,
    and the difference is a real subtlety: a warm patch *does* raise the fed
    land's rainfall very slightly (+0.1% here), because it lifts the **row's
    ocean mean** and so pushes every other ocean cell in that row negative,
    suppressing land elsewhere and leaving this box a marginally larger share
    after renormalization.  That is an indirect consequence of the row-relative
    reference, not a warm-side boost, and pinning the factor is what
    distinguishes the two.
    """
    elev = _continent()
    u, v = _westerly()
    pp = dataclasses.replace(EARTH, sst_land_target_weight=1.5)

    warm = _uniform_sst()
    warm[16:32, 20:40] += 6.0

    debug: dict = {}
    generate_precipitation(
        H, W, elev, temperature=warm, wind_u=u, wind_v=v,
        day_of_year=80.0, dt_days=1.0,
        surface_pressure_hpa=pp.surface_pressure_pa / 100.0,
        planet_params=pp, debug_fields=debug,
    )
    factor = debug["sst_target_factor"]
    assert factor.max() <= 1.0 + 1e-6, (
        f"the target factor must never exceed 1.0 -- warm upwind water may not "
        f"claim extra share (max {factor.max():.4f})"
    )
    # The suppressing direction is covered by
    # `test_target_side_transmits_with_the_right_sign` and the saturation test;
    # asserting it here would only re-test the same branch on a fixture whose
    # land happens to see no net-cold fetch at all.
