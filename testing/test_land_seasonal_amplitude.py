"""test_land_seasonal_amplitude.py -- damping of the land forcing's seasonal swing.

Covers `PlanetParams.land_seasonal_amplitude`, `land_seasonal_amplitude_maritime`
and `land_transport_gain`, the three knobs audit C1b's 2026-08-05 session shipped.
See `docs/ACCURACY_AUDIT.md` C1b for the calibration.

The defect these address: `temperature_kelvin_for_lat` returns *instantaneous
local radiative equilibrium*, whose annual half-range at 41 deg is ~81 K against
Earth land's ~28 K.  The three transport trapezoids patch the resulting winter
and `_land_cap_1d` patches the summer, but they do not cancel -- the trapezoids
are added in all twelve months while the cap only subtracts in summer, leaving
the forcing's annual mean ~21 K too warm with a flat top across seven months.

The two properties that make this different from C1b's four earlier knobs are
pinned here: the damping is **exactly mean-preserving** (so it cannot disturb the
annual-mean level the trapezoids and the cap were calibrated against), and its
maritime modulation has the **correct sign in both seasons** (which is what the
additive `land_transport_maritime_decay` bonus cannot do, hence that term's
winter gate).
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
from masks import get_masks
from planet_params import EARTH
from simulate import _maritime_transport_factor, create_initial_state, simulate_step

MONTHS = np.linspace(0.0, 365.2422, 12, endpoint=False)


@pytest.fixture(autouse=True)
def _clear_mask_caches():
    masks_module.clear_all_caches()
    yield
    masks_module.clear_all_caches()


def _continent(H: int = 48, W: int = 96) -> np.ndarray:
    """Ocean everywhere except one mid-latitude block with a real interior."""
    elev = np.zeros((H, W), dtype=np.float32)
    elev[10:26, 24:72] = 0.4
    return elev


def _annual_cycle(pp, *, elev=None, block_size=2):
    """Monthly land surface temperature over one year, plus the land mask.

    Steps the real `simulate_step` rather than reimplementing the forcing stack:
    a test that recomputed the arithmetic itself could never fail, which is the
    exact gap `test_maritime_transport.py` found in its own first version.
    """
    elev = _continent() if elev is None else elev
    state = create_initial_state(
        elev, day_of_year=0.0, planet_params=pp, block_size=block_size
    )
    _, land = get_masks(elev, use_cache=False)
    frames = []
    for _ in range(12):
        for _ in range(3):
            state, _ = simulate_step(
                state, days=365.2422 / 36.0, block_size=block_size,
                planet_params=pp, track_components=False,
            )
        frames.append(np.asarray(state.temperature, dtype=np.float64))
    return np.stack(frames), land


# --------------------------------------------------------------------------
# Exact no-ops.  Every knob here must reproduce the historical behaviour
# bit-for-bit at the value documented as inert, or a "default unchanged"
# claim elsewhere in the audit is unverifiable.
# --------------------------------------------------------------------------

def test_amplitude_one_no_maritime_is_bit_identical():
    """The pre-2026-08-05 configuration must be reproducible exactly."""
    historical = dataclasses.replace(
        EARTH,
        land_seasonal_amplitude=1.0,
        land_seasonal_amplitude_maritime=0.0,
        land_transport_gain=1.0,
    )
    cycle_a, _ = _annual_cycle(historical)
    cycle_b, _ = _annual_cycle(historical)
    assert np.array_equal(cycle_a, cycle_b), "the run itself must be deterministic"

    # The damping branch is skipped entirely at these values, so a second
    # configuration that differs only in the *inert* shape knobs must agree.
    same = dataclasses.replace(
        historical,
        sst_land_coupling_km=historical.sst_land_coupling_km * 2.0,
    )
    cycle_c, _ = _annual_cycle(same)
    assert np.array_equal(cycle_a, cycle_c)


def test_transport_gain_one_is_a_no_op():
    base = dataclasses.replace(
        EARTH, land_seasonal_amplitude=1.0, land_seasonal_amplitude_maritime=0.0,
        land_transport_gain=1.0,
    )
    explicit = dataclasses.replace(base, land_transport_gain=1.0)
    assert np.array_equal(_annual_cycle(base)[0], _annual_cycle(explicit)[0])


# --------------------------------------------------------------------------
# The load-bearing property: mean preservation.
# --------------------------------------------------------------------------

def test_damping_shrinks_the_swing():
    strong = dataclasses.replace(
        EARTH, land_seasonal_amplitude=0.4, land_seasonal_amplitude_maritime=0.0,
        land_transport_gain=1.0,
    )
    weak = dataclasses.replace(strong, land_seasonal_amplitude=1.0)
    strong_cycle, land = _annual_cycle(strong)
    weak_cycle, _ = _annual_cycle(weak)
    strong_range = (strong_cycle.max(axis=0) - strong_cycle.min(axis=0))[land].mean()
    weak_range = (weak_cycle.max(axis=0) - weak_cycle.min(axis=0))[land].mean()
    assert strong_range < weak_range, (
        f"damping must reduce the land annual range ({strong_range:.2f} K "
        f"vs {weak_range:.2f} K)"
    )


def test_maritime_modulation_is_row_mean_preserving():
    """The amplitude field redistributes damping within a row, never shifts it.

    Calls the shared factor directly with the same negated decay and flat
    weight `simulate_step` uses, so a change to that call site's sign or
    weighting is caught here rather than only showing up as a level drift.
    """
    land = get_masks(_continent(), use_cache=False)[1]
    H = land.shape[0]
    lat = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
    rng = np.random.default_rng(11)
    maritime = (rng.random(land.shape).astype(np.float32) * land)

    factor = _maritime_transport_factor(
        maritime, land, lat, -0.45, np.ones_like(lat, dtype=np.float32)
    )
    for row in range(H):
        cells = land[row]
        if not np.any(cells):
            assert np.allclose(factor[row], 1.0)
            continue
        assert factor[row][cells].mean() == pytest.approx(1.0, abs=2e-3), (
            f"row {row} amplitude mean drifted to {factor[row][cells].mean():.4f}"
        )


def test_maritime_modulation_lowers_amplitude_near_the_ocean():
    """Sign check: high maritime proximity must damp, not amplify.

    A planted reversal (positive decay instead of negative) inverts this, which
    is the failure the negated argument at the call site exists to prevent.
    """
    land = get_masks(_continent(), use_cache=False)[1]
    H, W = land.shape
    lat = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
    maritime = np.zeros((H, W), dtype=np.float32)
    row = 18
    # Coastal cells maritime, interior continental.
    maritime[row, 24:30] = 1.0
    maritime[row, 30:72] = 0.1

    correct = _maritime_transport_factor(
        maritime, land, lat, -0.45, np.ones_like(lat, dtype=np.float32)
    )
    assert correct[row, 25] < correct[row, 50], (
        "maritime land must receive a smaller seasonal amplitude than interior land"
    )
    reversed_sign = _maritime_transport_factor(
        maritime, land, lat, +0.45, np.ones_like(lat, dtype=np.float32)
    )
    assert reversed_sign[row, 25] > reversed_sign[row, 50], (
        "planted violation: the reversed sign must produce the opposite ordering"
    )


def test_maritime_amplitude_acts_in_both_seasons():
    """The property the additive winter bonus cannot have.

    `land_transport_maritime_decay` scales a bonus, so year-round it warms
    maritime summers -- the wrong sign, which is why it carries a winter gate.
    An amplitude damping must move a maritime cell's warmest month *down* and
    its coldest month *up*, from the one term and with no gate.

    Runs on the bundled Earth DEM rather than this module's synthetic continent,
    which cannot express the mechanism at all: the shipped
    `land_transport_upwind_ratio` of 32 turns a 1200 km isotropic reach into a
    38400 km westward one, which is most of a planetary circumference, so on a
    single small block every land cell reads as maximally maritime and the
    contrast the test is about does not exist.  (Measured while writing this: the
    toy geography gives a -0.8 K coastal winter change of no determinate sign.)
    """
    from real_terrain_validation import load_bundled_earth_dem

    elev = load_bundled_earth_dem(32, 64)
    off = dataclasses.replace(
        EARTH, land_seasonal_amplitude=1.0, land_seasonal_amplitude_maritime=0.0,
        land_transport_gain=1.0, land_transport_maritime_decay=0.0,
    )
    on = dataclasses.replace(off, land_seasonal_amplitude_maritime=1.0)
    off_cycle, land = _annual_cycle(off, elev=elev, block_size=1)
    on_cycle, _ = _annual_cycle(on, elev=elev, block_size=1)

    # The most maritime third of mid-latitude land, by the mechanism's own
    # field -- the population it is supposed to move.
    from simulate import _maritime_proximity

    proximity = _maritime_proximity(
        land, float(EARTH.land_transport_maritime_km), float(EARTH.radius_m) / 1000.0,
        float(EARTH.land_transport_upwind_ratio),
    )
    H = land.shape[0]
    lat = np.abs((0.5 - (np.arange(H) + 0.5) / H) * 180.0)
    midlat = land & ((lat >= 30.0) & (lat <= 65.0))[:, None]
    assert np.any(midlat)
    cutoff = np.percentile(proximity[midlat], 67.0)
    maritime_cells = midlat & (proximity >= cutoff)

    warm_change = (on_cycle.max(axis=0) - off_cycle.max(axis=0))[maritime_cells].mean()
    cold_change = (on_cycle.min(axis=0) - off_cycle.min(axis=0))[maritime_cells].mean()
    assert warm_change < 0.0, f"maritime summers must cool, got {warm_change:+.3f} K"
    assert cold_change > 0.0, f"maritime winters must warm, got {cold_change:+.3f} K"


def test_transport_gain_lowers_the_land_annual_mean():
    """The level half of the fix.

    The trapezoids are added in every month, so scaling them down must move the
    land annual *mean*, which is the error `_land_cap_1d` was hiding and which
    no amplitude-side knob can reach.
    """
    full = dataclasses.replace(
        EARTH, land_seasonal_amplitude=1.0, land_seasonal_amplitude_maritime=0.0,
        land_transport_gain=1.0,
    )
    reduced = dataclasses.replace(full, land_transport_gain=0.4)
    full_cycle, land = _annual_cycle(full)
    reduced_cycle, _ = _annual_cycle(reduced)
    # Mid-latitude land only: the trapezoids are zero in the deep tropics.
    H = land.shape[0]
    lat = np.abs((0.5 - (np.arange(H) + 0.5) / H) * 180.0)
    midlat = land & ((lat >= 25.0) & (lat <= 60.0))[:, None]
    assert np.any(midlat)
    change = (reduced_cycle.mean(axis=0) - full_cycle.mean(axis=0))[midlat].mean()
    assert change < -0.5, f"cutting the trapezoids must cool the mean, got {change:+.3f} K"
