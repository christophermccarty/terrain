"""test_evap_cooling.py -- the evapotranspiration cooling term's gates and guards.

Covers the three knobs audit C1b-EVAP (2026-08-05) added to
`simulate._evolve_temperature`'s forcing stack: `evap_cooling_threshold_k` and
`evap_cooling_coeff` (which gate two constants that had been hardcoded since the
term was written) and `evap_cooling_amplitude` (the shape-only sibling of
`evap_cooling_strength`).  See `docs/ACCURACY_AUDIT.md` C1b-EVAP.

Two of the properties pinned here are correctness rather than calibration, and
both were live defects before that session:

- **The contraction cannot invert.**  The term removes
  `season * coeff * strength * soil` of a cell's above-threshold excess.  That
  product exceeds 1 once `coeff * strength > 1 / soil` -- around
  `evap_cooling_strength` 1.18 at the shipped coefficient -- past which the term
  removes *more* than the whole excess and the mapping from pre- to post-cooling
  temperature turns around: a hotter cell comes out colder than a cooler one.
  The knob's documented sweep range crossed that line silently.
- **The threshold is the term's reach, not a neutral tuning constant.**  At its
  historical 290 K the term is identically zero on every cell whose forcing stays
  below 16.85 C, which is all of the sub-polar and Southern-Hemisphere
  mid-latitude land carrying the model's largest warmest-month error.
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
from simulate import (
    _maritime_transport_factor,
    create_initial_state,
    _evap_cooling_fraction,
    simulate_step,
)


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

    Steps the real `simulate_step` rather than reimplementing the forcing stack.
    A test that recomputed the arithmetic itself could not fail when the shipped
    code changed -- audit process note 19(b), and the exact gap
    `test_maritime_transport.py` found in its own first version.
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
# Exact no-ops.  Both newly-gated constants must reproduce the historical
# hardcoded behaviour bit-for-bit, or "shipped inert" is unverifiable.
# --------------------------------------------------------------------------

def test_gated_constants_at_their_historical_values_are_bit_identical():
    explicit = dataclasses.replace(
        EARTH, evap_cooling_threshold_k=290.0, evap_cooling_coeff=0.85
    )
    assert np.array_equal(_annual_cycle(EARTH)[0], _annual_cycle(explicit)[0])


def test_amplitude_zero_is_a_no_op():
    explicit = dataclasses.replace(EARTH, evap_cooling_amplitude=0.0)
    assert np.array_equal(_annual_cycle(EARTH)[0], _annual_cycle(explicit)[0])


def test_an_unreachable_threshold_disables_the_term_exactly():
    """The threshold is the reach: put it above every land temperature and the
    term must vanish, matching a zeroed strength bit-for-bit."""
    unreachable = dataclasses.replace(EARTH, evap_cooling_threshold_k=1.0e4)
    disabled = dataclasses.replace(EARTH, evap_cooling_strength=0.0)
    assert np.array_equal(_annual_cycle(unreachable)[0], _annual_cycle(disabled)[0])


# --------------------------------------------------------------------------
# The correctness guard: the contraction must never invert.
# --------------------------------------------------------------------------

# These call `simulate._evap_cooling_fraction`, the function the shipped forcing
# paths use, rather than asserting on a simulated temperature field.  That is
# deliberate and was measured: the land cap sits immediately downstream of this
# term and absorbs ~99% of it (audit C1b-EVAP), so an inverted contraction shows
# up at the output as ~2 K on 16 of 591 cells -- comfortably inside any tolerance
# an end-to-end test would have to allow.  An output-level version of these two
# tests was written first and **passed with the guard removed**, which is
# process note 19(b)'s failure exactly.

def test_contraction_fraction_never_exceeds_one():
    """The invariant the clip exists for, on the population that violates it.

    At these settings the unclipped product reaches 2.55, i.e. the term would
    remove 255% of a cell's excess.
    """
    season = np.ones((4, 4), dtype=np.float32)
    soil = np.full((4, 4), 1.0, dtype=np.float32)
    fraction = _evap_cooling_fraction(season, soil, coeff=0.85, strength=3.0)
    assert fraction.max() <= 1.0, (
        f"contraction fraction reached {fraction.max():.3f}: the term removes "
        "more than the whole excess and inverts"
    )


def test_contraction_is_monotone_in_temperature_at_any_strength():
    """A hotter cell must never come out colder than a cooler one.

    Applies the shipped fraction to a ramp of excesses the way both forcing
    paths do (`T - fraction * max(T - threshold, 0)`) and checks the result is
    non-decreasing.  Without the clip this ramp runs backwards above the
    threshold.
    """
    threshold = 290.0
    temperature = np.linspace(threshold, threshold + 25.0, 40, dtype=np.float32)
    season = np.ones_like(temperature)
    soil = np.full_like(temperature, 0.8)
    for strength in (0.5, 1.0, 1.18, 2.0, 3.0, 10.0):
        fraction = _evap_cooling_fraction(season, soil, coeff=0.85, strength=strength)
        cooled = temperature - fraction * np.maximum(temperature - threshold, 0.0)
        assert np.all(np.diff(cooled) >= -1e-6), (
            f"at evap_cooling_strength={strength} the contraction is "
            "non-monotonic: a hotter cell comes out colder"
        )
        assert cooled.min() >= threshold - 1e-4, (
            f"at evap_cooling_strength={strength} the contraction overshot its "
            f"own {threshold} K reference, reaching {cooled.min():.2f} K"
        )


def test_contraction_fraction_is_inert_at_the_shipped_defaults():
    """The clip must not be silently active at the values the model ships with,
    or it would be a physics change wearing a guard's clothes."""
    season = np.ones((8, 8), dtype=np.float32)
    soil = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    guarded = _evap_cooling_fraction(
        season, soil, coeff=EARTH.evap_cooling_coeff,
        strength=EARTH.evap_cooling_strength,
    )
    unguarded = (
        season * (EARTH.evap_cooling_coeff * EARTH.evap_cooling_strength) * soil
    )
    assert np.allclose(guarded, unguarded), (
        "the [0, 1] clip changes the result at the shipped defaults"
    )


def test_coefficient_and_strength_are_interchangeable_below_the_clip():
    """Below saturation the two knobs enter as a product, so halving one and
    doubling the other must be a no-op.  Above it they would not be, which is
    what makes the clip observable rather than cosmetic."""
    a = dataclasses.replace(EARTH, evap_cooling_coeff=0.40, evap_cooling_strength=1.0)
    b = dataclasses.replace(EARTH, evap_cooling_coeff=0.20, evap_cooling_strength=2.0)
    assert np.allclose(_annual_cycle(a)[0], _annual_cycle(b)[0], atol=1e-6)


# --------------------------------------------------------------------------
# The amplitude form: shape only, and signed by soil moisture.
# --------------------------------------------------------------------------

def test_amplitude_damping_is_row_mean_preserving():
    """It redistributes damping within a row rather than shifting the row.

    Calls the shared factor with the same negated strength and flat weight
    `simulate_step` uses, so a sign or weighting change at that call site is
    caught here rather than surfacing later as a zonal level drift.
    """
    land = get_masks(_continent(), use_cache=False)[1]
    H = land.shape[0]
    lat = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
    rng = np.random.default_rng(5)
    soil = (rng.random(land.shape).astype(np.float32) * land)

    factor = _maritime_transport_factor(
        soil, land, lat, -0.45, np.ones_like(lat, dtype=np.float32)
    )
    for row in range(H):
        cells = land[row]
        if not np.any(cells):
            assert np.allclose(factor[row], 1.0)
            continue
        assert factor[row][cells].mean() == pytest.approx(1.0, abs=2e-3), (
            f"row {row} amplitude mean drifted to {factor[row][cells].mean():.4f}"
        )


def test_wet_land_is_damped_more_than_dry_land():
    """Sign check: latent heat flux buffers a moist surface, not a dry one.

    A planted reversal (positive strength instead of negative) inverts the
    ordering, which is the failure the negated argument at the call site exists
    to prevent.
    """
    land = get_masks(_continent(), use_cache=False)[1]
    H, W = land.shape
    lat = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
    soil = np.zeros((H, W), dtype=np.float32)
    row = 18
    soil[row, 24:30] = 0.9   # wet
    soil[row, 30:72] = 0.1   # dry

    correct = _maritime_transport_factor(
        soil, land, lat, -0.45, np.ones_like(lat, dtype=np.float32)
    )
    assert correct[row, 25] < correct[row, 50], (
        "wet land must receive a smaller seasonal amplitude than dry land"
    )
    reversed_sign = _maritime_transport_factor(
        soil, land, lat, +0.45, np.ones_like(lat, dtype=np.float32)
    )
    assert reversed_sign[row, 25] > reversed_sign[row, 50], (
        "planted violation: the reversed sign must produce the opposite ordering"
    )


def test_amplitude_damping_leaves_the_land_annual_mean_alone():
    """The property that distinguishes it from `evap_cooling_strength`.

    The threshold form subtracts from a cell's annual mean, which is why
    strengthening it cools wet tropical land under Koppen's 18 C A-boundary and
    costs group accuracy there (audit C1b-EVAP: -2.05pp in 0:10 at 128x256).  An
    amplitude damping is mean-preserving in time by construction, so the same
    soil-moisture field can be applied without that cost.
    """
    off = dataclasses.replace(EARTH, evap_cooling_amplitude=0.0)
    on = dataclasses.replace(EARTH, evap_cooling_amplitude=0.6)
    off_cycle, land = _annual_cycle(off)
    on_cycle, _ = _annual_cycle(on)

    mean_shift = (on_cycle.mean(axis=0) - off_cycle.mean(axis=0))[land].mean()
    range_off = (off_cycle.max(axis=0) - off_cycle.min(axis=0))[land]
    assert abs(mean_shift) < 0.25 * range_off.mean(), (
        f"amplitude damping moved the land annual mean by {mean_shift:+.3f} K, "
        "which a mean-preserving contraction must not do"
    )


def test_amplitude_damping_changes_the_swing_somewhere():
    """Mean preservation must not be achieved by the term doing nothing at all."""
    off = dataclasses.replace(EARTH, evap_cooling_amplitude=0.0)
    on = dataclasses.replace(EARTH, evap_cooling_amplitude=0.6)
    off_cycle, land = _annual_cycle(off)
    on_cycle, _ = _annual_cycle(on)
    range_off = (off_cycle.max(axis=0) - off_cycle.min(axis=0))[land]
    range_on = (on_cycle.max(axis=0) - on_cycle.min(axis=0))[land]
    assert np.abs(range_on - range_off).max() > 0.05, (
        "the amplitude knob left every land cell's annual range unchanged"
    )
