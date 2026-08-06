"""test_maritime_transport.py -- the continentality gradient on land heat transport.

Covers `simulate._maritime_proximity` (anisotropic upwind reach),
`simulate._maritime_proximity_coarse` (native-resolution field, land-only block
mean), and the row-mean-preserving winter-weighted application in
`simulate_step`.  See audit C1b and `PlanetParams.land_transport_maritime_decay`.

The mechanism exists because all three land heat-transport trapezoids are pure
functions of |latitude|, so without it every land cell in a row gets an identical
winter bonus and the model has no maritime moderation gradient at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from masks import get_masks
from planet_params import EARTH
from simulate import (
    _maritime_proximity, _maritime_proximity_coarse, _maritime_transport_factor,
    clear_simulation_caches,
)

R_KM = 6371.0


@pytest.fixture(autouse=True)
def _clear_mask_caches():
    """`get_masks` is keyed on id(array) plus a 512-sample strided fingerprint.

    Several tests here build same-shaped elevation arrays that differ in a single
    cell, which that fingerprint can miss entirely once Python reuses an id --
    exactly the collision its own docstring warns about.  Clearing between tests
    keeps these deterministic in any order.

    Uses `simulate.clear_simulation_caches` rather than `masks.clear_all_caches`
    (which it calls anyway): `_maritime_proximity_coarse` consults its own
    `_MARITIME_COARSE_CACHE`, with the same id+strided-fingerprint weakness, and
    clearing only the mask caches left that one live. The fixture's "deterministic
    in any order" promise was therefore not being kept -- adding an unrelated test
    module to the same pytest run shifted allocation enough to make
    `test_coarse_block_mean_is_over_land_cells_only` take a stale hit and read 0.0
    for its lone land cell (2026-08-05). `clear_simulation_caches` was itself
    missing that cache; fixed there too, so this is belt-and-braces.
    """
    clear_simulation_caches()
    yield
    clear_simulation_caches()


def _continent(H: int = 64, W: int = 128) -> np.ndarray:
    """Ocean everywhere except one wide land block with a real interior."""
    elev = np.zeros((H, W), dtype=np.float32)
    elev[16:48, 32:96] = 0.5
    return elev


def _land(elev):
    return get_masks(elev, use_cache=False)[1]


# --------------------------------------------------------------------------
# _maritime_proximity: the field itself
# --------------------------------------------------------------------------

def test_zero_on_ocean_and_bounded_on_land():
    land = _land(_continent())
    f = _maritime_proximity(land, 1200.0, R_KM)
    assert np.all(f[~land] == 0.0)
    assert np.all((f[land] >= 0.0) & (f[land] <= 1.0))


def test_decays_inland_from_the_coast():
    land = _land(_continent())
    f = _maritime_proximity(land, 1200.0, R_KM)
    row = 32  # through the middle of the continent
    assert f[row, 33] > f[row, 40] > f[row, 63], "proximity must fall inland"


def test_upwind_ratio_one_is_exactly_isotropic():
    """The default anisotropy must reproduce the original field bit-for-bit."""
    land = _land(_continent())
    a = _maritime_proximity(land, 1200.0, R_KM, 1.0)
    b = _maritime_proximity(land, 1200.0, R_KM)
    assert np.array_equal(a, b)


def test_isotropic_field_is_east_west_symmetric():
    """Guards the direction convention: with no anisotropy a symmetric continent
    must produce a symmetric field, so any asymmetry below is the ratio's doing."""
    land = _land(_continent())
    f = _maritime_proximity(land, 1200.0, R_KM, 1.0)
    row = 32
    assert f[row, 33] == pytest.approx(f[row, 94], rel=1e-5)


def test_upwind_ratio_reaches_further_east_of_a_western_ocean():
    """`upwind_ratio` must lengthen the reach of ocean lying to a cell's WEST.

    This is the load-bearing direction convention -- reversing it would build a
    field that reports New York as maritime and Lisbon as continental, i.e. the
    exact inversion the mechanism exists to avoid.  On this symmetric continent
    the western ocean is at column 31 and the eastern one at column 96, so a
    cell just inside the west coast is upwind-exposed and its mirror image just
    inside the east coast is not.
    """
    land = _land(_continent())
    iso = _maritime_proximity(land, 1200.0, R_KM, 1.0)
    aniso = _maritime_proximity(land, 1200.0, R_KM, 8.0)
    row, west_inland, east_inland = 32, 45, 82
    # Anisotropy must raise the cell downwind of the western ocean...
    assert aniso[row, west_inland] > iso[row, west_inland]
    # ...by more than it raises its mirror image near the eastern coast.
    assert (aniso[row, west_inland] - iso[row, west_inland]) > (
        aniso[row, east_inland] - iso[row, east_inland]
    )
    # And the field must end up asymmetric in the correct direction.
    assert aniso[row, west_inland] > aniso[row, east_inland]


def test_upwind_ratio_does_not_change_ocean_or_bounds():
    land = _land(_continent())
    f = _maritime_proximity(land, 1200.0, R_KM, 16.0)
    assert np.all(f[~land] == 0.0)
    assert np.all(f[land] <= 1.0 + 1e-6)


# --------------------------------------------------------------------------
# _maritime_proximity_coarse: native resolution + land-only block mean
# --------------------------------------------------------------------------

def test_coarse_field_shape_and_ocean_cells():
    elev = _continent(64, 128)
    f = _maritime_proximity_coarse(elev, 16, 32, 4, 1200.0, R_KM, 1.0)
    assert f.shape == (16, 32)
    # A block with no land at all averages nothing and must read exactly zero.
    assert f[0, 0] == 0.0


def test_coarse_block_mean_is_over_land_cells_only():
    """A coastal block that is mostly ocean must report its *land's* proximity.

    A plain block mean would divide a coastal sliver's high proximity by the
    whole block and report the most maritime cells on the grid as the most
    continental ones.  Constructed so one coarse block holds a single land cell
    right on the coast: its coarse value must match that cell's fine value, not
    a sixteenth of it.
    """
    H, W, block = 64, 128, 4
    elev = np.zeros((H, W), dtype=np.float32)
    elev[16:48, 32:96] = 0.5
    elev[12, 36] = 0.5  # lone land cell, alone in its 4x4 block
    land = _land(elev)
    fine = _maritime_proximity(land, 1200.0, R_KM, 1.0)
    coarse = _maritime_proximity_coarse(elev, H // block, W // block, block,
                                        1200.0, R_KM, 1.0)
    assert land[12, 36] and land.reshape(16, 4, 32, 4)[3, :, 9, :].sum() == 1
    assert coarse[3, 9] == pytest.approx(fine[12, 36], rel=1e-5)


def test_coarse_field_beats_the_coarse_mask_on_coastal_contrast():
    """The point of computing at native resolution.

    `_coarsen` block-means elevation and `get_masks` then calls any block holding
    a sliver of land "land", which pushes coastlines outward by up to a full
    coarse cell.  On the same terrain the native-resolution field must retain
    more coast-to-interior contrast than one built on that mask.
    """
    from sim_grid import _coarsen_elevation_cached

    H, W, block = 64, 128, 4
    elev = _continent(H, W)
    Hc, Wc = H // block, W // block
    native = _maritime_proximity_coarse(elev, Hc, Wc, block, 1200.0, R_KM, 1.0)
    coarse_mask = _land(_coarsen_elevation_cached(elev, Hc, Wc, block))
    from_mask = _maritime_proximity(coarse_mask, 1200.0, R_KM, 1.0)
    land_c = native > 0.0
    assert native[land_c].std() > from_mask[land_c].std()


def test_coarse_falls_back_when_block_size_does_not_divide():
    """`Hc` is a *ceiling* division, so a grid whose height is not a multiple of
    the block size gives Hc*block > H and the exact reshape is unavailable.  The
    fallback must still return a usable coarse field rather than raising."""
    elev = _continent(62, 126)
    f = _maritime_proximity_coarse(elev, 16, 32, 4, 1200.0, R_KM, 1.0)
    assert f.shape == (16, 32)
    assert np.all(np.isfinite(f))


# --------------------------------------------------------------------------
# Application in simulate_step
# --------------------------------------------------------------------------

def _run(**overrides):
    import dataclasses

    from real_terrain_validation import (
        RealTerrainValidationConfig, run_real_terrain_validation,
    )

    config = RealTerrainValidationConfig(height=64, width=128,
                                         spinup_years=0.0, evaluation_years=1.0)
    pp = dataclasses.replace(EARTH, **overrides) if overrides else EARTH
    state, report = run_real_terrain_validation(config, planet_params=pp)
    return state, report


def test_shape_knob_is_inert_when_nothing_consumes_the_field():
    """`upwind_ratio` only shapes the maritime field; with no consumer it is inert.

    Was `test_decay_zero_is_an_exact_no_op`, and the rename is the point: since
    2026-08-05 the maritime-proximity field has **two** consumers, this winter
    transport bonus and `land_seasonal_amplitude_maritime` (audit C1b-2026-08-05).
    Zeroing this knob alone therefore no longer switches the field off, and the
    old assertion was correct for one-consumer code only. Both have to be off for
    a shape knob to be inert.
    """
    off = dict(land_transport_maritime_decay=0.0, land_seasonal_amplitude_maritime=0.0)
    a, _ = _run(**off, land_transport_upwind_ratio=1.0)
    b, _ = _run(**off, land_transport_upwind_ratio=4.0)
    assert np.array_equal(
        np.asarray(a.monthly_temp), np.asarray(b.monthly_temp)
    ), "upwind_ratio must be inert while no mechanism consumes the maritime field"


def test_decay_zero_still_disables_this_mechanism():
    """This knob must remain its own mechanism's off switch.

    Held apart from the shape-knob test above so a failure says which property
    broke: that one is about the shared *field*, this one about *this* consumer.
    """
    base = dict(land_seasonal_amplitude_maritime=0.0)
    a, _ = _run(**base, land_transport_maritime_decay=0.0)
    b, _ = _run(**base, land_transport_maritime_decay=1.0)
    assert not np.array_equal(
        np.asarray(a.monthly_temp), np.asarray(b.monthly_temp)
    ), "the decay knob must still change the state when turned on"


def _earth_factor_inputs(H=128, W=256, block=4, upwind=32.0):
    from sim_grid import _coarsen_elevation_cached
    from real_terrain_validation import load_bundled_earth_dem

    Hc, Wc = H // block, W // block
    elev = load_bundled_earth_dem(H, W)
    land_c = _land(_coarsen_elevation_cached(elev, Hc, Wc, block))
    maritime = _maritime_proximity_coarse(elev, Hc, Wc, block, 1200.0, R_KM, upwind)
    lat = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    return maritime, land_c, lat


def test_mechanism_is_row_mean_preserving_in_the_forcing():
    """The calibrated zonal-mean winter level must not move.

    Checked on the factor `simulate` actually applies, not on output
    temperature: land temperature is a nonlinear function of the forcing, so the
    *output* mean legitimately drifts (~0.1 K at 40-60N, documented), but the
    redistribution applied to the forcing has to be mean-zero over each row's
    land by construction.
    """
    maritime, land_c, lat = _earth_factor_inputs()
    winter = np.ones(land_c.shape[0], dtype=np.float32)
    factor = _maritime_transport_factor(maritime, land_c, lat, 1.0, winter)

    lf = land_c.astype(np.float64)
    per_row = lf.sum(axis=1)
    rows = per_row > 0
    # Mean of (factor - 1) over each row's land must vanish.
    residual = ((factor - 1.0) * lf).sum(axis=1)[rows] / per_row[rows]
    assert np.abs(residual).max() < 1e-3, (
        f"row-mean drift {np.abs(residual).max():.4g} -- the mechanism is "
        "changing the calibrated zonal level, not just the contrast"
    )
    # And it must actually be doing something, or the check above is vacuous.
    assert factor[land_c].std() > 0.1


def test_factor_is_one_off_land_and_in_summer():
    maritime, land_c, lat = _earth_factor_inputs()
    winter = np.ones(land_c.shape[0], dtype=np.float32)
    assert np.all(_maritime_transport_factor(
        maritime, land_c, lat, 1.0, winter)[~land_c] == 1.0)
    summer = np.zeros(land_c.shape[0], dtype=np.float32)
    assert np.all(_maritime_transport_factor(
        maritime, land_c, lat, 1.0, summer) == 1.0)


def test_factor_warms_maritime_and_cools_continental_cells():
    """Direction of effect, on the real DEM: the mechanism must give *more*
    bonus to the maritime end of a row and less to the continental end."""
    maritime, land_c, lat = _earth_factor_inputs()
    winter = np.ones(land_c.shape[0], dtype=np.float32)
    factor = _maritime_transport_factor(maritime, land_c, lat, 1.0, winter)
    vals = maritime[land_c]
    hi = vals >= np.percentile(vals, 75)
    lo = vals <= np.percentile(vals, 25)
    assert factor[land_c][hi].mean() > 1.0 > factor[land_c][lo].mean()


def test_factor_strength_is_invariant_to_the_shape_knobs():
    """`decay` must mean the same thing at any upwind reach.

    Without the spread normalization the strength and shape knobs are degenerate
    -- flattening the field with a longer reach silently weakens the mechanism,
    which is how an earlier sweep needed decay 12 at one reach and 2 at another
    for the same physical effect.
    """
    spreads = []
    for upwind in (4.0, 32.0):
        maritime, land_c, lat = _earth_factor_inputs(upwind=upwind)
        winter = np.ones(land_c.shape[0], dtype=np.float32)
        factor = _maritime_transport_factor(maritime, land_c, lat, 1.0, winter)
        spreads.append(float(factor[land_c].std()))
    assert spreads[0] == pytest.approx(spreads[1], rel=0.25), (
        f"factor spread {spreads} varies with the shape knob -- `decay` is not "
        "scale-free"
    )


def test_upwind_ratio_does_not_touch_the_meridional_direction():
    """Anisotropy must be purely zonal.

    Guards against the ratio leaking into `decay_lat`, which would silently turn
    it into a global reach multiplier -- it would still look like an improvement
    on aggregate scores while no longer encoding "upwind" at all.  Terrain here
    is a land band spanning every longitude, so the only ocean is north and
    south of it and a purely zonal anisotropy cannot change the field.
    """
    elev = np.zeros((64, 128), dtype=np.float32)
    elev[24:40, :] = 0.5  # full-circumference band: no ocean east or west
    land = _land(elev)
    a = _maritime_proximity(land, 1200.0, R_KM, 1.0)
    b = _maritime_proximity(land, 1200.0, R_KM, 32.0)
    assert np.array_equal(a, b)


def test_improves_coldest_month_threshold_accuracy():
    """Regression guard on the result this mechanism shipped for.

    Both reference groups must improve together -- that is what distinguishes an
    added continentality *contrast* from a shifted band mean, which is what five
    earlier latitude-only knobs did instead.

    **Measured against a baseline with `land_seasonal_amplitude_maritime` off
    too** (added 2026-08-05, audit C1b-2026-08-05). That knob expresses
    continentality through the seasonal *amplitude* rather than through this
    winter bonus, so the two overlap and this one's *incremental* gain on top of
    it is small -- under this test's original 0.01 bar, which is why it started
    failing. Isolating against "both off" keeps this a guard on *this*
    mechanism's own contribution rather than on a number that moves whenever the
    other one is retuned.

    **REBASELINED 2026-08-05 (second time, A2-REOPEN), bar 0.01 -> 0.005.**
    `climate_averages.update_monthly_statistics` stopped blending its flat
    spin-up seed into the Köppen monthly bins. `_run` above uses
    `spinup_years=0.0`, so each bin got exactly one real sample and the old code
    left **36.8%** of a zero-amplitude annual cycle in it -- i.e. this test's
    entire basis was a heavily damped seasonal cycle, which is precisely the
    error both continentality mechanisms exist to correct. Coldest-month
    accuracy on the tracked 64x128 fixture, before and after that fix:

        case                    before    after     C (after)   D (after)
        both off                0.8414    0.8772     0.7579      0.8691
        amplitude only          0.8710    0.8918     0.8164      0.9116
        this mechanism only     0.8590    0.8868     0.7870      0.8843
        both on (shipped)       0.8785    0.8959     0.8270      0.9205

    The seed fix raises the "both off" floor by **+0.036**, which is *larger
    than either mechanism's own contribution* -- so the artifact was a bigger
    source of coldest-month error than the physics these knobs were built to add,
    and they were partly compensating for it. Their measured value shrinks
    accordingly: this one's aggregate delta goes +0.0176 -> **+0.0096**.

    **What did not change is the property this test actually guards**: both
    reference groups still improve together (C **+0.0291**, D **+0.0152**, with
    the warmest month exactly neutral at +0.0000), which is the signature of an
    added continentality *contrast* rather than a shifted band mean. Those
    per-group assertions below are untouched and pass with room to spare; only
    the aggregate bar moved, and it moved because its baseline did. 0.005 keeps a
    real guard (the mechanism would have to lose ~half its remaining effect to
    trip it) without re-encoding a floor that no longer exists.
    """
    off, _ = _run(land_transport_maritime_decay=0.0,
                  land_seasonal_amplitude_maritime=0.0)
    on, _ = _run(land_seasonal_amplitude_maritime=0.0)

    from koppen_reference import score_temperature_thresholds

    s_off = score_temperature_thresholds(np.asarray(off.monthly_temp))
    s_on = score_temperature_thresholds(np.asarray(on.monthly_temp))
    cold_off = s_off["coldest_month"]
    cold_on = s_on["coldest_month"]
    assert cold_on["accuracy"] > cold_off["accuracy"] + 0.005
    for group in ("C", "D"):
        assert (
            cold_on["by_reference_group"][group]["accuracy"]
            > cold_off["by_reference_group"][group]["accuracy"]
        ), f"reference-{group} must improve too, not trade against the other"


def test_winter_weighting_leaves_the_warmest_month_alone():
    """Maritime moderation is a winter effect.

    Applied year-round the same contrast warms maritime *summers*, which is the
    wrong sign and cost 0.26-0.45pp of warmest-month accuracy when measured.
    The winter weight must keep that score from regressing.

    Isolated the same way as the test above: only this knob varies, with the
    amplitude-side continentality mechanism held off in both runs. (That one
    deliberately *does* act in summer -- being able to, without a gate, is the
    property it was built for. See `land_seasonal_amplitude_maritime`.)
    """
    off, _ = _run(land_transport_maritime_decay=0.0,
                  land_seasonal_amplitude_maritime=0.0)
    on, _ = _run(land_seasonal_amplitude_maritime=0.0)

    from koppen_reference import score_temperature_thresholds

    warm_off = score_temperature_thresholds(
        np.asarray(off.monthly_temp))["warmest_month"]["accuracy"]
    warm_on = score_temperature_thresholds(
        np.asarray(on.monthly_temp))["warmest_month"]["accuracy"]
    assert warm_on >= warm_off - 0.002
