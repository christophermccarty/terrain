"""Guards on the two 2026-08-05 fixes that restored the tropical rainforest band.

Both bugs were invisible to every aggregate metric the project tracked (zonal
means, desert boxes, `reference_error_score`, even the area-weighted Köppen group
shares) and together drove deep-tropical driest-month precipitation to <10 mm
against Earth's 60-150 mm, collapsing Af from ~6% of land to 2.6%. See
`PlanetParams.itcz_seasonal_target_min_fraction`, ACCURACY_AUDIT.md A2 and
process note 24.

Per ACCURACY_AUDIT.md process note 19(b), these assert the invariants against the
real code paths rather than re-deriving the arithmetic -- a test that recomputes
`1 + k*(window - mean)` itself would have passed throughout the entire period the
bug was shipped.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from atmosphere import generate_precipitation
from climate_averages import update_monthly_statistics
from planet_params import EARTH
from testing.conftest import make_mixed_elev

H, W = 32, 64


def _modulation_over_orbit(pp, n_samples: int = 36) -> np.ndarray:
    """Sample `itcz_seasonal_target_modulation` across one orbit via the real call."""
    elevation = make_mixed_elev(H, W)
    temperature = np.full((H, W), 295.0, dtype=np.float32)
    wind_u = np.full((H, W), 4.0, dtype=np.float32)
    wind_v = np.zeros((H, W), dtype=np.float32)
    rows = []
    for i in range(n_samples):
        debug: dict = {}
        generate_precipitation(
            H,
            W,
            elevation,
            temperature=temperature,
            wind_u=wind_u,
            wind_v=wind_v,
            day_of_year=pp.orbital_period_days * i / n_samples,
            dt_days=1.0,
            planet_params=pp,
            debug_fields=debug,
        )
        assert "itcz_seasonal_target_modulation" in debug, (
            "the seasonal target modulation is not running -- this test would "
            "vacuously pass"
        )
        rows.append(np.asarray(debug["itcz_seasonal_target_modulation"], dtype=np.float64))
    return np.stack(rows)


def test_seasonal_target_modulation_respects_min_fraction_floor():
    """The dry-season trough may not drop below `min_fraction` of the row target.

    The shipped `k=2.0` makes the raw additive form go NEGATIVE for every row
    with `itcz_window_annual_mean > 1/k` -- ~9 deg of the equator, i.e. the whole
    rainforest band. Those rows were previously held up only by a bare
    `clip(0.05)`, a 95% dry-season shutoff.
    """
    pp = EARTH
    assert pp.itcz_seasonal_target_min_fraction > 0.0
    mod = _modulation_over_orbit(pp)

    assert np.all(np.isfinite(mod))
    assert mod.min() >= pp.itcz_seasonal_target_min_fraction - 1e-6, (
        f"modulation fell to {mod.min():.4f}, below the "
        f"{pp.itcz_seasonal_target_min_fraction} floor"
    )
    # The cap must hold with real margin rather than by landing on the safety
    # clip -- if the clip is doing the work, mean-preservation below is broken.
    assert mod.min() > pp.itcz_seasonal_target_min_fraction + 1e-3


def test_seasonal_target_modulation_is_mean_preserving_over_an_orbit():
    """Time-average must be 1.0 per row, so annual-mean calibration is untouched.

    This is the property `itcz_seasonal_target_response` has always *claimed*;
    the old `clip(0.05)` silently broke it once `k` exceeded ~1.0 (truncating the
    trough while the crest stayed), which is why raising `k` quietly lowered
    tropical annual-mean precipitation as well as deepening its dry season.
    """
    mod = _modulation_over_orbit(EARTH, n_samples=72)
    per_row_mean = mod.mean(axis=0)
    # Tolerance is for the finite orbit sampling, not for clipping: the old form
    # was off by ~5.8e-3 here, three orders of magnitude worse.
    assert np.abs(per_row_mean - 1.0).max() < 1e-4, (
        f"max per-row time-mean deviation {np.abs(per_row_mean - 1.0).max():.2e}"
    )


def test_min_fraction_only_binds_in_the_deep_tropics():
    """The floor must not weaken the savanna belt this knob exists to serve.

    `min_fraction` is a targeted repair of the deep tropics, not a global damping
    of seasonality: rows outside `itcz_window`'s core keep the full `k` and their
    full dry season. If this ever fails, the fix has become a blunt instrument.
    """
    lat = (0.5 - (np.arange(H) + 0.5) / H) * 180.0
    floored = _modulation_over_orbit(EARTH)
    unfloored = _modulation_over_orbit(
        dataclasses.replace(EARTH, itcz_seasonal_target_min_fraction=0.0)
    )
    changed = np.abs(floored - unfloored).max(axis=0) > 1e-6
    assert changed.any(), "the floor changed nothing at all"
    assert np.abs(lat[changed]).max() < 20.0, (
        f"floor reached |lat|={np.abs(lat[changed]).max():.1f} deg; it must stay "
        "inside the deep tropics"
    )
    # And it must genuinely repair those rows, not merely touch them.
    assert unfloored.min() < 0.06 < floored.min()


def test_zero_min_fraction_restores_the_unbounded_form():
    """0.0 is documented as an exact opt-out; keep that true."""
    off = _modulation_over_orbit(
        dataclasses.replace(EARTH, itcz_seasonal_target_min_fraction=0.0), n_samples=36
    )
    # The unbounded form's trough sits on the legacy 0.05 clip in the deep
    # tropics -- i.e. down in the old 95%-shutoff regime, far below the shipped
    # floor. (Not exactly 0.05: a finite orbit sampling need not land on the
    # exact solstice day where the raw form bottoms out.)
    assert 0.05 <= off.min() < 0.055
    assert off.min() < EARTH.itcz_seasonal_target_min_fraction


def _state(H_: int, W_: int, temp: float, precip: float, day: float):
    return SimpleNamespace(
        temperature=np.full((H_, W_), temp, dtype=np.float32),
        precipitation=np.full((H_, W_), precip, dtype=np.float32),
        day_of_year=day,
        monthly_temp=None,
        monthly_precip=None,
        monthly_sample_count=None,
    )


def test_monthly_bin_first_sample_discards_the_spinup_seed():
    """A bin's first real sample must REPLACE the flat seed, not blend with it.

    Blending left `(1-alpha)**n` of a zero-amplitude annual cycle in the bins
    (~13.5% after two simulated years at `window_years=1.0`). Because the seed is
    flat it lands on whichever bin should be the year's extreme, which is exactly
    what Köppen's "driest month >= 60 mm" test reads -- worth ~+23 mm/month in the
    deep tropics, enough to fake an entire rainforest belt that then dissolved as
    the residue decayed over later years.
    """
    period = EARTH.orbital_period_days
    month_days = period / 12.0

    # Step 1: month 0, wet. All 12 bins get seeded with this flat field.
    s = _state(2, 2, 300.0, 10.0, day=0.5 * month_days)
    mt, mp, counts = update_monthly_statistics(
        s, month_days, orbital_period_days=period
    )
    assert counts[0] > 0.0 and np.count_nonzero(counts) == 1
    assert mp[0] == pytest.approx(10.0)

    # Step 2: month 6, bone dry. This is bin 6's FIRST real sample, so it must
    # read exactly 0.2 -- not a blend with the 10.0 seed.
    s2 = SimpleNamespace(
        temperature=np.full((2, 2), 280.0, dtype=np.float32),
        precipitation=np.full((2, 2), 0.2, dtype=np.float32),
        day_of_year=6.5 * month_days,
        monthly_temp=mt,
        monthly_precip=mp,
        monthly_sample_count=counts,
    )
    mt2, mp2, counts2 = update_monthly_statistics(
        s2, month_days, orbital_period_days=period
    )
    assert mp2[6] == pytest.approx(0.2), (
        "bin 6 retained spin-up seed; a flat seed blended into the driest bin is "
        "the A2 measurement bug"
    )
    assert mt2[6] == pytest.approx(280.0)
    # Untouched bins keep the seed (so Köppen stays classifiable from step 1)...
    assert mp2[3] == pytest.approx(10.0)
    assert counts2[3] == 0.0
    # ...and the already-visited bin still EMAs normally rather than overwriting.
    assert mp2[0] == pytest.approx(10.0)


def test_monthly_bin_second_sample_still_averages():
    """First-visit overwrite must not turn the bins into last-sample-wins."""
    period = EARTH.orbital_period_days
    month_days = period / 12.0
    s = _state(2, 2, 300.0, 10.0, day=0.5 * month_days)
    mt, mp, counts = update_monthly_statistics(s, month_days, orbital_period_days=period)

    # Revisit month 0 a year later with a different value.
    s2 = SimpleNamespace(
        temperature=np.full((2, 2), 290.0, dtype=np.float32),
        precipitation=np.full((2, 2), 2.0, dtype=np.float32),
        day_of_year=0.5 * month_days,
        monthly_temp=mt,
        monthly_precip=mp,
        monthly_sample_count=counts,
    )
    _, mp2, _ = update_monthly_statistics(s2, month_days, orbital_period_days=period)
    assert 2.0 < float(mp2[0].mean()) < 10.0, "second sample should blend, not replace"
