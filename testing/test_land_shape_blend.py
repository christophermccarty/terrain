"""Guards for the land-wide raw-shape target blend (ACCURACY_AUDIT.md A5-LEAD).

`precip_land_shape_weight` ships **inert at 0.0** and is expected to stay that
way: swept on the tracked benchmark it degrades every bounded H10 metric
monotonically, reproducing the 2026-07-31 rejection of the ungated
`precip_raw_shape_weight` (arid share up, US Midwest collapsing toward 400
mm/yr). It exists so that the next session to reach for "blend raw production
shape into the target over all land" finds a tested mechanism and the audit's
sweep table rather than rebuilding it.

These tests therefore guard two things: that the default really is an exact
no-op, and that the parameter still does what it claims when enabled -- an inert
knob whose mechanism has silently rotted is worse than no knob, because the next
session would measure a null result and draw the wrong conclusion from it.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atmosphere import generate_precipitation  # noqa: E402
from planet_params import EARTH  # noqa: E402


def _terrain(height: int = 64, width: int = 128) -> np.ndarray:
    """Flat continent plus one ridge, so gated and ungated blends are separable."""
    elevation = np.zeros((height, width), dtype=np.float32)
    elevation[:, width // 4 : 3 * width // 4] = 0.02
    for offset in range(-6, 7):
        elevation[:, 64 + offset] = 0.02 + 0.5 * np.exp(-((offset / 2.5) ** 2))
    return elevation


def _humidity(height: int, width: int, *, uniform: bool) -> np.ndarray:
    """Uniform, or a strong within-row gradient.

    `_raw_shape` is each cell's `precip_potential` relative to its row's *land*
    mean, clipped to [0.2, 3.0]. Under a uniform humidity field it barely leaves
    1.0, so the clip guard's failure mode is unreachable and a test written
    against that fixture passes vacuously. The gradient below reproduces the
    real-terrain condition closely: `_raw_shape` sits at its 0.2 floor on 13.7%
    of land here versus 13.9% on the tracked 64x128 benchmark.
    """
    if uniform:
        return np.full((height, width), 0.01, dtype=np.float32)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    return (0.002 + 0.020 * np.exp(-(((x - 0.30) / 0.10) ** 2))).repeat(height, 0)


def _precip(elevation, planet_params, *, uniform_humidity: bool = True) -> np.ndarray:
    height, width = elevation.shape
    result = generate_precipitation(
        height,
        width,
        elevation,
        temperature=np.full((height, width), 288.0, dtype=np.float32),
        wind_u=np.full((height, width), 6.0, dtype=np.float32),
        wind_v=np.zeros((height, width), dtype=np.float32),
        humidity=_humidity(height, width, uniform=uniform_humidity),
        day_of_year=80.0,
        dt_days=1.0,
        planet_params=planet_params,
    )
    return np.asarray(result[0])


class TestDefaultIsInert:
    def test_default_is_zero(self):
        assert EARTH.precip_land_shape_weight == 0.0

    def test_zero_weight_is_bit_identical(self):
        """0.0 must be an exact no-op, not merely a small one.

        Every shape-blend weight in this project ships behind an exactly-zero
        default so a regression can be bisected by ablation instead of by git
        (process note 6). That only holds if zero is byte-exact.
        """
        elevation = _terrain()
        explicit_zero = dataclasses.replace(EARTH, precip_land_shape_weight=0.0)
        assert np.array_equal(_precip(elevation, EARTH), _precip(elevation, explicit_zero))


class TestMechanismStillWorks:
    def test_enabling_it_changes_precipitation(self):
        elevation = _terrain()
        base = _precip(elevation, EARTH)
        blended = _precip(
            elevation, dataclasses.replace(EARTH, precip_land_shape_weight=0.5)
        )
        assert not np.allclose(base, blended)

    def test_it_acts_on_flat_land_unlike_the_orographic_gate(self):
        """The defining difference from `precip_orographic_shape_weight`.

        A5-LEAD's whole finding is that these are *different mechanisms* -- the
        orographic gate is terrain-weighted and confined to relief (asserted in
        test_orographic_contrast.py), this one is deliberately land-wide. If this
        assertion ever flips, the two knobs have converged and the audit's
        conclusion no longer describes the code.
        """
        elevation = _terrain()
        flat = slice(36, 56)  # continent interior, well clear of the ridge

        def flat_effect(overrides) -> float:
            changed = _precip(elevation, dataclasses.replace(EARTH, **overrides))
            base = _precip(elevation, EARTH)
            return float(np.abs(changed[:, flat] / base[:, flat] - 1.0).mean())

        land_wide = flat_effect({"precip_land_shape_weight": 0.5})
        orographic = flat_effect({"precip_orographic_shape_weight": 1.5})
        assert land_wide > 5.0 * orographic, (
            f"land-wide blend is not reaching flat land: {land_wide:.4f} "
            f"vs orographic {orographic:.4f}"
        )


class TestBlendClipGuard:
    """The blend is a linear interpolation and is only valid on [0, 1].

    `_raw_shape` is clipped to [0.2, 3.0] and **sits at its 0.2 floor on 13.9% of
    land** on the tracked benchmark, so a weight above 1.25 extrapolates past the
    endpoint: measured with the guard removed, `_desert_factor` goes negative on
    **20.0% of land at weight 1.5** and 46.8% at weight 5.0. A negative
    `_desert_factor` becomes a negative `cell_weight`, i.e. a negative share of
    the row's precipitation target, which `_moisture_budget_precip_rescale` reads
    as "this cell is over its share by its whole current value plus more".

    Final precipitation stays non-negative regardless, because downstream clips
    (`np.clip(dq * cell_weight, 0.0, q)` and the row trim) absorb it -- so this
    fails *silently*, as a corrupted spatial weighting rather than an obviously
    broken field. That is exactly why it needs a test rather than a runtime
    assertion, and why asserting `precip >= 0` would not catch it. No shipped
    default reaches the extrapolating range, but these weights exist to be swept
    (A5-LEAD swept the orographic one to 5.0).
    """

    def test_weights_above_one_saturate_rather_than_extrapolate(self):
        elevation = _terrain()
        at_one = _precip(elevation, dataclasses.replace(EARTH, precip_land_shape_weight=1.0))
        at_five = _precip(elevation, dataclasses.replace(EARTH, precip_land_shape_weight=5.0))
        # This gate is land-only and unweighted by terrain, so at any weight >= 1
        # the blend saturates at exactly 1.0 on every land cell -- the two runs
        # must be identical, not merely close. Verified to fail when the clip is
        # removed (planted-violation check, 2026-08-02).
        assert np.array_equal(at_one, at_five)

    # `precip_orographic_shape_weight` -- the knob actually swept in practice --
    # shares `_apply_shape_blend` with this one, so it is covered by the same
    # assertion. It has no separate test here because its own gate is a
    # continuum of small values rather than a step, so there is no pair of
    # weights whose results must be *exactly* equal: cells with a gate between
    # 1/w1 and 1/w2 legitimately differ, and an approximate comparison would not
    # discriminate the guard from its absence.
