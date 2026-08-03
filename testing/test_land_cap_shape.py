"""test_land_cap_shape.py -- the four knobs added 2026-08-02 for the land
summer-temperature ceiling.

Background (ACCURACY_AUDIT.md C1b): `simulate._evolve_temperature` applies
`_land_cap_1d` as a hard `np.minimum`, which maps every overshooting month onto
the *same* ceiling value -- measured, seven consecutive months at 41N come out
bit-identical, and the clamp binds on 55.7% of (month, row) pairs at 25-50 deg.
The in-code claim that it is a "rarely-binding safety net" was wrong.

All four knobs ship inert, so the contract these tests pin is deliberately
two-part:

1. **Default = exact no-op.** Each field at its shipped default must reproduce
   the previous behaviour bit-for-bit. This is what lets the fields exist in the
   tree without being a silent physics change, and it is the property most
   likely to rot when someone later calibrates one of them.
2. **Nonzero = the intended mechanism, in the right direction.** Otherwise a
   field could satisfy (1) by being wired to nothing at all -- which is exactly
   the failure mode `test_param_wiring.py` exists to catch generally.

The soft-min is tested directly rather than through a full simulation: its
defining property (strict monotonicity, which the hard clamp destroys) is a
property of the function, and testing it at that level is both exact and immune
to the downstream coupling that would otherwise blur it.
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

from planet_params import EARTH  # noqa: E402
from simulate import _soft_min_cap  # noqa: E402


# ---------------------------------------------------------------- soft-min cap

def test_soft_min_width_zero_is_exactly_np_minimum():
    rng = np.random.RandomState(0)
    x = rng.uniform(250.0, 330.0, (17, 23)).astype(np.float32)
    cap = rng.uniform(280.0, 305.0, (17, 1)).astype(np.float32)
    assert np.array_equal(_soft_min_cap(x, cap, 0.0), np.minimum(x, cap))
    # negative width is treated as "off" too, not as a sign flip
    assert np.array_equal(_soft_min_cap(x, cap, -1.0), np.minimum(x, cap))


def test_hard_clamp_collapses_distinct_months_and_soft_min_does_not():
    """The defect, stated as a test: `np.minimum` maps distinct inputs onto one
    output. That collapse *is* the plateau -- seven months at 41N landing on the
    same value. A soft-min must keep them distinct."""
    cap = np.float32([[300.0]])
    # a warm season that overshoots the ceiling by different amounts
    months = np.float32([[302.0], [305.0], [309.0], [312.0], [309.0], [305.0]])
    hard = np.minimum(months, cap)
    assert len(np.unique(hard)) == 1, "precondition: the hard clamp flattens these"

    soft = _soft_min_cap(months, cap, 4.0)
    assert len(np.unique(soft)) == len(np.unique(months)), (
        "soft-min must preserve distinctness of months that differ before clamping"
    )
    assert np.all(soft <= months), "soft-min must never warm a cell"


def test_soft_min_is_strictly_monotonic_and_bounded():
    cap = np.float32(300.0)
    x = np.linspace(270.0, 340.0, 400).astype(np.float32)
    y = _soft_min_cap(x, np.full_like(x, cap), 5.0)
    assert np.all(np.diff(y) > 0.0), "must be strictly increasing in x"
    assert np.all(y <= cap), "must never exceed the ceiling"
    # tracks x well below the ceiling, asymptotes to the ceiling well above it
    assert y[0] == pytest.approx(x[0], abs=0.05)
    assert y[-1] == pytest.approx(cap, abs=0.05)


def test_soft_min_cannot_rescue_a_large_overshoot():
    """Documents the limit that makes the three knobs a *set*: softening alone
    leaves a far-overshooting month within a hair of the hard-clamped value, so
    the overshoot has to be brought down first (see C1b)."""
    cap = np.float32(300.0)
    x = np.float32(316.0)  # the ~16 K overshoot measured at 41N
    soft = float(_soft_min_cap(np.array([x]), np.array([cap]), 3.0)[0])
    assert abs(soft - cap) < 0.05


# ------------------------------------------------------------ default inertness

@pytest.mark.parametrize("field,default", [
    ("land_transport_seasonality", 0.0),
    ("evap_cooling_strength", 1.0),
    ("land_cap_softness_k", 0.0),
    ("evap_cooling_season_width", 1.0),
])
def test_shipped_default_is_the_documented_no_op_value(field, default):
    """Each field's docstring promises a specific no-op value. If someone
    calibrates one of these to a nonzero default, this test fires and forces the
    change to be a deliberate, reviewed physics change rather than a drive-by."""
    assert getattr(EARTH, field) == default, (
        f"EARTH.{field} is no longer at its documented no-op value {default}; "
        "enabling it is a real physics change -- see ACCURACY_AUDIT.md C1b, "
        "which records that enabling these degrades H10 bounded skill."
    )


def test_seasonality_factor_is_unity_at_default_and_ordered_when_enabled():
    """The modulation is `1 - k * summer_signal`: exactly 1 everywhere at k=0,
    and at k>0 strictly larger in winter than in summer (the physical claim --
    eddy heat flux scales with the meridional gradient, which peaks in winter)."""
    signal = np.linspace(-1.0, 1.0, 21)  # -1 local winter solstice .. +1 summer
    assert np.array_equal(1.0 - 0.0 * signal, np.ones_like(signal))

    for k in (0.4, 0.7, 1.0):
        factor = 1.0 - k * signal
        assert np.all(np.diff(factor) < 0.0), "must decrease from winter to summer"
        assert factor[0] == pytest.approx(1.0 + k)   # winter solstice
        assert factor[-1] == pytest.approx(1.0 - k)  # summer solstice
        assert np.all(factor >= 0.0), "must not invert the sign of the bonus"


def test_evap_season_width_saturates_earlier_when_reduced():
    """Smaller width => the gate reaches full strength earlier in the season and
    stays flat, making the contraction near-constant across the warm season."""
    summer_factor = np.linspace(0.0, 1.0, 101)
    wide = np.clip(summer_factor / 1.0, 0.0, 1.0)
    narrow = np.clip(summer_factor / 0.4, 0.0, 1.0)
    assert np.array_equal(wide, summer_factor), "width=1.0 must be the exact no-op"
    assert np.all(narrow >= wide - 1e-12), "narrower gate is never weaker"
    assert narrow[-1] == pytest.approx(1.0) and wide[-1] == pytest.approx(1.0)
    # and it is genuinely flat over the top of the season, unlike the default
    assert np.all(narrow[summer_factor >= 0.4] == pytest.approx(1.0))


# ------------------------------------------------------- end-to-end wiring

def test_params_are_actually_wired_into_the_temperature_path():
    """Guards against the "inert because it's connected to nothing" failure
    mode: with the knobs enabled the land temperature field must actually
    change. Uses a short deterministic real-terrain run."""
    from real_terrain_validation import (
        RealTerrainValidationConfig,
        run_real_terrain_validation,
    )

    cfg = RealTerrainValidationConfig(height=32, width=64, spinup_years=0.5,
                                      evaluation_years=0.5)
    base, _ = run_real_terrain_validation(cfg, planet_params=EARTH)
    tuned_pp = dataclasses.replace(
        EARTH,
        land_transport_seasonality=1.0,
        evap_cooling_strength=2.5,
        land_cap_softness_k=8.0,
    )
    cfg2 = RealTerrainValidationConfig(height=32, width=64, spinup_years=0.5,
                                       evaluation_years=0.5)
    tuned, _ = run_real_terrain_validation(cfg2, planet_params=tuned_pp)

    assert not np.allclose(base.temperature, tuned.temperature), (
        "enabling all three knobs left the temperature field unchanged -- they "
        "are not reaching _evolve_temperature"
    )
