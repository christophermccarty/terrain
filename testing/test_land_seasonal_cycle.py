"""Land annual temperature cycle: the C1b mechanisms and the metric that sees them.

ACCURACY_AUDIT.md C1b has been worked across several sessions on numbers from
throwaway offline probes, so nothing about the land cycle was ever
regression-gated and each session re-derived the same figures. These tests cover
both halves of the 2026-08-03 pass: the tracked metric
(`real_terrain_validation.metrics.land_seasonal_cycle`) and the three gated
mechanisms it was built to evaluate.

The mechanisms all ship inert. What must hold regardless of any later
calibration is that each is an exact no-op at its default, that the deficit gate
really is self-limiting (zero in summer, full in a cold winter), and that the
thermal-inertia knob actually removes the step-length dependence it exists to
remove -- that last one is a latent bug in its own right, independent of C1b.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import EARTH, MARS  # noqa: E402
from real_terrain_validation import (  # noqa: E402
    EARTH_LAND_CYCLE_REFERENCE,
    _land_seasonal_cycle_metrics,
)
from optimizer.headless import _make_default_elevation  # noqa: E402
from simulate import (  # noqa: E402
    _LAND_TRANSPORT_DEFICIT_REF_K,
    create_initial_state,
    simulate_step,
)


def _fresh_state(planet_params=EARTH, height: int = 48, width: int = 96):
    return create_initial_state(
        _make_default_elevation(height, width), planet_params=planet_params
    )


class _FakeState:
    """Just enough of PlanetState for the metric under test."""

    def __init__(self, monthly_temp):
        self.monthly_temp = monthly_temp


def _cycle(shape, months_series):
    """Broadcast a 12-value cycle over a grid."""
    return np.broadcast_to(
        np.asarray(months_series, dtype=np.float32)[:, None, None], (12,) + shape
    ).copy()


def _land(shape, rows):
    mask = np.zeros(shape, dtype=bool)
    mask[rows, :] = True
    return mask


def _rows_for_band(height, name):
    reference = EARTH_LAND_CYCLE_REFERENCE[name]
    lat = (0.5 - (np.arange(height) + 0.5) / height) * 180.0
    return np.where((lat >= reference["lat_s"]) & (lat < reference["lat_n"]))[0]


class TestMetric:
    """The metric must separate shape from level -- the whole point of C1b."""

    shape = (64, 8)

    def _measure(self, series, band="35-45N"):
        rows = _rows_for_band(self.shape[0], band)
        state = _FakeState(_cycle(self.shape, series) + 273.15)
        report = _land_seasonal_cycle_metrics(state, _land(self.shape, rows))
        return report["bands"][band]

    def test_a_sinusoid_scores_six_months(self):
        months = np.arange(12)
        sinusoid = 10.0 + 14.0 * np.cos(2.0 * np.pi * (months + 0.5 - 6) / 12.0)
        assert self._measure(sinusoid)["squareness_months"] == pytest.approx(6.0, abs=0.51)

    def test_a_broad_warm_season_with_a_narrow_trough_scores_above_six(self):
        """The defect's actual signature, which is *asymmetry* in time.

        Worth stating precisely, because the obvious test is wrong: clamping a
        symmetric sinusoid does **not** move this score (checked -- it stays at
        exactly 6, because the clamp lowers the peak values and the annual mean
        together and the shape is still symmetric about its solstices). What
        raises it is a broad warm season against a *narrow deep* winter trough,
        which is what the model actually produces and what the clamp creates on
        an already-skewed radiative base.
        """
        square = [-25.0, -12.0, 18.0, 20.0, 21.0, 22.0, 22.0, 21.0, 20.0, 19.0, 15.0, -8.0]
        assert self._measure(square)["squareness_months"] > 7.0

    def test_plateau_months_detects_the_flat_top(self):
        """`squareness` and `plateau` must not be the same measurement twice.

        They are independent: the symmetric clamp above leaves squareness at 6
        while driving plateau from 2 to 6, so a run that flattened its summers
        without skewing the cycle would be invisible to one and obvious to the
        other.
        """
        months = np.arange(12)
        sinusoid = 10.0 + 14.0 * np.cos(2.0 * np.pi * (months + 0.5 - 6) / 12.0)
        clamped = np.minimum(sinusoid, 14.0)
        assert self._measure(sinusoid)["plateau_months"] < 3.0
        assert self._measure(clamped)["plateau_months"] > 5.0
        assert self._measure(clamped)["squareness_months"] == pytest.approx(
            self._measure(sinusoid)["squareness_months"]
        )

    def _score(self, series, band="35-45N"):
        rows = _rows_for_band(self.shape[0], band)
        state = _FakeState(_cycle(self.shape, series) + 273.15)
        return _land_seasonal_cycle_metrics(state, _land(self.shape, rows))[
            "cycle_error_score"
        ]

    def test_the_score_is_shape_only_and_ignores_a_level_shift(self):
        """`cycle_error_score` must not respond to a uniform warm/cold offset.

        Added 2026-08-04 (audit C1b-2026-08-04). The score used to include
        `warmest_bias_c` and `coldest_bias_c`, measured against
        `EARTH_LAND_CYCLE_REFERENCE`'s mid-continental station anchors -- while
        the metric averages over *all* land in the band. That population
        mismatch reported a 13 K warm bias at 25-35N where an anchor-free check
        finds the model correctly placed, and optimising it drove the subtropics
        cold. Level now belongs to `koppen_temperature_thresholds`; this score
        keeps only the two population-robust shape terms, so a pure offset must
        leave it untouched.
        """
        months = np.arange(12)
        sinusoid = 10.0 + 14.0 * np.cos(2.0 * np.pi * (months + 0.5 - 6) / 12.0)
        assert self._score(sinusoid) == pytest.approx(self._score(sinusoid + 15.0))
        assert self._score(sinusoid) == pytest.approx(self._score(sinusoid - 15.0))

    def test_the_score_still_responds_to_shape(self):
        """The complement of the test above: it must not be inert in general."""
        months = np.arange(12)
        sinusoid = 10.0 + 14.0 * np.cos(2.0 * np.pi * (months + 0.5 - 6) / 12.0)
        square = [-25.0, -12.0, 18.0, 20.0, 21.0, 22.0, 22.0, 21.0, 20.0, 19.0, 15.0, -8.0]
        assert self._score(square) > self._score(sinusoid) + 1.0

    def test_plateau_excess_is_amplitude_invariant_for_a_sinusoid(self):
        """A sinusoid must score ~0 excess plateau at *any* amplitude.

        The plateau test is in absolute kelvin, so a low-amplitude cycle spends
        more months within 1 K of its peak for reasons that have nothing to do
        with a clamp: a sinusoid scores ~1.4 months at 28 K peak-to-peak but
        ~1.9 at 16 K. Comparing each cell against its own amplitude's sinusoid
        removes that, which is what lets one score cover bands whose real
        amplitudes differ by 2x.
        """
        months = np.arange(12)
        for amplitude in (5.0, 8.0, 14.0, 20.0):
            series = 10.0 + amplitude * np.cos(2.0 * np.pi * (months + 0.5 - 6) / 12.0)
            excess = self._measure(series)["plateau_excess_months"]
            # 0.85 months of slack absorbs the 12-sample quantisation, and that
            # is its true worst case rather than a guess: the measured count is
            # an integer and cannot go below 2 when the peak falls between two
            # month centres, while the continuous reference reaches 1.21 at
            # 20 K amplitude. A real clamp shows up as 3-5 months of excess, so
            # the floor costs nothing in separation.
            assert excess == pytest.approx(0.0, abs=0.85), (
                f"amplitude {amplitude} K leaks {excess:.2f} months of false plateau"
            )

    def test_plateau_excess_still_catches_a_real_clamp(self):
        months = np.arange(12)
        sinusoid = 10.0 + 14.0 * np.cos(2.0 * np.pi * (months + 0.5 - 6) / 12.0)
        clamped = np.minimum(sinusoid, 14.0)
        assert self._measure(clamped)["plateau_excess_months"] > 2.0

    def test_level_and_shape_are_independent(self):
        """A pure offset must move the biases and leave squareness alone.

        Guards the failure mode that made C1b's earlier knobs look good: a
        metric that only watched shape would score a 10 K warm bias as neutral.
        """
        months = np.arange(12)
        sinusoid = 10.0 + 14.0 * np.cos(2.0 * np.pi * (months + 0.5 - 6) / 12.0)
        cold = self._measure(sinusoid)
        warm = self._measure(sinusoid + 10.0)
        assert warm["squareness_months"] == pytest.approx(cold["squareness_months"])
        assert warm["mean_bias_c"] == pytest.approx(cold["mean_bias_c"] + 10.0, abs=0.01)

    def test_returns_empty_without_monthly_temperature(self):
        assert _land_seasonal_cycle_metrics(
            _FakeState(None), np.ones(self.shape, dtype=bool)
        ) == {}


class TestDefaultsAreNoOps:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("land_transport_deficit_k", 0.0),
            ("land_transport_deficit_gain", 1.0),
            ("land_thermal_inertia_days", 0.0),
        ],
    )
    def test_shipped_default(self, field, value):
        assert getattr(EARTH, field) == pytest.approx(value)
        assert getattr(MARS, field) == pytest.approx(value)

    def test_gain_is_inert_while_the_gate_is_off(self):
        """The gain multiplies the *gated* term only, so it must do nothing
        until the gate is enabled -- otherwise it is a second, hidden magnitude
        knob on a calibrated set of trapezoids."""
        state = _fresh_state()
        base, _ = simulate_step(state, days=5.0, planet_params=EARTH)
        loud, _ = simulate_step(
            state,
            days=5.0,
            planet_params=dataclasses.replace(EARTH, land_transport_deficit_gain=4.0),
        )
        assert np.array_equal(base.temperature, loud.temperature)


class TestDeficitGate:
    """`clip((273.15 - T_pre) / D, 0, 1)`: self-limiting, no seasonal schedule."""

    @staticmethod
    def _gate(temperature_k, width_k):
        return float(
            np.clip((_LAND_TRANSPORT_DEFICIT_REF_K - temperature_k) / width_k, 0.0, 1.0)
        )

    def test_closed_in_summer(self):
        assert self._gate(303.0, 25.0) == 0.0

    def test_open_in_a_cold_winter(self):
        assert self._gate(240.0, 25.0) == 1.0

    def test_partially_open_near_freezing(self):
        assert 0.0 < self._gate(268.0, 25.0) < 1.0

    def test_narrower_width_admits_more_winter_transport(self):
        assert self._gate(261.0, 12.0) > self._gate(261.0, 40.0)

    def test_the_gate_changes_the_state(self):
        state = _fresh_state()
        base, _ = simulate_step(state, days=5.0, planet_params=EARTH)
        gated, _ = simulate_step(
            state,
            days=5.0,
            planet_params=dataclasses.replace(EARTH, land_transport_deficit_k=25.0),
        )
        assert not np.array_equal(base.temperature, gated.temperature)


class TestThermalInertia:
    """The land relaxation rate, and an honest bound on what fixing it buys.

    The historical `land_blend = 0.2` is a fraction per *call*, so the same
    elapsed time integrated as one long step or several short ones keeps
    `1 - 0.2 = 0.800` of the old temperature versus `(1 - 0.2)**12 = 0.069` --
    an order of magnitude, from identical physics, decided by the caller's step
    length. A real time constant makes the two agree exactly.

    **What this does NOT do, measured rather than assumed**: it does not make
    `simulate_step` step-length invariant end to end. On a 12-day span split 12
    ways the mean land-temperature discrepancy is 4.34 K with the fixed blend and
    **5.44 K** with tau = 27 days -- it goes *up*, because the residual is
    dominated by other terms that scale linearly in `days` (advection, diffusion,
    the evaporation budget), and changing the land rate changes how much of that
    shows through. So this knob is a correctness fix for one term, not a remedy
    for speed-switch differences; `test_end_to_end_invariance_is_not_restored`
    pins that so the claim is not quietly over-stated later.
    """

    @staticmethod
    def _retained(tau_days: float, total_days: float, splits: int) -> tuple[float, float]:
        """Fraction of the prior temperature the land term keeps, both ways."""
        if tau_days > 0.0:
            one = 1.0 - (1.0 - np.exp(-total_days / tau_days))
            each = 1.0 - (1.0 - np.exp(-(total_days / splits) / tau_days))
        else:
            one = 1.0 - 0.2
            each = 1.0 - 0.2
        return float(one), float(each ** splits)

    @pytest.mark.parametrize("tau", [15.0, 27.0, 60.0])
    def test_a_physical_time_constant_is_split_invariant(self, tau):
        one, split = self._retained(tau, 12.0, 12)
        assert one == pytest.approx(split, rel=1e-9)

    def test_the_historical_blend_is_not(self):
        """The bug, asserted so a revert cannot pass silently."""
        one, split = self._retained(0.0, 12.0, 12)
        assert one == pytest.approx(0.8)
        assert split == pytest.approx(0.0687, abs=1e-3)

    def test_end_to_end_invariance_is_not_restored(self):
        """Measured negative result, kept so it is not re-derived."""
        def discrepancy(planet_params) -> float:
            state = _fresh_state()
            one, _ = simulate_step(state, days=12.0, planet_params=planet_params)
            many = state
            for _ in range(12):
                many, _ = simulate_step(many, days=1.0, planet_params=planet_params)
            from masks import get_masks

            _, land = get_masks(state.elevation, use_cache=False)
            return float(
                np.mean(np.abs(np.asarray(one.temperature)[land]
                               - np.asarray(many.temperature)[land]))
            )

        fixed = discrepancy(EARTH)
        physical = discrepancy(dataclasses.replace(EARTH, land_thermal_inertia_days=27.0))
        assert fixed > 1.0 and physical > 1.0, (
            "step-length discrepancy has become small in both configurations -- "
            "if some other term was fixed, revisit this test's premise"
        )

    def test_inertia_changes_the_state(self):
        state = _fresh_state()
        base, _ = simulate_step(state, days=6.0, planet_params=EARTH)
        slow, _ = simulate_step(
            state,
            days=6.0,
            planet_params=dataclasses.replace(EARTH, land_thermal_inertia_days=60.0),
        )
        assert not np.array_equal(base.temperature, slow.temperature)
