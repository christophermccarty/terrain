"""Seasonal migration of the subtropical dry belt and the storm track.

`itcz_window` and `_zonal_precip_target_profile` were each found (2026-07-30 and
2026-07-31) to be pure functions of latitude with **zero `day_of_year`
dependence**, and both fixes recorded in their own docstrings that
`drybelt_window`/`storm_window` were left untouched. Those two windows are the
third instance of the same defect, and the one that carries Mediterranean
climate: `climate_averages.classify_koppen` reaches Cs only when the driest
summer month is under a third of the wettest winter month, which cannot happen
anywhere on a planet whose subtropical high never moves.

Most tests here assert the *structure* of the migration rather than a calibrated
strength, so they stay valid if the responses are re-tuned: it must be
antisymmetric between hemispheres, single-peaked, periodic over one orbit,
driven by the planet's own obliquity rather than an Earth constant, and an exact
no-op at 0.0. `TestShippedValues` is the deliberate exception -- it pins the
calibrated pair so moving either is a conscious edit rather than a drift.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atmosphere import generate_precipitation  # noqa: E402
from planet_params import EARTH, MARS  # noqa: E402

NH_SOLSTICE = 172.0
SH_SOLSTICE = 355.0
EQUINOX = 80.0


def _fields(planet_params, day_of_year: float, *, height: int = 64, width: int = 128):
    elevation = np.zeros((height, width), dtype=np.float32)
    elevation[:, width // 4:3 * width // 4] = 0.02
    debug: dict = {}
    generate_precipitation(
        height,
        width,
        elevation,
        temperature=np.full((height, width), 288.0, dtype=np.float32),
        wind_u=np.full((height, width), 6.0, dtype=np.float32),
        wind_v=np.zeros((height, width), dtype=np.float32),
        humidity=np.full((height, width), 0.01, dtype=np.float32),
        day_of_year=day_of_year,
        dt_days=1.0,
        planet_params=planet_params,
        debug_fields=debug,
    )
    return debug


def _latitudes(height: int = 64) -> np.ndarray:
    return (0.5 - (np.arange(height, dtype=np.float64) + 0.5) / height) * 180.0


def _belt_peak_latitude(planet_params, day_of_year: float, hemisphere: int) -> float:
    """Latitude where `subsidence_suppression` is strongest, i.e. the dry belt.

    Read from the zonal mean of a shipped debug field rather than from the
    window directly, so the test fails if the window stops reaching the physics
    it is supposed to drive -- a window that migrates but is cached, gated away,
    or overwritten downstream would pass a direct check and fail this one.
    """
    suppression = np.asarray(_fields(planet_params, day_of_year)["subsidence_suppression"])
    zonal = suppression.mean(axis=1)
    lat = _latitudes(suppression.shape[0])
    band = (lat > 5.0) if hemisphere > 0 else (lat < -5.0)
    return float(lat[band][np.argmin(zonal[band])])


STATIC = dataclasses.replace(EARTH, drybelt_seasonal_response=0.0)


class TestShippedValues:
    def test_shipped_calibration(self):
        """Pin the calibrated pair, so a change to either is a deliberate edit.

        0.25 is the group-accuracy/kappa optimum of the 128x256 sweep *and* the
        largest response leaving US Midwest inside its 800-1000 target; the
        equatorward fraction is 0.0 because a rigid translation carries the belt
        off the Sahara for half the year. See ACCURACY_AUDIT.md A6.
        """
        assert EARTH.drybelt_seasonal_response == 0.25
        assert EARTH.drybelt_seasonal_equatorward_fraction == 0.0

    def test_storm_track_ships_inert(self):
        """Measured, not assumed: at 0.3 it moves nothing above noise."""
        assert EARTH.storm_track_seasonal_response == 0.0
        assert MARS.storm_track_seasonal_response == 0.0

    @pytest.mark.parametrize("day", [EQUINOX, NH_SOLSTICE, SH_SOLSTICE])
    def test_zero_response_restores_a_static_belt_exactly(self, day):
        """0.0 must remain an exact no-op, so the mechanism stays ablatable."""
        reference = _fields(STATIC, EQUINOX)["subsidence_suppression"]
        assert np.array_equal(
            np.asarray(_fields(STATIC, day)["subsidence_suppression"]),
            np.asarray(reference),
        )

    @pytest.mark.parametrize("day", [NH_SOLSTICE, SH_SOLSTICE])
    def test_the_shipped_default_is_not_static(self, day):
        """Guards against the belt silently reverting to a |lat|-only window --
        the defect this whole mechanism exists to fix, which sat undetected in
        `itcz_window` and `_zonal_precip_target_profile` before it."""
        assert not np.array_equal(
            np.asarray(_fields(EARTH, day)["drybelt_regime_window"]),
            np.asarray(_fields(EARTH, EQUINOX)["drybelt_regime_window"]),
        )


def _rigid(response: float):
    """The pure translation, isolating the shift itself from the edge pinning."""
    return dataclasses.replace(
        EARTH,
        drybelt_seasonal_response=response,
        drybelt_seasonal_equatorward_fraction=1.0,
    )


class TestMigration:
    def test_northern_belt_moves_poleward_in_northern_summer(self):
        moving = _rigid(0.4)
        summer = _belt_peak_latitude(moving, NH_SOLSTICE, hemisphere=+1)
        winter = _belt_peak_latitude(moving, SH_SOLSTICE, hemisphere=+1)
        assert summer > winter, f"NH belt: summer {summer:.1f} !> winter {winter:.1f}"

    def test_migration_is_hemisphere_antisymmetric(self):
        """The two halves of the belt move in opposite directions at once.

        This is the property that distinguishes this mechanism from
        `itcz_seasonal_response`, whose single signed centre moves the whole
        belt one way. A symmetric implementation (shifting `abs_lat_deg` by a
        signless amount) would pass the test above and fail this one.
        """
        moving = _rigid(0.4)
        north = _belt_peak_latitude(moving, NH_SOLSTICE, hemisphere=+1)
        south = _belt_peak_latitude(moving, NH_SOLSTICE, hemisphere=-1)
        static_north = _belt_peak_latitude(STATIC, NH_SOLSTICE, hemisphere=+1)
        static_south = _belt_peak_latitude(STATIC, NH_SOLSTICE, hemisphere=-1)
        assert north > static_north, "NH belt did not move poleward"
        assert abs(south) < abs(static_south), "SH belt did not move equatorward"

    def test_migration_amplitude_scales_with_the_response(self):
        swing = lambda pp: (  # noqa: E731
            _belt_peak_latitude(pp, NH_SOLSTICE, hemisphere=+1)
            - _belt_peak_latitude(pp, SH_SOLSTICE, hemisphere=+1)
        )
        assert swing(_rigid(0.6)) > swing(_rigid(0.2)) > 0.0

    def test_belt_returns_to_the_same_place_after_one_orbit(self):
        moving = _rigid(0.4)
        day = 120.0
        first = _fields(moving, day)["subsidence_suppression"]
        later = _fields(moving, day + EARTH.orbital_period_days)["subsidence_suppression"]
        assert np.allclose(np.asarray(first), np.asarray(later), atol=1e-5)

    def test_storm_track_response_is_independently_gated(self):
        """Each belt must be ablatable on its own -- process note 6."""
        drybelt_only = dataclasses.replace(EARTH, drybelt_seasonal_response=0.4)
        storm_only = dataclasses.replace(EARTH, storm_track_seasonal_response=0.4)
        base = np.asarray(_fields(EARTH, NH_SOLSTICE)["precip_potential_prerescale"])
        moved_dry = np.asarray(
            _fields(drybelt_only, NH_SOLSTICE)["precip_potential_prerescale"]
        )
        moved_storm = np.asarray(
            _fields(storm_only, NH_SOLSTICE)["precip_potential_prerescale"]
        )
        assert not np.allclose(base, moved_dry)
        assert not np.allclose(base, moved_storm)
        assert not np.allclose(moved_dry, moved_storm)


class TestAsymmetricMigration:
    """The belt widens poleward in summer rather than translating rigidly.

    A rigid translation is what a first implementation naturally does, and it is
    measurably wrong here: it carries `subsidence_suppression` off the
    subtropical deserts for half the year (Sahara 129 -> 223 mm/yr at
    `drybelt_seasonal_response=0.3`, through its <200 target). Earth's Hadley
    descending branch is bounded equatorward by the ITCZ, which migrates far
    less than its poleward edge does.
    """

    @staticmethod
    def _suppression(equatorward_fraction: float, day: float) -> np.ndarray:
        planet = dataclasses.replace(
            EARTH,
            drybelt_seasonal_response=0.4,
            drybelt_seasonal_equatorward_fraction=equatorward_fraction,
        )
        return np.asarray(_fields(planet, day)["subsidence_suppression"]).mean(axis=1)

    def test_pinning_the_equatorward_edge_keeps_the_desert_belt_suppressed(self):
        """The whole point: 15-22N must stay dry through NH summer."""
        lat = _latitudes()
        sahara = (lat >= 15.0) & (lat <= 22.0)
        rigid = self._suppression(1.0, NH_SOLSTICE)[sahara].mean()
        pinned = self._suppression(0.0, NH_SOLSTICE)[sahara].mean()
        # Lower `subsidence_suppression` means MORE suppression of precipitation.
        assert pinned < rigid, (
            f"pinned edge suppression {pinned:.3f} !< rigid {rigid:.3f}"
        )

    def test_the_poleward_edge_still_migrates_when_pinned(self):
        """Pinning must not disable the Mediterranean mechanism itself."""
        lat = _latitudes()
        mediterranean = (lat >= 35.0) & (lat <= 42.0)
        summer = self._suppression(0.0, NH_SOLSTICE)[mediterranean].mean()
        winter = self._suppression(0.0, SH_SOLSTICE)[mediterranean].mean()
        assert summer < winter, (
            f"35-42N is not drier in summer: {summer:.3f} vs winter {winter:.3f}"
        )

    @pytest.mark.parametrize("response", [0.3, 0.5, 0.8, 1.0])
    def test_belt_never_develops_a_second_peak(self, response):
        """A latitude-dependent shift can fold the coordinate onto itself.

        If it does, two distinct latitudes map to the same point in the belt and
        the window grows a spurious second maximum. The transition width widens
        with the shift specifically to bound the warp's slope below 1; these
        weights exist to be swept, so the guard is checked past every shipped
        value rather than only at one. Read from `drybelt_window` directly --
        `subsidence_suppression` also carries the divergence field and a
        smoothing pass, and would hide the defect.
        """
        planet = dataclasses.replace(
            EARTH,
            drybelt_seasonal_response=response,
            drybelt_seasonal_equatorward_fraction=0.0,
        )
        window = np.asarray(_fields(planet, NH_SOLSTICE)["drybelt_window"])
        lat = _latitudes(window.shape[0]) if window.ndim > 1 else _latitudes(window.size)
        profile = window.mean(axis=1) if window.ndim > 1 else window
        northern = profile[lat > 2.0]
        rising = np.diff(northern) > 0
        # A single-peaked profile switches from rising to falling exactly once.
        switches = int(np.count_nonzero(np.diff(rising.astype(np.int8)) != 0))
        assert switches <= 1, (
            f"drybelt_window has {switches + 1} peaks at response={response}"
        )


class TestObliquityDriven:
    def test_a_zero_obliquity_planet_has_no_migration(self):
        """The driver is `solar_declination`, not a hardcoded Earth season."""
        flat = dataclasses.replace(
            EARTH, obliquity_deg=0.0, drybelt_seasonal_response=0.6
        )
        summer = np.asarray(_fields(flat, NH_SOLSTICE)["subsidence_suppression"])
        winter = np.asarray(_fields(flat, SH_SOLSTICE)["subsidence_suppression"])
        assert np.allclose(summer, winter, atol=1e-6)

    def test_higher_obliquity_migrates_further(self):
        def swing(obliquity: float) -> float:
            planet = dataclasses.replace(_rigid(0.4), obliquity_deg=obliquity)
            return _belt_peak_latitude(
                planet, NH_SOLSTICE, hemisphere=+1
            ) - _belt_peak_latitude(planet, SH_SOLSTICE, hemisphere=+1)

        assert swing(40.0) > swing(15.0) > 0.0

    def test_shift_matches_the_declination_formula(self):
        """Pin the actual magnitude, not just its ordering."""
        response = 0.4
        expected = response * math.degrees(EARTH.solar_declination(NH_SOLSTICE))
        moved = _belt_peak_latitude(_rigid(response), NH_SOLSTICE, hemisphere=+1)
        static = _belt_peak_latitude(STATIC, NH_SOLSTICE, hemisphere=+1)
        # One grid row at 64 rows is 2.8 degrees, so allow that quantization.
        assert moved - static == pytest.approx(expected, abs=3.0)
