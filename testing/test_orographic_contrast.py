"""Guards for the orographic-uplift mechanism (ACCURACY_AUDIT.md A5).

The bug this suite exists to prevent recurring is subtle and was live for a long
time: `orog` was normalized by a percentile taken over the *whole grid* while
ocean cells had already been zeroed, so on Earth the divisor was ~3.8x too small,
the clip downstream saturated, and the term reported an identical value on both
flanks of every mountain range. Nothing about that is visible in a magnitude
check or a global mean -- only a contrast check catches it, which is why these
tests assert on windward-vs-leeward differences rather than on levels.
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

from atmosphere import generate_precipitation  # noqa: E402
from planet_params import EARTH  # noqa: E402
from regional_validation import (  # noqa: E402
    OROGRAPHIC_PAIRS,
    orographic_contrast,
    region_mask,
)


def _ridge(height: int = 64, width: int = 128, *, crest: int = 64) -> np.ndarray:
    """A single north-south ridge on an otherwise flat continent."""
    elevation = np.zeros((height, width), dtype=np.float32)
    land = slice(width // 4, 3 * width // 4)
    elevation[:, land] = 0.02
    for offset in range(-6, 7):
        elevation[:, crest + offset] = 0.02 + 0.5 * np.exp(-((offset / 2.5) ** 2))
    return elevation


def _precip_fields(elevation, planet_params, *, u=6.0, v=0.0):
    height, width = elevation.shape
    debug: dict = {}
    result = generate_precipitation(
        height,
        width,
        elevation,
        temperature=np.full((height, width), 288.0, dtype=np.float32),
        wind_u=np.full((height, width), u, dtype=np.float32),
        wind_v=np.full((height, width), v, dtype=np.float32),
        humidity=np.full((height, width), 0.01, dtype=np.float32),
        day_of_year=80.0,
        dt_days=1.0,
        planet_params=planet_params,
        debug_fields=debug,
    )
    debug["P"] = np.asarray(result[0])
    return debug


class TestOrographicSign:
    def test_windward_uplift_exceeds_leeward_on_a_ridge(self):
        """The whole point of the term. Westerly wind -> west flank wet."""
        crest = 64
        fields = _precip_fields(_ridge(crest=crest), EARTH, u=6.0)
        orog = fields["orog"]
        windward = float(orog[:, crest - 4:crest - 1].mean())
        leeward = float(orog[:, crest + 2:crest + 5].mean())
        assert windward > leeward, f"windward {windward:.4f} !> leeward {leeward:.4f}"

    def test_reversing_the_wind_reverses_the_flank(self):
        """Guards the gx/gy sign convention independently of terrain shape."""
        crest = 64
        west = _precip_fields(_ridge(crest=crest), EARTH, u=6.0)["orog"]
        east = _precip_fields(_ridge(crest=crest), EARTH, u=-6.0)["orog"]
        west_flank = slice(crest - 4, crest - 1)
        east_flank = slice(crest + 2, crest + 5)
        assert west[:, west_flank].mean() > west[:, east_flank].mean()
        assert east[:, east_flank].mean() > east[:, west_flank].mean()

    def test_rain_shadow_suppresses_the_lee(self):
        crest = 64
        fields = _precip_fields(_ridge(crest=crest), EARTH, u=6.0)
        suppression = fields["rain_shadow_suppression"]
        assert (
            suppression[:, crest + 2:crest + 5].mean()
            < suppression[:, crest - 4:crest - 1].mean()
        )


class TestNormalizer:
    def test_land_only_normalizer_reduces_clip_saturation(self):
        """The bug's fingerprint: a grid-wide percentile over a mostly-ocean
        field drives the clip into saturation and flattens the contrast."""
        from masks import get_masks
        from real_terrain_validation import load_bundled_earth_dem

        elevation = load_bundled_earth_dem(128, 256)
        _, land = get_masks(elevation, use_cache=False)
        clip = float(EARTH.orographic_uplift_clip)

        def saturated(land_only: bool) -> float:
            params = dataclasses.replace(
                EARTH, orographic_normalizer_land_only=land_only
            )
            orog = _precip_fields(elevation, params)["orog"]
            return float(np.mean(orog[land] >= clip - 1e-6))

        assert saturated(True) < 0.5 * saturated(False)

    def test_land_only_normalizer_raises_windward_leeward_contrast(self):
        from masks import get_masks
        from real_terrain_validation import load_bundled_earth_dem

        elevation = load_bundled_earth_dem(256, 512)
        _, land = get_masks(elevation, use_cache=False)

        def contrast(land_only: bool) -> float:
            params = dataclasses.replace(
                EARTH, orographic_normalizer_land_only=land_only
            )
            orog = _precip_fields(elevation, params)["orog"]
            ratios = [
                orographic_contrast(orog, pair, land_mask=land)["ratio"]
                for pair in OROGRAPHIC_PAIRS
            ]
            return float(np.mean([r for r in ratios if np.isfinite(r)]))

        assert contrast(True) > contrast(False)


class TestInertHandles:
    """The three A5 ablation handles that ship at their historical values.

    Each must be a genuine no-op at its default, so a future session can move one
    in isolation and attribute the result -- process note 6's whole argument for
    gating new mechanisms.
    """

    @pytest.mark.parametrize(
        "field,value",
        [
            ("precip_orographic_weight", 0.20),
            ("precip_potential_ceiling", 3.0),
            ("precip_rain_out_ceiling", 0.85),
            ("orographic_uplift_clip", 2.0),
        ],
    )
    def test_default_value_is_the_shipped_value(self, field, value):
        assert getattr(EARTH, field) == pytest.approx(value)

    def test_orographic_weight_preserves_total_sum_weight(self):
        """Raising orography's share must not silently inflate the whole sum."""
        elevation = _ridge()
        flat = np.full_like(elevation, 0.02)  # no orography at all
        base = _precip_fields(flat, EARTH)["precip_potential_prerescale"]
        raised = _precip_fields(
            flat, dataclasses.replace(EARTH, precip_orographic_weight=0.6)
        )["precip_potential_prerescale"]
        # With orog identically zero, the five rescaled terms carry everything,
        # so a change here would be pure magnitude drift.
        assert float(base.mean()) > 0.0
        assert float(raised.mean()) < float(base.mean())

    def test_shape_blend_acts_on_relief_not_on_flat_land(self):
        """The gate is the orographic signal, so flat land must barely move.

        Not *exactly* zero: the blend feeds `cell_weight`, which is renormalized
        to a row mean of 1.0, so a change anywhere in a row redistributes weight
        across the whole row -- including its flat cells. That is inherent to a
        mean-preserving weight, not a leak in the gate. What must hold is that
        the effect is confined to relief by orders of magnitude, so the
        assertion is a comparison rather than a hardcoded tolerance.
        """
        off = dataclasses.replace(EARTH, precip_orographic_shape_weight=0.0)
        flat = np.full((64, 128), 0.02, dtype=np.float32)
        flat[:, :16] = 0.0

        def relative_change(elevation, columns) -> float:
            on_p = _precip_fields(elevation, EARTH)["P"][:, columns]
            off_p = _precip_fields(elevation, off)["P"][:, columns]
            return float(np.abs(on_p / off_p - 1.0).mean())

        flat_effect = relative_change(flat, slice(24, 112))
        ridge_effect = relative_change(_ridge(), slice(58, 63))
        assert flat_effect < 0.02, f"flat land moved {flat_effect:.3f}"
        assert ridge_effect > 20.0 * flat_effect, (
            f"gate is not orographically confined: flat {flat_effect:.4f} "
            f"vs ridge {ridge_effect:.4f}"
        )


class TestUpwindFootprint:
    """A5-OROG's named next lever: the uplift term's *shape*, not its magnitude.

    All four pointwise ceilings in the pipeline were measured and exhausted, and
    the residual defect is geometric -- `clip(gx*u + gy*v, 0, None)` on a real
    DEM is a 1-2 cell spike on the crest, while real orographic precipitation
    covers a broad windward flank. A box mean therefore dilutes a signal that is
    correct exactly where it exists and absent everywhere else in the box.
    """

    @staticmethod
    def _with(**overrides):
        return dataclasses.replace(EARTH, **overrides)

    def test_defaults_are_no_ops(self):
        assert EARTH.orographic_upwind_footprint_km == 0.0
        assert EARTH.orographic_spillover_km == 0.0

    def test_zero_footprint_is_bit_identical(self):
        elevation = _ridge()
        base = _precip_fields(elevation, EARTH)["orog"]
        explicit = _precip_fields(
            elevation, self._with(orographic_upwind_footprint_km=0.0)
        )["orog"]
        assert np.array_equal(base, explicit)

    def test_footprint_broadens_the_windward_flank(self):
        """The mechanism's whole purpose: signal further upwind of the crest."""
        crest = 64
        elevation = _ridge(crest=crest)
        far_upwind = slice(crest - 12, crest - 7)  # outside the resolved slope

        def upwind_share(footprint_km: float) -> float:
            orog = _precip_fields(
                elevation, self._with(orographic_upwind_footprint_km=footprint_km)
            )["orog"]
            return float(orog[:, far_upwind].mean() / orog[:, crest - 2:crest + 1].mean())

        assert upwind_share(400.0) > upwind_share(0.0)

    def test_footprint_does_not_leak_into_the_lee(self):
        """Upwind and downwind are separate knobs precisely so this holds.

        A symmetric smoothing would raise the leeward flank as much as the
        windward one and destroy the contrast it is meant to build.
        """
        crest = 64
        elevation = _ridge(crest=crest)
        lee = slice(crest + 7, crest + 12)

        def lee_share(footprint_km: float) -> float:
            orog = _precip_fields(
                elevation, self._with(orographic_upwind_footprint_km=footprint_km)
            )["orog"]
            return float(orog[:, lee].mean() / orog[:, crest - 2:crest + 1].mean())

        assert lee_share(400.0) <= lee_share(0.0) + 1e-4

    def test_spillover_moves_signal_the_other_way(self):
        crest = 64
        elevation = _ridge(crest=crest)
        lee = slice(crest + 7, crest + 12)

        def lee_share(spillover_km: float) -> float:
            orog = _precip_fields(
                elevation, self._with(orographic_spillover_km=spillover_km)
            )["orog"]
            return float(orog[:, lee].mean() / orog[:, crest - 2:crest + 1].mean())

        assert lee_share(400.0) > lee_share(0.0)

    def test_reversing_the_wind_reverses_the_footprint(self):
        crest = 64
        elevation = _ridge(crest=crest)
        params = self._with(orographic_upwind_footprint_km=400.0)
        west = _precip_fields(elevation, params, u=6.0)["orog"]
        east = _precip_fields(elevation, params, u=-6.0)["orog"]
        west_side = slice(crest - 12, crest - 7)
        east_side = slice(crest + 7, crest + 12)
        assert west[:, west_side].mean() > west[:, east_side].mean()
        assert east[:, east_side].mean() > east[:, west_side].mean()

    def test_uniform_field_is_preserved(self):
        """The smear is a weighted average, so flat terrain must not drift.

        Guards against the footprint acting as a hidden global gain: `orog` is
        percentile-normalized downstream, so a magnitude change there would be
        silently absorbed and only show up as a mis-calibrated clip fraction.
        """
        from atmosphere import _smear_along_wind

        shape = (32, 64)
        field = np.full(shape, 0.7, dtype=np.float32)
        smeared = _smear_along_wind(
            field,
            np.full(shape, 5.0, dtype=np.float32),
            np.full(shape, 2.0, dtype=np.float32),
            np.full(shape, 3.0e5, dtype=np.float64),
            3.0e5,
            300.0,
            120.0,
        )
        assert np.allclose(smeared, field, atol=1e-5)

    def test_footprint_is_a_physical_distance_not_a_cell_count(self):
        """Resolution invariance, the failure mode A5 had to fix once already.

        The monsoon inland mask shipped with a fixed 20-cell reach, which meant
        7 degrees at 1024 columns and 56 degrees in the 128-column fixture. A
        footprint expressed in kilometres must cover the same fraction of a
        ridge at any resolution, so the same physical smear measures the same
        upwind share on a ridge scaled with the grid.
        """
        params = self._with(orographic_upwind_footprint_km=600.0)

        def scaled_ridge(height: int, width: int, crest: int) -> np.ndarray:
            """`_ridge` with a crest of fixed *angular* width, not cell width."""
            elevation = np.zeros((height, width), dtype=np.float32)
            elevation[:, width // 4:3 * width // 4] = 0.02
            scale = width / 128.0
            for offset in range(-int(6 * scale), int(6 * scale) + 1):
                elevation[:, crest + offset] = 0.02 + 0.5 * np.exp(
                    -((offset / (2.5 * scale)) ** 2)
                )
            return elevation

        def upwind_share(height: int) -> float:
            width = 2 * height
            crest = width // 2
            orog = _precip_fields(scaled_ridge(height, width, crest), params)["orog"]
            scale = width // 128
            far = slice(crest - 12 * scale, crest - 7 * scale)
            peak = slice(crest - 2 * scale, crest + 1 * scale)
            return float(orog[:, far].mean() / orog[:, peak].mean())

        coarse, fine = upwind_share(64), upwind_share(128)
        assert abs(coarse - fine) < 0.35 * max(coarse, fine), (
            f"footprint is resolution-dependent: {coarse:.3f} vs {fine:.3f}"
        )


class TestPairDefinitions:
    def test_pairs_are_well_formed(self):
        for pair in OROGRAPHIC_PAIRS:
            assert pair.ratio_min > 1.0, f"{pair.name}: a ratio floor <=1 is not a contrast"
            assert pair.ratio_max > pair.ratio_min
            assert pair.windward.lat_n > pair.windward.lat_s
            assert pair.leeward.lat_n > pair.leeward.lat_s

    def test_windward_and_leeward_boxes_are_disjoint(self):
        shape = (256, 512)
        for pair in OROGRAPHIC_PAIRS:
            windward = region_mask(shape, pair.windward)
            leeward = region_mask(shape, pair.leeward)
            assert not np.any(windward & leeward), f"{pair.name} boxes overlap"

    def test_pairs_resolve_on_the_real_dem(self):
        from masks import get_masks
        from real_terrain_validation import load_bundled_earth_dem

        elevation = load_bundled_earth_dem(256, 512)
        _, land = get_masks(elevation, use_cache=False)
        for pair in OROGRAPHIC_PAIRS:
            for box in (pair.windward, pair.leeward):
                cells = np.count_nonzero(region_mask(land.shape, box, cell_mask=land))
                assert cells > 0, f"{box.name} contains no land at 256x512"

    def test_contrast_returns_none_when_a_box_is_empty(self):
        empty = np.zeros((64, 128), dtype=bool)
        assert orographic_contrast(
            np.ones((64, 128)), OROGRAPHIC_PAIRS[0], land_mask=empty
        ) is None
