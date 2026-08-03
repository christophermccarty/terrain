"""Guards for the gridded Köppen map-skill instrument (ACCURACY_AUDIT.md H10).

This is validation *infrastructure*, so its own failure modes are silent by
nature: a mis-decoded palette, a half-cell regridding offset, or a missing
cos(latitude) weight would all still produce plausible-looking scores. These
tests pin the properties that make the score meaningful rather than the score
itself.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import koppen_reference as kr  # noqa: E402
from climate_averages import (  # noqa: E402
    KOPPEN_AF,
    KOPPEN_BWH,
    KOPPEN_CFB,
    KOPPEN_DFA,
    KOPPEN_DFB,
    KOPPEN_EF,
    KOPPEN_OCEAN,
)

pytestmark = pytest.mark.skipif(
    not kr.DEFAULT_REFERENCE_PATH.exists(),
    reason="bundled Köppen reference map is not present in this checkout",
)


def _code_at(grid: kr.ReferenceGrid, lat: float, lon: float) -> int:
    height, width = grid.shape
    row = int((90.0 - lat) / 180.0 * height)
    col = int((lon + 180.0) / 360.0 * width)
    return int(grid.codes[min(row, height - 1), min(col, width - 1)])


class TestReferenceDecoding:
    def test_every_source_pixel_matches_a_known_legend_colour(self):
        """No silent fallback: an unmapped colour must raise, not become ocean."""
        codes = kr._decode_reference_pixels(kr.DEFAULT_REFERENCE_PATH)
        assert codes.shape == (1800, 3600)
        assert codes.min() >= 0 and codes.max() < kr.N_KOPPEN_CODES

    def test_unknown_colour_raises_rather_than_misclassifying(self, tmp_path):
        bogus = np.zeros((8, 16, 3), dtype=np.uint8)
        bogus[...] = (123, 45, 67)  # not in the legend
        path = tmp_path / "bogus.png"
        Image.fromarray(bogus).save(path)
        with pytest.raises(ValueError, match="unrecognized colours"):
            kr._decode_reference_pixels(path)

    def test_non_equirectangular_source_rejected(self, tmp_path):
        path = tmp_path / "square.png"
        Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8)).save(path)
        with pytest.raises(ValueError, match="2:1 equirectangular"):
            kr._decode_reference_pixels(path)


class TestRegridding:
    @pytest.mark.parametrize("height,width", [(64, 128), (90, 180), (180, 360)])
    def test_known_geography_lands_in_the_right_cell(self, height, width):
        """Catches half-cell offsets and lat/lon flips, which a score cannot."""
        grid = kr.load_reference_grid(height, width)
        assert _code_at(grid, -3.0, -60.0) == KOPPEN_AF      # central Amazon
        assert _code_at(grid, 25.0, 10.0) == KOPPEN_BWH      # central Sahara
        assert _code_at(grid, -80.0, 0.0) == KOPPEN_EF       # East Antarctica
        assert _code_at(grid, 41.9, -87.6) == KOPPEN_DFA     # Chicago
        assert _code_at(grid, 55.75, 37.6) == KOPPEN_DFB     # Moscow

    def test_open_ocean_carries_no_land_class(self):
        grid = kr.load_reference_grid(180, 360)
        assert _code_at(grid, 30.0, -160.0) == KOPPEN_OCEAN  # mid North Pacific
        row = int((90.0 - 30.0) / 180.0 * 180)
        col = int((-160.0 + 180.0) / 360.0 * 360)
        assert grid.land_fraction[row, col] == pytest.approx(0.0)

    def test_land_fraction_is_a_fraction_and_tracks_codes(self):
        grid = kr.load_reference_grid(64, 128)
        assert grid.land_fraction.min() >= 0.0
        assert grid.land_fraction.max() <= 1.0
        # A cell has a land class if and only if it contains land pixels.
        assert np.array_equal(grid.codes != KOPPEN_OCEAN, grid.land_fraction > 0.0)

    def test_regridding_conserves_total_land_area(self):
        """Every source pixel must land in exactly one target cell."""
        source = kr._decode_reference_pixels(kr.DEFAULT_REFERENCE_PATH)
        source_land = float(np.count_nonzero(source))
        grid = kr.load_reference_grid(180, 360)
        # land_fraction * pixels-per-cell, summed, must recover the source count.
        pixels_per_cell = (1800 / 180) * (3600 / 360)
        assert grid.land_fraction.sum() * pixels_per_cell == pytest.approx(
            source_land, rel=1e-9
        )

    def test_cached_grid_cannot_be_mutated_by_a_caller(self):
        first = kr.load_reference_grid(64, 128)
        first.codes[:] = 0
        second = kr.load_reference_grid(64, 128)
        assert np.any(second.codes != 0)

    def test_rejects_non_two_to_one_target(self):
        with pytest.raises(ValueError, match="2:1 equirectangular"):
            kr.load_reference_grid(64, 100)


class TestEarthShares:
    def test_reference_reproduces_earth_climate_shares(self):
        """A swapped or corrupted reference map would move these badly."""
        shares = kr.earth_group_shares()
        assert sum(shares.values()) == pytest.approx(100.0)
        assert 17.0 < shares["A"] < 23.0   # tropical
        assert 25.0 < shares["B"] < 32.0   # arid
        assert 12.0 < shares["C"] < 18.0   # temperate
        assert 19.0 < shares["D"] < 26.0   # continental
        assert 11.0 < shares["E"] < 18.0   # polar

    def test_shares_are_stable_across_resolution(self):
        coarse = kr.earth_group_shares(64, 128)
        fine = kr.earth_group_shares(360, 720)
        for name in coarse:
            assert abs(coarse[name] - fine[name]) < 2.0, f"{name} share is resolution-dependent"


class TestScoring:
    def test_perfect_agreement_scores_one(self):
        grid = kr.load_reference_grid(64, 128)
        land = grid.codes != KOPPEN_OCEAN
        report = kr.score_koppen_map(grid.codes.astype(np.int32), land_mask=land)
        assert report["group"]["accuracy"] == pytest.approx(1.0)
        assert report["group"]["kappa"] == pytest.approx(1.0)
        assert report["class"]["accuracy"] == pytest.approx(1.0)
        assert report["group"]["share_mae_pp"] == pytest.approx(0.0)

    def test_kappa_punishes_a_degenerate_single_class_model(self):
        """The reason kappa is reported alongside accuracy.

        Painting all land with the single most common class earns a respectable
        raw accuracy purely from base rates; kappa must see through that, or the
        instrument would reward a model that had stopped resolving climate.
        """
        grid = kr.load_reference_grid(64, 128)
        land = grid.codes != KOPPEN_OCEAN
        degenerate = np.where(land, KOPPEN_BWH, KOPPEN_OCEAN).astype(np.int32)
        report = kr.score_koppen_map(degenerate, land_mask=land)
        assert report["group"]["accuracy"] > 0.2      # base rate alone
        assert abs(report["group"]["kappa"]) < 1e-9   # but zero real skill

    def test_scoring_is_area_weighted_not_cell_counted(self):
        """Process note 8's bug class, applied to this metric.

        The same number of *cells* corrupted at high latitude must cost less
        than at the equator. Without cos(lat) weighting these two scores would
        be identical.
        """
        grid = kr.load_reference_grid(64, 128)
        land = grid.codes != KOPPEN_OCEAN
        rows_polar = np.zeros_like(land)
        rows_polar[2:6, :] = True                    # ~80-70 N
        rows_tropic = np.zeros_like(land)
        rows_tropic[30:34, :] = True                 # ~0-10 N
        polar_cells = int(np.count_nonzero(land & rows_polar))
        tropic_cells = int(np.count_nonzero(land & rows_tropic))
        assert polar_cells > 0 and tropic_cells > 0

        def corrupt(selector):
            codes = grid.codes.astype(np.int32).copy()
            wrong = np.where(codes == KOPPEN_AF, KOPPEN_BWH, KOPPEN_AF)
            codes[land & selector] = wrong[land & selector]
            return kr.score_koppen_map(codes, land_mask=land)["group"]["accuracy"]

        # Normalize per corrupted cell so the comparison isolates the weighting.
        polar_cost = (1.0 - corrupt(rows_polar)) / polar_cells
        tropic_cost = (1.0 - corrupt(rows_tropic)) / tropic_cells
        assert tropic_cost > 2.0 * polar_cost

    def test_zero_threshold_admits_no_ocean_cells(self):
        """Regression: land_fraction >= 0.0 is true for pure ocean too."""
        grid = kr.load_reference_grid(64, 128)
        land = grid.codes != KOPPEN_OCEAN
        report = kr.score_koppen_map(
            grid.codes.astype(np.int32), land_mask=land, min_reference_land_fraction=0.0
        )
        assert report["group"]["accuracy"] == pytest.approx(1.0)

    def test_grid_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="does not match"):
            kr.score_koppen_map(
                np.zeros((32, 64), dtype=np.int32),
                reference=kr.load_reference_grid(64, 128),
            )

    def test_disjoint_land_masks_report_cleanly(self):
        codes = np.zeros((64, 128), dtype=np.int32)
        report = kr.score_koppen_map(codes, land_mask=np.zeros((64, 128), dtype=bool))
        assert "error" in report and "group" not in report


class TestVocabularyFolding:
    def test_every_reference_label_folds_to_a_real_model_code(self):
        for label in kr.REFERENCE_PALETTE.values():
            assert label in kr.REFERENCE_TO_MODEL_CODE, f"{label} has no model fold"
            code = kr.REFERENCE_TO_MODEL_CODE[label]
            assert 0 <= code < kr.N_KOPPEN_CODES

    def test_folding_preserves_the_group_letter(self):
        """Group-level scoring is only folding-independent if this holds."""
        for label, code in kr.REFERENCE_TO_MODEL_CODE.items():
            if label == "Ocean":
                continue
            assert kr.short_class_name(code)[0] == label[0], (
                f"{label} folds to {kr.short_class_name(code)}, changing its group"
            )

    def test_group_mapping_covers_the_whole_vocabulary(self):
        codes = np.arange(kr.N_KOPPEN_CODES)
        groups = kr.koppen_group(codes)
        assert groups[KOPPEN_OCEAN] == kr.GROUP_NONE
        assert np.all(groups[1:] > 0)
        assert kr.koppen_group(np.array([KOPPEN_CFB]))[0] == kr.GROUP_C

    def test_out_of_range_codes_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            kr.koppen_group(np.array([kr.N_KOPPEN_CODES]))
