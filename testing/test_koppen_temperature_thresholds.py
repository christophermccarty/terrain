"""Guards for the anchor-free land-temperature instrument (ACCURACY_AUDIT.md C1b).

This scores the model's coldest/warmest month against Köppen's *definitional*
bounds rather than against station anchors, which is the whole point: the metric
it complements, ``_land_seasonal_cycle_metrics``, compares an area-weighted
band mean against mid-continental station values and is wrong by ~13 K at
25-35N for that reason alone.

Like the map-skill guards next door, these pin the properties that make the
score meaningful -- correct bound assignment, correct direction reporting,
area weighting, and refusal to score cells the reference does not constrain --
rather than the score itself. Each is checked to *fail* on a planted violation
so the suite cannot pass by having stopped looking.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import koppen_reference as kr  # noqa: E402
from climate_averages import (  # noqa: E402
    KOPPEN_AF,
    KOPPEN_BWH,
    KOPPEN_CFA,
    KOPPEN_CFB,
    KOPPEN_CWA,
    KOPPEN_DFB,
    KOPPEN_DWD,
    KOPPEN_EF,
    KOPPEN_ET,
    KOPPEN_OCEAN,
)

pytestmark = pytest.mark.skipif(
    not kr.DEFAULT_REFERENCE_PATH.exists(),
    reason="bundled Köppen reference map is not present in this checkout",
)

H, W = 16, 32


def _grid(codes_1d: list[int]) -> kr.ReferenceGrid:
    """A synthetic reference grid: one class per row, ocean padding elsewhere."""
    codes = np.full((H, W), KOPPEN_OCEAN, dtype=np.int8)
    for row, code in enumerate(codes_1d):
        codes[row, :] = code
    land_fraction = np.where(codes != KOPPEN_OCEAN, 1.0, 0.0).astype(np.float32)
    return kr.ReferenceGrid(codes=codes, land_fraction=land_fraction)


def _grid_columns(codes_1d: list[int]) -> kr.ReferenceGrid:
    """Classes side by side in the *same* rows, so they carry equal area weight.

    ``_grid`` puts one class per row, which is convenient but gives each class a
    different cos(latitude) weight -- fine when asserting per-group results,
    wrong when asserting a pooled 50/50 split.
    """
    codes = np.full((H, W), KOPPEN_OCEAN, dtype=np.int8)
    span = W // len(codes_1d)
    for index, code in enumerate(codes_1d):
        codes[:, index * span:(index + 1) * span] = code
    land_fraction = np.where(codes != KOPPEN_OCEAN, 1.0, 0.0).astype(np.float32)
    return kr.ReferenceGrid(codes=codes, land_fraction=land_fraction)


def _monthly(tcold_c: float, twarm_c: float) -> np.ndarray:
    """A 12-month field whose min/max are exactly the requested values."""
    field = np.full((12, H, W), (tcold_c + twarm_c) / 2.0, dtype=np.float64)
    field[0] = tcold_c
    field[6] = twarm_c
    return field + 273.15


def test_group_bounds_are_the_koppen_definitions():
    """A/C/D coldest-month bounds must be 18 C and the map's -3 C variant line."""
    reference = _grid([KOPPEN_AF, KOPPEN_CFB, KOPPEN_DFB])
    # Tcold = +20 C: correct for Af, too warm for both Cfb (< 18) and Dfb (< -3).
    report = kr.score_temperature_thresholds(_monthly(20.0, 30.0), reference=reference)
    by_group = report["coldest_month"]["by_reference_group"]
    assert by_group["A"]["accuracy"] == pytest.approx(1.0)
    assert by_group["C"]["too_warm_fraction"] == pytest.approx(1.0)
    assert by_group["D"]["too_warm_fraction"] == pytest.approx(1.0)

    # Tcold = 0 C: too cold for Af, correct for Cfb, still too warm for Dfb.
    report = kr.score_temperature_thresholds(_monthly(0.0, 30.0), reference=reference)
    by_group = report["coldest_month"]["by_reference_group"]
    assert by_group["A"]["too_cold_fraction"] == pytest.approx(1.0)
    assert by_group["C"]["accuracy"] == pytest.approx(1.0)
    assert by_group["D"]["too_warm_fraction"] == pytest.approx(1.0)

    # Tcold = -10 C: too cold for both Af and Cfb, correct for Dfb.
    report = kr.score_temperature_thresholds(_monthly(-10.0, 30.0), reference=reference)
    by_group = report["coldest_month"]["by_reference_group"]
    assert by_group["A"]["too_cold_fraction"] == pytest.approx(1.0)
    assert by_group["C"]["too_cold_fraction"] == pytest.approx(1.0)
    assert by_group["D"]["accuracy"] == pytest.approx(1.0)


def test_boundary_is_half_open_at_minus_three():
    """The reference is the -3 C variant, so exactly -3 C is C, not D."""
    reference = _grid([KOPPEN_CFB, KOPPEN_DFB])
    report = kr.score_temperature_thresholds(_monthly(-3.0, 20.0), reference=reference)
    by_group = report["coldest_month"]["by_reference_group"]
    assert by_group["C"]["accuracy"] == pytest.approx(1.0)
    assert by_group["D"]["too_warm_fraction"] == pytest.approx(1.0)


def test_arid_cells_are_never_scored_for_coldest_month():
    """B is defined by aridity and constrains no temperature -- scoring it is a bug."""
    reference = _grid([KOPPEN_BWH, KOPPEN_BWH, KOPPEN_BWH])
    report = kr.score_temperature_thresholds(_monthly(-40.0, 60.0), reference=reference)
    # Absurd temperatures, but B carries no bound, so there is nothing to score.
    assert "coldest_month" not in report
    assert report.get("warmest_month") is None or "warmest_month" not in report


def test_polar_cells_are_scored_on_warmest_not_coldest():
    """E is a warmest-month definition; its coldest month is unconstrained."""
    reference = _grid_columns([KOPPEN_EF, KOPPEN_ET])
    report = kr.score_temperature_thresholds(_monthly(-60.0, 5.0), reference=reference)
    assert "coldest_month" not in report
    by_group = report["warmest_month"]["by_reference_group"]
    # Twarm = +5 C: too warm for EF (< 0) and correct for ET ([0, 10)). Equal
    # areas, so the pooled E score must land exactly halfway.
    assert by_group["E"]["accuracy"] == pytest.approx(0.5, abs=0.01)
    assert by_group["E"]["too_warm_fraction"] == pytest.approx(0.5, abs=0.01)


def test_dwd_uses_the_tighter_minus_38_bound():
    """The `d` sub-letter requires a coldest month below -38 C, not merely -3."""
    reference = _grid([KOPPEN_DWD])
    warm = kr.score_temperature_thresholds(_monthly(-20.0, 20.0), reference=reference)
    assert warm["coldest_month"]["too_warm_fraction"] == pytest.approx(1.0)
    cold = kr.score_temperature_thresholds(_monthly(-45.0, 20.0), reference=reference)
    assert cold["coldest_month"]["accuracy"] == pytest.approx(1.0)


def test_thermal_sub_letter_sets_the_warmest_month_bound():
    """`a` requires warmest >= 22 C; `b` requires < 22 C."""
    reference = _grid_columns([KOPPEN_CFA, KOPPEN_CFB])
    report = kr.score_temperature_thresholds(_monthly(0.0, 25.0), reference=reference)
    assert report["warmest_month"]["accuracy"] == pytest.approx(0.5, abs=0.01)
    # Cfa is satisfied at 25 C and Cfb is violated; flip the temperature and the
    # verdicts must swap rather than both moving the same way.
    flipped = kr.score_temperature_thresholds(_monthly(0.0, 18.0), reference=reference)
    assert flipped["warmest_month"]["accuracy"] == pytest.approx(0.5, abs=0.01)
    assert flipped["warmest_month"]["too_cold_fraction"] == pytest.approx(0.5, abs=0.01)
    assert report["warmest_month"]["too_warm_fraction"] == pytest.approx(0.5, abs=0.01)


def test_cwa_is_excluded_from_the_warmest_month_score():
    """Cwa/Cwb/Cwc fold together and disagree about 22 C -- scoring it scores the fold."""
    reference = _grid([KOPPEN_CWA])
    report = kr.score_temperature_thresholds(_monthly(5.0, 30.0), reference=reference)
    assert "warmest_month" not in report
    # Its *group* bound still applies: Cwa is C, so the coldest month is scored.
    assert report["coldest_month"]["accuracy"] == pytest.approx(1.0)


def test_scoring_is_area_weighted():
    """A tropical row must outweigh a polar row of the same cell count."""
    codes = np.full((H, W), KOPPEN_OCEAN, dtype=np.int8)
    equator_row = H // 2
    codes[equator_row, :] = KOPPEN_CFB      # will be scored correct
    codes[0, :] = KOPPEN_CFB                # near the pole, will be scored wrong
    reference = kr.ReferenceGrid(
        codes=codes,
        land_fraction=np.where(codes != KOPPEN_OCEAN, 1.0, 0.0).astype(np.float32),
    )
    monthly = np.full((12, H, W), 10.0 + 273.15)
    monthly[0, equator_row, :] = 5.0 + 273.15    # Tcold +5 C -> correct for C
    monthly[0, 0, :] = -40.0 + 273.15            # Tcold -40 C -> too cold for C
    report = kr.score_temperature_thresholds(monthly, reference=reference)
    accuracy = report["coldest_month"]["accuracy"]
    # Equal cell counts, so an unweighted score would read exactly 0.5. The
    # equatorial row carries ~cos(0)=1 against the polar row's ~cos(84)=0.1.
    assert accuracy > 0.85, f"area weighting appears to be missing (got {accuracy})"


def test_land_mask_restricts_scoring():
    """Cells the model calls ocean must not be scored against reference land."""
    reference = _grid([KOPPEN_DFB, KOPPEN_DFB])
    monthly = _monthly(-10.0, 20.0)
    land = np.zeros((H, W), dtype=bool)
    land[0, :] = True
    full = kr.score_temperature_thresholds(monthly, reference=reference)
    masked = kr.score_temperature_thresholds(monthly, land_mask=land, reference=reference)
    assert masked["coldest_month"]["scored_cells"] == W
    assert full["coldest_month"]["scored_cells"] == 2 * W


def test_direction_split_sums_to_the_complement_of_accuracy():
    """too_warm + too_cold + accuracy == 1 exactly, or a cell is being lost."""
    reference = _grid([KOPPEN_AF, KOPPEN_CFB, KOPPEN_DFB, KOPPEN_ET])
    for tcold, twarm in ((-50.0, -5.0), (0.0, 15.0), (25.0, 40.0)):
        report = kr.score_temperature_thresholds(
            _monthly(tcold, twarm), reference=reference
        )
        for entry in report.values():
            if not isinstance(entry, dict) or "accuracy" not in entry:
                continue
            total = (
                entry["accuracy"]
                + entry["too_warm_fraction"]
                + entry["too_cold_fraction"]
            )
            assert total == pytest.approx(1.0)


def test_rejects_a_field_that_is_not_twelve_months():
    reference = _grid([KOPPEN_CFB])
    with pytest.raises(ValueError):
        kr.score_temperature_thresholds(
            np.zeros((6, H, W)) + 273.15, reference=reference
        )


def test_rejects_a_reference_of_the_wrong_shape():
    reference = _grid([KOPPEN_CFB])
    with pytest.raises(ValueError):
        kr.score_temperature_thresholds(
            np.zeros((12, H * 2, W)) + 273.15, reference=reference
        )


def test_real_reference_scores_a_substantial_fraction_of_land():
    """Sanity: the bounds must actually bind on real geography, not a fringe."""
    grid = kr.load_reference_grid(64, 128)
    monthly = np.full((12, 64, 128), 288.0)
    monthly[0] -= 20.0
    monthly[6] += 20.0
    report = kr.score_temperature_thresholds(monthly, reference=grid)
    land_cells = int(np.count_nonzero(grid.codes != KOPPEN_OCEAN))
    scored = report["coldest_month"]["scored_cells"]
    assert scored > 0.3 * land_cells, (
        f"only {scored} of {land_cells} land cells carry a coldest-month bound"
    )
