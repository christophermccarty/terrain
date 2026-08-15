"""Deterministic real-terrain climate validation and regression scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace as dataclasses_replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from atmosphere import generate_precipitation
from balanced_dynamics import pressure_level_geopotential
from circulation_diagnostics import circulation_scorecard, seasonal_jet_scorecard
from masks import get_masks
from temperature import elevation_to_alt_km
from two_layer_overturning_diagnostics import diagnose_two_layer_overturning_state
from monthly_climatology import (
    MonthlyClimatology,
    load_monthly_climatology,
    regrid_monthly_climatology,
    score_monthly_climatology,
)
from planet_params import EARTH, PlanetParams
from regional_moisture_budget import (
    regional_moisture_budget_snapshot,
    season_for_day,
    seasonal_regional_moisture_budget,
    time_average_regional_moisture_budget,
)
from regional_validation import (
    EARTH_PRECIP_REGIONS,
    OROGRAPHIC_PAIRS,
    orographic_contrast,
    precipitation_by_region_mm_year,
    region_mean,
    region_mask,
    target_error_fraction,
)
from simulate import (
    PlanetState,
    TimeScaleMode,
    clear_simulation_caches,
    create_initial_state,
    load_state,
    simulate_step,
)
from time_policy import cycle_days, substeps_for_mode


VALIDATION_SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parent
DEFAULT_DEM_PATH = ROOT / "images" / "16_bit_dem_small_512.tif"
DEFAULT_BASELINE_PATH = ROOT / "testing" / "fixtures" / "real_terrain_validation_baseline.json"
EARTH_DEM_OCEAN_THRESHOLD = 8070.0

# Published-climatology summary anchors already used by the test suite. These
# are zonal-band checks, not a substitute for a licensed gridded ERA5/CRU pack.
EARTH_ZONAL_REFERENCE: dict[str, dict[str, float]] = {
    "0-10N": {"lat_s": 0.0, "lat_n": 10.0, "t_c": 26.5, "p_mm_yr": 2000.0},
    "10-20N": {"lat_s": 10.0, "lat_n": 20.0, "t_c": 25.5, "p_mm_yr": 1800.0},
    "40-50N": {"lat_s": 40.0, "lat_n": 50.0, "t_c": 9.0, "p_mm_yr": 600.0},
    "50-60N": {"lat_s": 50.0, "lat_n": 60.0, "t_c": 4.0, "p_mm_yr": 500.0},
    "0-10S": {"lat_s": -10.0, "lat_n": 0.0, "t_c": 25.5, "p_mm_yr": 1500.0},
    "40-50S": {"lat_s": -50.0, "lat_n": -40.0, "t_c": 12.0, "p_mm_yr": 800.0},
}


@dataclass(frozen=True)
class RealTerrainValidationConfig:
    """Deterministic real-DEM validation run.

    `spinup_years` is short on purpose (the whole benchmark is ~14 s, which is
    what makes it usable for A/B direction-finding), but note what that does and
    does not buy. The Köppen monthly bins have their own spin-up timescale,
    separate from temperature/soil/ocean: each of the 12 bins is visited once per
    simulated year in MONTHLY mode, so `spinup_years=1.0` gives every bin exactly
    **one** real sample before evaluation begins.

    That is sufficient only because `climate_averages.update_monthly_statistics`
    now *discards* its flat spin-up seed on a bin's first real sample. While it
    blended the seed instead, this config left ~13.5% of a zero-amplitude annual
    cycle in the bins at evaluation time, which biased every seasonal-extremum
    statistic (Köppen's "driest month >= 60 mm" above all) toward "less seasonal
    than reality" -- and three separate calibration sessions fitted parameters to
    that decaying transient. See ACCURACY_AUDIT.md process note 24 and A2.

    **So: before calibrating anything against a statistic taken over the seasonal
    cycle, re-run it at a longer `spinup_years` and confirm the number is
    stationary.** Process note 14 says to confirm at a second resolution; this is
    the same rule on the time axis, and it is cheaper (one long run, not a sweep).
    """

    height: int = 64
    width: int = 128
    spinup_years: float = 1.0
    evaluation_years: float = 1.0
    time_scale: str = "MONTHLY"
    block_size: int = 4
    wind_block_size: int = 4
    precip_block_size: int = 1
    start_day: float = 80.0

    def validate(self) -> None:
        if self.height < 16 or self.width < 32:
            raise ValueError("validation grid must be at least 16x32")
        if self.width != 2 * self.height:
            raise ValueError("validation grid must use a 2:1 equirectangular aspect ratio")
        if self.spinup_years < 0.0 or self.evaluation_years <= 0.0:
            raise ValueError("spinup_years must be non-negative and evaluation_years positive")
        if self.time_scale not in TimeScaleMode.__members__:
            raise ValueError(f"unknown time scale {self.time_scale!r}")
        if self.block_size < 1 or self.wind_block_size < 1:
            raise ValueError("block sizes must be positive")
        if self.precip_block_size not in (1, 2):
            raise ValueError("precip_block_size must be 1 or 2")


def load_bundled_earth_dem(
    height: int = 512,
    width: int = 1024,
    *,
    path: str | Path = DEFAULT_DEM_PATH,
) -> np.ndarray:
    """Load and deterministically resize the bundled Earth DEM to normalized elevation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bundled Earth DEM not found: {path}")
    raw = np.asarray(Image.open(path), dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Earth DEM must be grayscale, got shape {raw.shape}")
    source_max = float(np.max(raw))
    if source_max <= EARTH_DEM_OCEAN_THRESHOLD:
        raise ValueError("Earth DEM contains no elevations above the ocean threshold")

    if raw.shape != (height, width):
        src_h, src_w = raw.shape
        if src_h % height == 0 and src_w % width == 0:
            fy, fx = src_h // height, src_w // width
            raw = raw.reshape(height, fy, width, fx).mean(axis=(1, 3))
        else:
            raw = np.asarray(
                Image.fromarray(raw, mode="F").resize(
                    (width, height), resample=Image.Resampling.BOX
                ),
                dtype=np.float32,
            )

    elevation = np.maximum(
        0.0,
        (raw - EARTH_DEM_OCEAN_THRESHOLD)
        / (source_max - EARTH_DEM_OCEAN_THRESHOLD),
    )
    return np.clip(elevation, 0.0, 1.0).astype(np.float32)


def _area_weighted_mean(field: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(field, dtype=np.float64)
    H = values.shape[0]
    lat = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * np.pi
    weights = np.broadcast_to(np.cos(lat)[:, None], values.shape)
    selected = np.isfinite(values)
    if mask is not None:
        selected &= np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return float("nan")
    return float(np.sum(values[selected] * weights[selected]) / np.sum(weights[selected]))


def _zonal_metrics(
    temperature_k: np.ndarray,
    precipitation_mm_day: np.ndarray,
    *,
    days_per_year: float,
) -> dict[str, dict[str, float]]:
    H = temperature_k.shape[0]
    lat = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * 180.0
    metrics: dict[str, dict[str, float]] = {}
    for name, reference in EARTH_ZONAL_REFERENCE.items():
        rows = (lat >= reference["lat_s"]) & (lat < reference["lat_n"])
        if not np.any(rows):
            continue
        t_c = float(np.mean(temperature_k[rows], dtype=np.float64) - 273.15)
        p_mm_yr = float(
            np.mean(precipitation_mm_day[rows], dtype=np.float64) * days_per_year
        )
        metrics[name] = {
            "temperature_c": t_c,
            "precip_mm_year": p_mm_yr,
            "temperature_bias_c": t_c - reference["t_c"],
            "precip_ratio": p_mm_yr / max(reference["p_mm_yr"], 1.0),
        }
    return metrics


# Continental-land seasonal-cycle anchors, by latitude band. Warmest/coldest
# month and annual mean in C, from published climatology for mid-continental
# stations in each band (the same standing as EARTH_ZONAL_REFERENCE's numbers:
# summary anchors, not a gridded product). "Squareness" is months a cell spends
# above its own annual mean; a sinusoid gives exactly 6.00 and real land runs
# only slightly above that.
#
# **These level anchors do not match the population the metric measures, and
# must not be optimised against** (found 2026-08-04, audit process note 8's
# fourth instance). The anchors describe *mid-continental stations*;
# `_land_seasonal_cycle_metrics` below computes an area-weighted mean over
# **all** land in the band. Those diverge badly in exactly the bands C1b works
# on: by the bundled Köppen reference's own accounting, 25-35N land is 54% arid
# subtropics (BWh alone is 42%) plus 8% Tibetan plateau, and 35-45N is 34% cold
# desert -- neither is "mid-continental", and both have far milder winters than
# the -2/-4 C anchors below imply. Optimising `cycle_error_score` therefore
# drives the subtropics cold, which is the most likely explanation for C1b's
# recurring finding that knobs improving this score degrade H10 accuracy.
#
# Measured, on the same 128x256 run: this metric reports the 25-35N coldest
# month as +12.85 K too warm, while `koppen_temperature_thresholds` -- which
# needs no anchors, only the reference's own definitional bounds -- finds
# **100%** of that band's reference-C land on the correct side of both the -3 C
# and +18 C boundaries. The apparent 13 K bias is the anchor, not the model.
#
# `squareness_months` and `plateau_months` are unaffected: both are
# dimensionless properties of a cell's own cycle compared against a sinusoid,
# so they are valid for any population. `cycle_error_score` is built from those
# alone for that reason; the level biases are still reported, as diagnosis, but
# are not scored. Use `koppen_temperature_thresholds` for level questions.
EARTH_LAND_CYCLE_REFERENCE: dict[str, dict[str, float]] = {
    "25-35N": {"lat_s": 25.0, "lat_n": 35.0, "warmest_c": 26.0, "coldest_c": -2.0,
               "mean_c": 12.0, "amplitude_k": 28.0},
    "35-45N": {"lat_s": 35.0, "lat_n": 45.0, "warmest_c": 24.0, "coldest_c": -4.0,
               "mean_c": 10.0, "amplitude_k": 28.0},
    "45-55N": {"lat_s": 45.0, "lat_n": 55.0, "warmest_c": 21.0, "coldest_c": -8.0,
               "mean_c": 7.0, "amplitude_k": 29.0},
    "25-35S": {"lat_s": -35.0, "lat_n": -25.0, "warmest_c": 24.0, "coldest_c": 8.0,
               "mean_c": 16.0, "amplitude_k": 16.0},
}
EARTH_LAND_SQUARENESS = 6.2


def _land_seasonal_cycle_metrics(
    state: PlanetState, land_mask: np.ndarray
) -> dict[str, Any]:
    """Shape and level of the land annual temperature cycle (audit C1b).

    Built 2026-08-03 because **nothing in the platform measured this**. C1b has
    been worked across several sessions on numbers produced by throwaway offline
    probes, so no result was regression-gated and each session re-derived the
    same figures. The three quantities that matter are all here:

    - ``squareness_months``: months a cell spends above its own annual mean. A
      sinusoid gives 6.00; this model's land runs 7.0-7.9 (ocean 6.3-6.7), a
      broad warm plateau joined to a narrow deep winter trough.
    - ``warmest_c`` / ``coldest_c`` / ``mean_c``: the *level*, which turns out to
      be the binding error rather than the shape. Measured offline, the forcing
      entering ``_land_cap_1d`` has an annual mean ~21 K above Earth's at 41 deg,
      and the clamp is the only thing hiding it -- so a metric that watched shape
      alone would score a mean-bias fix as a regression.
    - ``plateau_months``: months within 1 K of the cell's own warmest month. A
      sinusoid gives ~1.4; a hard clamp writes a flat top and drives this toward
      7. This is the clamp's fingerprint *in the output*.

    Restricted to land, area-weighted, and to bands where a meaningful number of
    land cells exist -- an ocean cell's cycle is damped by ``ocean_lag_days`` and
    is not the phenomenon under test.

    **A rejected first version, recorded so it is not rebuilt** (process note 11):
    the obvious metric is the fraction of (cell, month) pairs sitting on
    ``_land_cap_1d``. It reads **0.00 in three of four bands** and is useless,
    because the clamp is applied to the ``T_base_land`` *forcing* while
    ``monthly_temp`` is the output of a subsequent relaxation toward it -- the
    output approaches the ceiling from below and essentially never touches it.
    C1b's "binds on 55.7% of (month, row) pairs" is a forcing-stage number and
    cannot be reproduced from any saved state. ``plateau_months`` measures the
    same defect where it is actually observable.
    """
    monthly = state.monthly_temp
    if monthly is None or np.asarray(monthly).ndim != 3:
        return {}
    monthly_c = np.asarray(monthly, dtype=np.float64) - 273.15
    if monthly_c.shape[0] != 12 or not np.any(land_mask):
        return {}
    H = land_mask.shape[0]
    lat = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * 180.0
    # The shipped ceiling, recomputed here rather than imported: this metric must
    # keep reporting a meaningful cap-bound fraction if the clamp is reshaped or
    # retired, and importing the live array would make it silently follow.
    weights = np.cos(np.radians(lat))

    annual_mean = monthly_c.mean(axis=0)
    above = (monthly_c > annual_mean[None, :, :]).sum(axis=0).astype(np.float64)
    warmest = monthly_c.max(axis=0)
    coldest = monthly_c.min(axis=0)
    plateau = (monthly_c >= warmest[None, :, :] - 1.0).sum(axis=0).astype(np.float64)
    # What a *pure sinusoid of this cell's own amplitude* would score on the same
    # 1 K / 12-sample plateau test. A fixed constant cannot serve here: the test
    # is in absolute kelvin, so a low-amplitude cycle spends more months within
    # 1 K of its peak for reasons that have nothing to do with a clamp (at
    # amplitude 28 K a sinusoid scores ~1.4 months, at 16 K it scores ~1.9).
    # Comparing each cell against its own amplitude's sinusoid removes that
    # dependence, so `plateau_excess_months` isolates the flat top itself and
    # stays valid across the differing amplitudes of different bands.
    # Closed form rather than a sampled model sinusoid: a sampled one has to
    # assume where the peak falls relative to a month centre, and that choice
    # alone moves the count by a whole month. The continuous fraction of a cycle
    # spending within 1 K of its peak is (12/pi) * arccos(1 - 1/A) months, for
    # half-amplitude A. Residual disagreement with the model's own 12-sample
    # count is the sampling quantisation itself, ~0.5 months, which is well
    # inside the 3-5 month signal a real clamp produces.
    half_amplitude = np.maximum((warmest - coldest) / 2.0, 1e-9)
    sinusoid_plateau = (12.0 / np.pi) * np.arccos(
        np.clip(1.0 - 1.0 / half_amplitude, -1.0, 1.0)
    )

    bands: dict[str, Any] = {}
    errors: list[float] = []
    for name, reference in EARTH_LAND_CYCLE_REFERENCE.items():
        rows = (lat >= reference["lat_s"]) & (lat < reference["lat_n"])
        cells = np.asarray(land_mask, dtype=bool) & rows[:, None]
        if int(np.count_nonzero(cells)) < 4:
            continue
        cell_weights = np.broadcast_to(weights[:, None], land_mask.shape)[cells]
        total = float(cell_weights.sum())

        def mean_of(field: np.ndarray) -> float:
            return float((field[cells] * cell_weights).sum() / total)

        warmest_c = mean_of(warmest)
        coldest_c = mean_of(coldest)
        mean_c = mean_of(annual_mean)
        bands[name] = {
            "squareness_months": mean_of(above),
            "warmest_c": warmest_c,
            "coldest_c": coldest_c,
            "mean_c": mean_c,
            "amplitude_k": warmest_c - coldest_c,
            "warmest_bias_c": warmest_c - reference["warmest_c"],
            "coldest_bias_c": coldest_c - reference["coldest_c"],
            "mean_bias_c": mean_c - reference["mean_c"],
            "plateau_months": mean_of(plateau),
            "plateau_excess_months": mean_of(plateau - sinusoid_plateau),
        }
        # Shape only. The `warmest_bias_c`/`coldest_bias_c` terms this score used
        # to include were scored against station anchors for a population this
        # metric does not measure -- see EARTH_LAND_CYCLE_REFERENCE's note. They
        # remain in `bands` for diagnosis; the anchor-free level authority is
        # `koppen_temperature_thresholds`.
        errors.extend(
            [
                abs(bands[name]["squareness_months"] - EARTH_LAND_SQUARENESS) * 5.0,
                abs(bands[name]["plateau_excess_months"]) * 5.0,
            ]
        )
    if not bands:
        return {}
    return {
        "bands": bands,
        # One scalar so this can be regression-gated and swept. Both terms are in
        # months and are scaled by 5 so that a month of shape error registers on
        # the same footing as 5 K would have -- keeping the score's magnitude
        # comparable to the pre-2026-08-04 version it replaces.
        "cycle_error_score": float(np.mean(errors)),
    }


def _koppen_land_percentages(state: PlanetState, land_mask: np.ndarray) -> dict[str, float]:
    """Köppen group shares of land, weighted by true cell AREA (cos-latitude).

    Area weighting added 2026-08-02 (ACCURACY_AUDIT.md A2). These were previously
    plain cell counts, but the grid is equirectangular: a cell at 85 deg covers
    ~cos(85 deg) ~= 8.7% of the area of an equatorial one, so counting cells
    equally massively over-weights the poles and under-weights the tropics.
    Measured on the tracked 64x128 benchmark, the distortion was severe -- polar
    read 38.3% of land (Earth ~16.6%) purely as a counting artifact, and because
    Köppen shares are a closed budget that inflation was subtracted from every
    other group. Correcting it moves the mean absolute error against Earth's real
    group shares from 8.7pp to 2.2pp without any physics change:

        group      cell-count    area-weighted    Earth
        A (trop)        12.7%            22.0%    19.0%
        B (arid)        18.3%            28.9%    26.4%
        C               9.0%            12.3%    13.4%
        D              21.7%            21.1%    24.6%
        E (polar)      38.3%            15.6%    16.6%

    Several sessions of tropical-biome work were calibrated against the
    unweighted numbers and were therefore chasing an inflated gap -- see A2.
    """
    if state.koppen_type is None or not np.any(land_mask):
        return {}
    from climate_averages import KOPPEN_NAMES

    codes = np.asarray(state.koppen_type)[land_mask]
    H, W = np.asarray(state.koppen_type).shape
    lat_rad = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * np.pi
    weights = np.repeat(np.cos(lat_rad)[:, None], W, axis=1)[land_mask]
    total = float(weights.sum())
    if total <= 0.0:
        return {}

    names = np.array([KOPPEN_NAMES.get(int(value), "") for value in codes])

    def _share(predicate) -> float:
        mask = np.array([predicate(name) for name in names], dtype=bool)
        return 100.0 * float(weights[mask].sum()) / total

    return {
        "arid": _share(lambda n: n.startswith(("BW", "BS"))),
        "humid_temperate_continental": _share(
            lambda n: n[:2] in ("Cf", "Cs", "Cw", "Df", "Dw")
        ),
        "polar": _share(lambda n: n.startswith(("ET", "EF"))),
        "tropical": _share(lambda n: n.startswith(("Af", "Am", "Aw"))),
    }


def _orographic_contrast_metrics(
    mean_precipitation_mm_day: np.ndarray, land_mask: np.ndarray
) -> dict[str, Any]:
    """Annual-mean windward:leeward ratio for each `OROGRAPHIC_PAIRS` range.

    `scripts/check_orographic_contrast.py` reports the same ratios but from a
    *single* precipitation call on one saved state, which pins every range to
    whatever the wind happened to be doing at that instant -- on
    `saves/earth.pkl` the Scandinavian flow is easterly, so that pair's
    "windward" box is downwind and its ratio is uninterpretable. This computes
    the ratio from the evaluation period's mean precipitation instead, which is
    the measure comparable to Earth's published box climatology (audit process
    note 3).

    Returns `{}` below ~256x512, where a single cell spans a whole range and the
    two flank boxes collapse onto it -- reporting a number there would be
    process note 11 in reverse, an instrument that cannot resolve the phenomenon
    quietly emitting a confident value.
    """
    ratios: dict[str, float] = {}
    for pair in OROGRAPHIC_PAIRS:
        contrast = orographic_contrast(
            mean_precipitation_mm_day, pair, land_mask=land_mask
        )
        if contrast is None or not np.isfinite(contrast["ratio"]):
            continue
        ratios[pair.name] = float(contrast["ratio"])
    if len(ratios) < len(OROGRAPHIC_PAIRS):
        return {}
    shortfalls = [
        max(0.0, pair.ratio_min - ratios[pair.name]) / pair.ratio_min
        for pair in OROGRAPHIC_PAIRS
    ]
    return {
        "ratios": ratios,
        "mean_ratio": float(np.mean(list(ratios.values()))),
        "mean_shortfall_vs_earth_floor": float(np.mean(shortfalls)),
    }


def _koppen_map_skill(state: PlanetState, land_mask: np.ndarray) -> dict[str, Any]:
    """Per-cell Köppen agreement against the gridded reference map (audit H10).

    This is the only validation signal in the platform that is not a spatial
    aggregate. The named boxes, zonal bands, and global Köppen shares can all be
    satisfied by a model whose regional *pattern* is wrong -- a share metric in
    particular is a closed budget, so two compensating regional errors cancel
    exactly. Group accuracy/kappa here cannot be fooled that way.

    Returns an empty dict (rather than raising) when the reference PNG is absent
    so that a checkout without it still validates on every other metric.
    """
    if state.koppen_type is None or not np.any(land_mask):
        return {}
    try:
        from koppen_reference import score_koppen_map

        report = score_koppen_map(state.koppen_type, land_mask=land_mask)
    except (FileNotFoundError, ValueError):
        return {}
    if "group" not in report:
        return {}
    return {
        "group_accuracy": report["group"]["accuracy"],
        "group_kappa": report["group"]["kappa"],
        "group_share_mae_pp": report["group"]["share_mae_pp"],
        "class_accuracy": report["class"]["accuracy"],
        "class_kappa": report["class"]["kappa"],
        "scored_cells": report["scored_cells"],
        "group_accuracy_by_zone": report["group_accuracy_by_zone"],
        "region_group_accuracy": {
            name: (None if value is None else value["group_accuracy"])
            for name, value in report["per_region"].items()
        },
    }


def _koppen_temperature_thresholds(
    state: PlanetState, land_mask: np.ndarray
) -> dict[str, Any]:
    """Land coldest/warmest month vs Köppen's own threshold bounds (audit C1b).

    The anchor-free companion to ``_land_seasonal_cycle_metrics``.  That metric
    scores an area-weighted band mean against mid-continental *station* anchors,
    which is a population mismatch -- see ``EARTH_LAND_CYCLE_REFERENCE``'s own
    revised note.  This one asks a question the reference can answer exactly:
    the reference calls this cell Dfb, so its coldest month is below -3 C; where
    does the model put it?

    Headline scalars are ``coldest_month.accuracy`` and ``warmest_month.accuracy``
    (area-weighted fraction of scorable reference land the model places inside
    the reference-implied interval), each paired with a signed
    ``too_warm_fraction``/``too_cold_fraction`` so a change's *direction* is
    visible rather than just its size.
    """
    if state.monthly_temp is None or not np.any(land_mask):
        return {}
    monthly = np.asarray(state.monthly_temp)
    if monthly.ndim != 3 or monthly.shape[0] != 12:
        return {}
    try:
        from koppen_reference import score_temperature_thresholds

        return score_temperature_thresholds(
            monthly, land_mask=land_mask, model_codes=state.koppen_type
        )
    except (FileNotFoundError, ValueError):
        return {}


def _precip_rescale_metrics(state: PlanetState, pp: PlanetParams) -> dict[str, Any]:
    if (
        state.temperature is None
        or state.wind_u is None
        or state.wind_v is None
        or state.elevation is None
    ):
        return {}
    debug: dict[str, Any] = {}
    H, W = state.elevation.shape
    final_precipitation, _, _, _ = generate_precipitation(
        H,
        W,
        state.elevation,
        temperature=state.temperature,
        wind_u=state.wind_u,
        wind_v=state.wind_v,
        humidity=state.humidity,
        soil_moisture=state.soil_moisture,
        soil_moisture_deep=state.soil_moisture_deep,
        cloud_fraction=state.cloud_cover,
        day_of_year=state.day_of_year,
        dt_days=1.0,
        surface_pressure_hpa=pp.surface_pressure_pa / 100.0,
        planet_params=pp,
        debug_fields=debug,
    )
    scale = np.asarray(debug.get("zonal_rescale_factor", []), dtype=np.float64)
    if scale.size == 0:
        return {}
    raw_column_strategy = bool(pp.enable_prognostic_column_water)
    budget_strategy = bool(pp.moisture_budget_precip_rescale) and not raw_column_strategy
    metrics: dict[str, Any] = {
        "strategy": (
            "raw_conserved_column" if raw_column_strategy
            else "moisture_budget" if budget_strategy else "legacy_multiplier"
        ),
        "row_target_rescale_applied": not raw_column_strategy,
        "mean": float(np.mean(scale)),
        "max": float(np.max(scale)),
        # Only the legacy strategy has a hard 5x multiplier ceiling. Large
        # effective ratios under the budget strategy occur when a near-zero raw
        # row receives bounded moisture and are not ceiling saturation.
        "saturated_fraction": (
            0.0 if (budget_strategy or raw_column_strategy) else float(np.mean(scale >= 4.999))
        ),
    }
    capacity_limited = np.asarray(
        debug.get("precip_rescale_capacity_limited", []), dtype=bool
    )
    unmet = np.asarray(debug.get("precip_rescale_unmet_mm_day", []), dtype=np.float64)
    achieved = np.asarray(
        debug.get("precip_target_achieved_fraction", []), dtype=np.float64
    )
    if capacity_limited.size:
        metrics["capacity_limited_fraction"] = float(np.mean(capacity_limited))
    if unmet.size:
        metrics["mean_unmet_mm_day"] = float(np.mean(unmet))
    if achieved.size:
        metrics["mean_target_achieved_fraction"] = float(np.mean(achieved))

    # The rescale diagnostics above say whether the target was met, but not how
    # much the target-side allocator changed the raw physical rainout. Keep that
    # distinction explicit: the replacement closure must reduce this dependency,
    # not merely find another way to hit the same target. The adjustment is
    # signed because seasonal raw production may also exceed a row target.
    raw_precipitation = np.asarray(
        debug.get("precipitation_raw_mm_day", []), dtype=np.float64
    )
    raw_dq = np.asarray(debug.get("rainout_raw_dq", []), dtype=np.float64)
    humidity = np.asarray(debug.get("humidity_before_rainout", []), dtype=np.float64)
    if raw_precipitation.shape == final_precipitation.shape:
        raw_mean = _area_weighted_mean(raw_precipitation)
        final_mean = _area_weighted_mean(final_precipitation)
        metrics["closure"] = {
            "raw_precip_mm_day": raw_mean,
            "final_precip_mm_day": final_mean,
            "target_adjustment_mm_day": (
                0.0 if raw_column_strategy else final_mean - raw_mean
            ),
            "raw_to_final_ratio": (
                raw_mean / final_mean if final_mean > 1e-12 else float("nan")
            ),
        }
        if raw_dq.shape == humidity.shape and raw_dq.size:
            metrics["closure"]["raw_column_rainout_fraction"] = _area_weighted_mean(
                np.divide(raw_dq, humidity, out=np.zeros_like(raw_dq), where=humidity > 1e-12)
            )
    return metrics


def _monthly_climatology_metrics(
    state: PlanetState,
    land_mask: np.ndarray,
    reference: MonthlyClimatology | None,
    mean_wind_speed_ms: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score the same twelve fields used by Köppen against a fixed reference.

    Wind is scored separately from temperature/precipitation: it uses the
    caller's simple evaluation-period time-mean (not the monthly Köppen
    bins) against the reference's own annual mean, since no wind reference
    here carries genuine monthly resolution on the model side yet -- see
    monthly_climatology.score_monthly_climatology's docstring. Still scored
    even if reference lacks temperature/precipitation (a wind-only
    reference) or the model's monthly bins aren't ready yet.
    """
    if reference is None:
        return {}
    monthly_temp = state.monthly_temp if reference.temperature_k is not None else None
    monthly_precip = state.monthly_precip if reference.precipitation_mm_day is not None else None
    wind_speed = mean_wind_speed_ms if reference.wind_speed_ms is not None else None
    if monthly_temp is None and monthly_precip is None and wind_speed is None:
        return {}
    return score_monthly_climatology(
        monthly_temp,
        monthly_precip,
        reference,
        model_land_mask=land_mask,
        annual_mean_wind_speed_ms=wind_speed,
    )


def _regional_land_temperature_metrics(
    state: PlanetState,
    land_mask: np.ndarray,
    reference: MonthlyClimatology | None,
) -> dict[str, dict[str, float]]:
    """CRU monthly-temperature errors for the existing named land regions."""
    if reference is None or state.monthly_temp is None:
        return {}
    model = np.asarray(state.monthly_temp, dtype=np.float64)
    observed = np.asarray(reference.temperature_k, dtype=np.float64)
    if model.shape != observed.shape:
        return {}
    result: dict[str, dict[str, float]] = {}
    for region in EARTH_PRECIP_REGIONS:
        mask = region_mask(model.shape[1:], region, cell_mask=land_mask)
        if reference.land_fraction is not None:
            mask &= np.asarray(reference.land_fraction >= 0.5, dtype=bool)
        if not np.any(mask):
            continue
        model_series = np.mean(model[:, mask], axis=1)
        observed_series = np.mean(observed[:, mask], axis=1)
        delta_c = model_series - observed_series
        worst_month = int(np.argmax(np.abs(delta_c)))
        result[region.name] = {
            "annual_bias_c": float(np.mean(delta_c)),
            "monthly_rmse_c": float(np.sqrt(np.mean(delta_c**2))),
            "monthly_bias_c": [float(value) for value in delta_c],
            "worst_month_index": worst_month + 1,
            "worst_month_bias_c": float(delta_c[worst_month]),
            "model_seasonal_range_c": float(np.max(model_series) - np.min(model_series)),
            "reference_seasonal_range_c": float(
                np.max(observed_series) - np.min(observed_series)
            ),
            "seasonal_range_error_c": float(
                (np.max(model_series) - np.min(model_series))
                - (np.max(observed_series) - np.min(observed_series))
            ),
        }
    return result


def summarize_real_terrain_climate(
    state: PlanetState,
    *,
    mean_temperature_k: np.ndarray,
    mean_precipitation_mm_day: np.ndarray,
    mean_cloud_fraction: np.ndarray,
    mean_soil_moisture: np.ndarray,
    planet_params: PlanetParams,
    monthly_climatology: MonthlyClimatology | None = None,
    mean_wind_speed_ms: np.ndarray | None = None,
    mean_surface_temperature_k: np.ndarray | None = None,
    mean_lower_meridional_wind_m_s: np.ndarray | None = None,
    mean_upper_meridional_wind_m_s: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute stable global, zonal, and named-region validation metrics."""
    sea_mask, land_mask = get_masks(state.elevation, use_cache=False)
    circulation: dict[str, Any] = {}
    if state.wind_u is not None and state.wind_v is not None:
        # Hydrostatic geopotential (g*z) at each level's nominal pressure, so the
        # transport diagnostic's dry-static-energy term is cp*T + g*z rather than
        # cp*T alone.  Best-effort: only supplied where the elevation field lines
        # up with the temperature grid actually being scored.
        elevation_m = elevation_to_alt_km(
            state.elevation, max_elevation_km=float(planet_params.max_elevation_km),
        ) * 1000.0
        lower_gz = mid_gz = upper_gz = None
        if elevation_m.shape == mean_temperature_k.shape:
            lower_gz = pressure_level_geopotential(
                elevation_m, mean_temperature_k,
                gravity_m_s2=float(planet_params.surface_gravity),
                gas_constant_dry=float(planet_params.gas_constant_dry),
                surface_pressure_pa=float(planet_params.surface_pressure_pa),
                level_pressure_pa=float(planet_params.native_balanced_surface_pressure_pa),
            )
            if (
                state.midlevel_temperature is not None
                and state.midlevel_temperature.shape == elevation_m.shape
            ):
                mid_gz = pressure_level_geopotential(
                    elevation_m, state.midlevel_temperature,
                    gravity_m_s2=float(planet_params.surface_gravity),
                    gas_constant_dry=float(planet_params.gas_constant_dry),
                    surface_pressure_pa=float(planet_params.surface_pressure_pa),
                    level_pressure_pa=float(planet_params.native_balanced_mid_pressure_pa),
                )
            if (
                state.upperlevel_temperature is not None
                and state.upperlevel_temperature.shape == elevation_m.shape
            ):
                upper_gz = pressure_level_geopotential(
                    elevation_m, state.upperlevel_temperature,
                    gravity_m_s2=float(planet_params.surface_gravity),
                    gas_constant_dry=float(planet_params.gas_constant_dry),
                    surface_pressure_pa=float(planet_params.surface_pressure_pa),
                    level_pressure_pa=float(planet_params.three_level_thermal_wind_upper_pressure_pa),
                )
        # Section 17 (PRIOR_ART_IMPLEMENTATION_PLAN.md): the three-level path's
        # upper level is now decoupled from the shared, always-on jet-stream
        # kernel (state.wind_u_aloft/wind_v_aloft). Score the independent
        # state whenever it has been populated (i.e. the three-level gate has
        # been active), so the transport/jet diagnostics that motivated this
        # investigation measure the level that the three-level physics
        # actually evolves. Falls back to the shared kernel exactly as before
        # for states that never ran the three-level path (state.upperlevel_wind_u
        # is None), so legacy/two-layer-only runs are unaffected.
        circulation = circulation_scorecard(
            state.wind_u,
            state.wind_v,
            mid_u=state.midlevel_wind_u,
            mid_v=state.midlevel_wind_v,
            upper_u=(
                state.upperlevel_wind_u
                if state.upperlevel_wind_u is not None
                else state.wind_u_aloft
            ),
            upper_v=(
                state.upperlevel_wind_v
                if state.upperlevel_wind_v is not None
                else state.wind_v_aloft
            ),
            omega_lower_mid_pa_s=state.omega_lower_mid_pa_s,
            omega_mid_upper_pa_s=state.omega_mid_upper_pa_s,
            temperature_k=mean_temperature_k,
            humidity=state.humidity,
            radius_m=float(planet_params.radius_m),
            surface_pressure_pa=float(planet_params.surface_pressure_pa),
            gravity_m_s2=float(planet_params.surface_gravity),
            cp_dry_j_kg_k=float(planet_params.cp_dry),
            midlevel_temperature_k=state.midlevel_temperature,
            midlevel_humidity=state.midlevel_humidity,
            upperlevel_temperature_k=state.upperlevel_temperature,
            upperlevel_humidity=state.upperlevel_humidity,
            layer_mass_fractions=(0.40, 0.35, 0.25),
            lower_geopotential_m2_s2=lower_gz,
            midlevel_geopotential_m2_s2=mid_gz,
            upperlevel_geopotential_m2_s2=upper_gz,
        )
    period = float(planet_params.orbital_period_days)
    regional_precip = precipitation_by_region_mm_year(
        mean_precipitation_mm_day,
        land_mask=land_mask,
        days_per_year=period,
    )
    regional_soil = {
        region.name: region_mean(mean_soil_moisture, region, cell_mask=land_mask)
        for region in EARTH_PRECIP_REGIONS
    }
    region_errors = {
        region.name: (
            None
            if regional_precip[region.name] is None
            else target_error_fraction(float(regional_precip[region.name]), region)
        )
        for region in EARTH_PRECIP_REGIONS
    }
    desert_values = [
        float(regional_precip[region.name])
        for region in EARTH_PRECIP_REGIONS
        if region.group == "desert" and regional_precip[region.name] is not None
    ]
    continental_values = [
        float(regional_precip[region.name])
        for region in EARTH_PRECIP_REGIONS
        if region.group == "continental" and regional_precip[region.name] is not None
    ]

    H = state.elevation.shape[0]
    lat = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * 180.0
    nh_midlat = land_mask & ((lat >= 45.0) & (lat <= 65.0))[:, None]
    polar_mask = np.broadcast_to((np.abs(lat) >= 70.0)[:, None], land_mask.shape)
    land_soil = mean_soil_moisture[land_mask]
    nh_soil = mean_soil_moisture[nh_midlat]
    zonal = _zonal_metrics(
        mean_temperature_k,
        mean_precipitation_mm_day,
        days_per_year=period,
    )

    finite_region_errors = [float(v) for v in region_errors.values() if v is not None]
    temperature_errors = [abs(v["temperature_bias_c"]) / 10.0 for v in zonal.values()]
    precipitation_errors = [
        abs(float(np.log(max(v["precip_ratio"], 1e-6)))) for v in zonal.values()
    ]
    score_terms = finite_region_errors + temperature_errors + precipitation_errors
    reference_error_score = float(np.mean(score_terms)) if score_terms else float("nan")
    two_layer_overturning = {}
    if mean_surface_temperature_k is not None:
        two_layer_overturning = diagnose_two_layer_overturning_state(
            mean_surface_temperature_k,
            mean_temperature_k,
            mean_precipitation_mm_day,
            mean_lower_meridional_wind_m_s,
            mean_upper_meridional_wind_m_s,
            hadley_edge_deg=float(planet_params.wind_upper_hadley_edge_deg),
            upper_temperature_k=state.upperlevel_temperature,
        )

    return {
        "global": {
            "temperature_k": _area_weighted_mean(mean_temperature_k),
            "precip_mm_day": _area_weighted_mean(mean_precipitation_mm_day),
                "polar_precip_mm_day": _area_weighted_mean(
                    mean_precipitation_mm_day, polar_mask
                ),
            "cloud_fraction": _area_weighted_mean(mean_cloud_fraction),
            "land_soil_moisture": _area_weighted_mean(mean_soil_moisture, land_mask),
            "ocean_temperature_k": _area_weighted_mean(mean_temperature_k, sea_mask),
            "land_soil_floor_fraction": (
                float(np.mean(land_soil <= 0.051)) if land_soil.size else float("nan")
            ),
            "nh_midlat_soil_floor_fraction": (
                float(np.mean(nh_soil <= 0.051)) if nh_soil.size else float("nan")
            ),
        },
        "regional_precip_mm_year": regional_precip,
        "regional_soil_moisture": regional_soil,
        "regional_target_error_fraction": region_errors,
        "continental_minus_desert_precip_mm_year": (
            float(np.mean(continental_values) - np.mean(desert_values))
            if continental_values and desert_values
            else None
        ),
        "zonal": zonal,
        "orographic_contrast": _orographic_contrast_metrics(
            mean_precipitation_mm_day, land_mask
        ),
        "land_seasonal_cycle": _land_seasonal_cycle_metrics(state, land_mask),
        "koppen_land_percent": _koppen_land_percentages(state, land_mask),
        "koppen_map_skill": _koppen_map_skill(state, land_mask),
        "koppen_temperature_thresholds": _koppen_temperature_thresholds(
            state, land_mask
        ),
        "monthly_climatology": _monthly_climatology_metrics(
            state, land_mask, monthly_climatology, mean_wind_speed_ms
        ),
        "regional_land_temperature": _regional_land_temperature_metrics(
            state, land_mask, monthly_climatology
        ),
        "precip_rescale": _precip_rescale_metrics(state, planet_params),
        "circulation": circulation,
        "two_layer_overturning": two_layer_overturning,
        "reference_error_score": reference_error_score,
    }


def _cycle_count(years: float, mode: TimeScaleMode, pp: PlanetParams) -> int:
    if years <= 0.0:
        return 0
    return max(1, int(round(years * pp.orbital_period_days / cycle_days(mode, pp))))


def _advance_cycle(
    state: PlanetState,
    mode: TimeScaleMode,
    pp: PlanetParams,
    config: RealTerrainValidationConfig,
) -> PlanetState:
    for step_days, update_wind in substeps_for_mode(mode, pp):
        state, _ = simulate_step(
            state,
            days=step_days,
            block_size=config.block_size,
            wind_block_size=config.wind_block_size,
            precip_block_size=config.precip_block_size,
            update_wind=update_wind,
            time_scale=mode,
            planet_params=pp,
            track_components=False,
        )
    return state


def _advance_cycle_with_heat_convergence(
    state: PlanetState,
    mode: TimeScaleMode,
    pp: PlanetParams,
    config: RealTerrainValidationConfig,
) -> tuple[PlanetState, list[np.ndarray], list[float]]:
    """Advance one cycle while retaining the Phase 3 forcing fields."""
    samples: list[np.ndarray] = []
    area_means: list[float] = []
    for step_days, update_wind in substeps_for_mode(mode, pp):
        state, components = simulate_step(
            state,
            days=step_days,
            block_size=config.block_size,
            wind_block_size=config.wind_block_size,
            precip_block_size=config.precip_block_size,
            update_wind=update_wind,
            time_scale=mode,
            planet_params=pp,
            track_components=True,
        )
        field = components.get("atmospheric_heat_convergence_w_m2")
        if field is not None:
            samples.append(np.asarray(field, dtype=np.float32))
            area_mean = components.get(
                "atmospheric_heat_convergence_applied_grid_area_mean_w_m2"
            )
            if area_mean is None:
                raise RuntimeError("Phase 3 forcing lacks applied-grid closure diagnostic")
            area_means.append(float(area_mean))
    return state, samples, area_means


def _regional_moisture_budget_snapshot(
    state: PlanetState,
    planet_params: PlanetParams,
) -> dict[str, dict[str, float | None]]:
    """Re-evaluate precipitation diagnostics without changing validation state.

    This deliberately uses a one-day native-grid call on each sampled climate
    state. It exposes the precipitation scheme's current drivers and allocator
    response, while keeping validation evolution bit-for-bit unchanged.
    """
    if state.elevation is None or state.wind_u is None or state.wind_v is None:
        raise RuntimeError("regional moisture diagnostic requires elevation and lower wind")
    temperature = state.air_temperature if state.air_temperature is not None else state.temperature
    if temperature is None or state.humidity is None:
        raise RuntimeError("regional moisture diagnostic requires temperature and humidity")
    H, W = state.elevation.shape
    debug: dict = {}
    generate_precipitation(
        H,
        W,
        state.elevation,
        temperature=temperature,
        wind_u=state.wind_u,
        wind_v=state.wind_v,
        wind_u_aloft=state.wind_u_aloft,
        wind_v_aloft=state.wind_v_aloft,
        wind_u_midlevel=state.midlevel_wind_u,
        wind_v_midlevel=state.midlevel_wind_v,
        humidity=state.humidity,
        soil_moisture=state.soil_moisture,
        soil_moisture_deep=state.soil_moisture_deep,
        condensate=state.atmospheric_condensate,
        precipitating_hydrometeors=state.precipitating_hydrometeors,
        midlevel_temperature=state.midlevel_temperature,
        midlevel_humidity=state.midlevel_humidity,
        upperlevel_temperature=state.upperlevel_temperature,
        upperlevel_humidity=state.upperlevel_humidity,
        cloud_fraction=state.cloud_cover,
        day_of_year=state.day_of_year,
        dt_days=1.0,
        surface_pressure_hpa=float(planet_params.surface_pressure_pa) / 100.0,
        planet_params=planet_params,
        debug_fields=debug,
    )
    return regional_moisture_budget_snapshot(
        state.elevation,
        debug,
        pathway_fields={
            "lower_wind_u_m_s": state.wind_u,
            "lower_wind_v_m_s": state.wind_v,
            "surface_temperature_k": state.temperature,
            "air_temperature_k": temperature,
            "cloud_fraction": state.cloud_cover,
        },
        radius_m=float(planet_params.radius_m),
    )


def run_real_terrain_validation(
    config: RealTerrainValidationConfig = RealTerrainValidationConfig(),
    *,
    planet_params: PlanetParams = EARTH,
    initial_state_path: str | Path | None = None,
    monthly_climatology_path: str | Path | None = None,
    wind_climatology_path: str | Path | None = None,
    track_phase3_heat_convergence: bool = False,
) -> tuple[PlanetState, dict[str, Any]]:
    """Run a deterministic real-DEM spinup/evaluation and return state plus report.

    monthly_climatology_path and wind_climatology_path are independent and
    both optional: CRU TS (temperature/precipitation) and NCEP/NCAR
    Reanalysis 1 (wind) are different providers at different native
    resolutions with no wind/T-P overlap (CRU publishes no wind variable at
    all -- see scripts/build_ncep_wind_reference.py), so they are loaded and
    regridded separately, then merged into one MonthlyClimatology for
    scoring. Either may be given without the other.
    """
    config.validate()
    mode = TimeScaleMode[config.time_scale]
    monthly_climatology = None
    if monthly_climatology_path is not None:
        monthly_climatology = regrid_monthly_climatology(
            load_monthly_climatology(monthly_climatology_path), config.height, config.width
        )
    if wind_climatology_path is not None:
        wind_reference = regrid_monthly_climatology(
            load_monthly_climatology(wind_climatology_path), config.height, config.width
        )
        if monthly_climatology is None:
            monthly_climatology = wind_reference
        else:
            # Two different providers merged into one object for scoring;
            # preserve wind's own attribution separately so the report does
            # not silently mislabel wind_speed_ms's provenance as CRU's.
            merged_metadata = dict(monthly_climatology.metadata)
            merged_metadata["wind_source"] = wind_reference.metadata["source"]
            merged_metadata["wind_period"] = wind_reference.metadata["period"]
            monthly_climatology = dataclasses_replace(
                monthly_climatology,
                wind_speed_ms=wind_reference.wind_speed_ms,
                metadata=merged_metadata,
            )
    clear_simulation_caches()
    if initial_state_path is None:
        elevation = load_bundled_earth_dem(config.height, config.width)
        state = create_initial_state(
            elevation,
            day_of_year=config.start_day,
            planet_params=planet_params,
            block_size=config.block_size,
            wind_block_size=config.wind_block_size,
            precip_block_size=config.precip_block_size,
        )
    else:
        state = load_state(initial_state_path)
        if state.elevation.shape != (config.height, config.width):
            raise ValueError(
                f"loaded state grid {state.elevation.shape} does not match "
                f"validation grid {(config.height, config.width)}"
            )

    spinup_cycles = _cycle_count(config.spinup_years, mode, planet_params)
    evaluation_cycles = _cycle_count(config.evaluation_years, mode, planet_params)
    started = time.perf_counter()
    for _ in range(spinup_cycles):
        state = _advance_cycle(state, mode, planet_params, config)

    accumulators: dict[str, np.ndarray | None] = {
        "temperature": None,
        "surface_temperature": None,
        "precipitation": None,
        "cloud": None,
        "soil": None,
    }
    wind_speed_accumulator: np.ndarray | None = None
    lower_v_accumulator: np.ndarray | None = None
    upper_v_accumulator: np.ndarray | None = None
    lower_u_samples: list[np.ndarray] = []
    upper_u_samples: list[np.ndarray] = []
    regional_moisture_snapshots: list[
        tuple[float, dict[str, dict[str, float | None]]]
    ] = []
    seasonal_regional_moisture_snapshots: list[
        tuple[str, float, dict[str, dict[str, float | None]]]
    ] = []
    heat_convergence_samples: list[np.ndarray] = []
    heat_convergence_area_means: list[float] = []
    sampled_days = 0.0
    cycle_duration = cycle_days(mode, planet_params)
    for _ in range(evaluation_cycles):
        if track_phase3_heat_convergence:
            state, cycle_heat_samples, cycle_heat_area_means = (
                _advance_cycle_with_heat_convergence(state, mode, planet_params, config)
            )
            heat_convergence_samples.extend(cycle_heat_samples)
            heat_convergence_area_means.extend(cycle_heat_area_means)
        else:
            state = _advance_cycle(state, mode, planet_params, config)
        regional_snapshot = _regional_moisture_budget_snapshot(state, planet_params)
        regional_moisture_snapshots.append((cycle_duration, regional_snapshot))
        seasonal_regional_moisture_snapshots.append(
            (
                season_for_day(
                    state.day_of_year,
                    orbital_period_days=float(planet_params.orbital_period_days),
                    vernal_equinox_day=float(planet_params.vernal_equinox_day),
                ),
                cycle_duration,
                regional_snapshot,
            )
        )
        temperature = (
            state.air_temperature if state.air_temperature is not None else state.temperature
        )
        fields = {
            "temperature": temperature,
            "surface_temperature": state.temperature,
            "precipitation": state.precipitation,
            "cloud": state.cloud_cover,
            "soil": state.soil_moisture,
        }
        for name, field in fields.items():
            if field is None:
                raise RuntimeError(f"simulation did not produce required field {name}")
            weighted = np.asarray(field, dtype=np.float64) * cycle_duration
            accumulators[name] = (
                weighted if accumulators[name] is None else accumulators[name] + weighted
            )
        # Optional: not every configuration guarantees wind (unlike the
        # required fields above), so this stays best-effort rather than
        # raising -- mean_wind_speed_ms is simply None (no CRU wind score)
        # if wind was never present.
        if state.wind_u is not None and state.wind_v is not None:
            lower_u_samples.append(np.asarray(state.wind_u, dtype=np.float64))
            speed = np.sqrt(
                np.asarray(state.wind_u, dtype=np.float64) ** 2
                + np.asarray(state.wind_v, dtype=np.float64) ** 2
            )
            weighted_speed = speed * cycle_duration
            wind_speed_accumulator = (
                weighted_speed if wind_speed_accumulator is None
                else wind_speed_accumulator + weighted_speed
            )
            weighted_lower_v = np.asarray(state.wind_v, dtype=np.float64) * cycle_duration
            lower_v_accumulator = (
                weighted_lower_v if lower_v_accumulator is None
                else lower_v_accumulator + weighted_lower_v
            )
        if state.wind_v_aloft is not None:
            if state.wind_u_aloft is None:
                raise RuntimeError("upper meridional wind present without upper zonal wind")
            upper_u_samples.append(np.asarray(state.wind_u_aloft, dtype=np.float64))
            weighted_upper_v = np.asarray(state.wind_v_aloft, dtype=np.float64) * cycle_duration
            upper_v_accumulator = (
                weighted_upper_v if upper_v_accumulator is None
                else upper_v_accumulator + weighted_upper_v
            )
        sampled_days += cycle_duration

    if sampled_days <= 0.0:
        raise RuntimeError("validation collected no evaluation samples")
    means = {
        name: np.asarray(value, dtype=np.float64) / sampled_days
        for name, value in accumulators.items()
        if value is not None
    }
    mean_wind_speed_ms = (
        wind_speed_accumulator / sampled_days if wind_speed_accumulator is not None else None
    )
    mean_lower_v = lower_v_accumulator / sampled_days if lower_v_accumulator is not None else None
    mean_upper_v = upper_v_accumulator / sampled_days if upper_v_accumulator is not None else None
    metrics = summarize_real_terrain_climate(
        state,
        mean_temperature_k=means["temperature"],
        mean_precipitation_mm_day=means["precipitation"],
        mean_cloud_fraction=means["cloud"],
        mean_soil_moisture=means["soil"],
        planet_params=planet_params,
        monthly_climatology=monthly_climatology,
        mean_wind_speed_ms=mean_wind_speed_ms,
        mean_surface_temperature_k=means["surface_temperature"],
        mean_lower_meridional_wind_m_s=mean_lower_v,
        mean_upper_meridional_wind_m_s=mean_upper_v,
    )
    if track_phase3_heat_convergence:
        from atmospheric_heat_transport import summarize_heat_convergence_samples

        latitude = (
            0.5 - (np.arange(config.height, dtype=np.float64) + 0.5) / config.height
        ) * np.pi
        metrics["atmospheric_heat_convergence"] = summarize_heat_convergence_samples(
            heat_convergence_samples,
            latitude,
            applied_grid_area_means_w_m2=heat_convergence_area_means,
        )
    metrics["regional_moisture_budget"] = time_average_regional_moisture_budget(
        regional_moisture_snapshots
    )
    metrics["seasonal_regional_moisture_budget"] = seasonal_regional_moisture_budget(
        seasonal_regional_moisture_snapshots
    )
    if lower_u_samples:
        metrics["seasonal_jet"] = seasonal_jet_scorecard(
            lower_u_samples,
            upper_zonal_wind_samples=(
                upper_u_samples if len(upper_u_samples) == len(lower_u_samples) else None
            ),
        )
    report = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "config": asdict(config),
        "planet": {
            "orbital_period_days": planet_params.orbital_period_days,
            "solar_constant": planet_params.solar_constant,
            "obliquity_deg": planet_params.obliquity_deg,
            "eccentricity": planet_params.eccentricity,
        },
        "planet_params": asdict(planet_params),
        "monthly_climatology_reference": (
            dict(monthly_climatology.metadata) if monthly_climatology is not None else None
        ),
        "simulation": {
            "spinup_cycles": spinup_cycles,
            "evaluation_cycles": evaluation_cycles,
            "sampled_days": sampled_days,
            "final_total_days": state.total_days,
            "wall_seconds": time.perf_counter() - started,
        },
        "metrics": metrics,
    }
    return state, report


def save_validation_report(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_validation_report(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    version = int(report.get("schema_version", -1))
    if version != VALIDATION_SCHEMA_VERSION:
        raise ValueError(
            f"validation report schema v{version} is unsupported; "
            f"expected v{VALIDATION_SCHEMA_VERSION}"
        )
    if "config" not in report or "metrics" not in report:
        raise ValueError("validation report is missing config or metrics")
    return report


def compare_validation_reports(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    relative_tolerance: float = 0.15,
) -> list[str]:
    """Return human-readable regressions against a same-configuration baseline."""
    if current.get("config") != baseline.get("config"):
        return ["validation configuration differs from baseline"]
    # Reference reports predate new default-off research gates. Compare every
    # parameter the baseline actually recorded, while requiring each of those
    # keys to still exist; newly added defaults must not invalidate an otherwise
    # identical climate regression fixture before its metrics are evaluated.
    current_params = current.get("planet_params") or {}
    baseline_params = baseline.get("planet_params") or {}
    current_planet = json.dumps(
        {name: current_params.get(name, "__missing__") for name in baseline_params},
        sort_keys=True,
    )
    baseline_planet = json.dumps(baseline_params, sort_keys=True)
    if current_planet != baseline_planet:
        return ["planet parameters differ from baseline"]
    failures: list[str] = []
    current_metrics = current["metrics"]
    baseline_metrics = baseline["metrics"]

    scalar_tolerances = {
        "temperature_k": 2.0,
        "precip_mm_day": 0.25,
        "polar_precip_mm_day": 0.15,
        "cloud_fraction": 0.04,
        "land_soil_moisture": 0.05,
        "land_soil_floor_fraction": 0.05,
        "nh_midlat_soil_floor_fraction": 0.05,
    }
    for name, absolute_tolerance in scalar_tolerances.items():
        if name not in baseline_metrics["global"]:
            continue
        got = float(current_metrics["global"][name])
        want = float(baseline_metrics["global"][name])
        if abs(got - want) > absolute_tolerance:
            failures.append(
                f"global.{name}: {got:.6g} vs baseline {want:.6g} "
                f"(allowed ±{absolute_tolerance:g})"
            )

    for section in ("regional_precip_mm_year", "regional_soil_moisture"):
        for name, want_value in baseline_metrics[section].items():
            got_value = current_metrics[section].get(name)
            if want_value is None or got_value is None:
                if want_value != got_value:
                    failures.append(f"{section}.{name}: missing-value mismatch")
                continue
            want = float(want_value)
            got = float(got_value)
            allowed = max(abs(want) * relative_tolerance, 0.02 if "soil" in section else 20.0)
            if abs(got - want) > allowed:
                failures.append(
                    f"{section}.{name}: {got:.6g} vs baseline {want:.6g} "
                    f"(allowed ±{allowed:.6g})"
                )

    # Gridded map skill (audit H10). Only the five headline scalars are gated:
    # the per-zone and per-region entries in the same block are recorded for
    # diagnosis but are far too noisy to assert on at 64x128, where a named box
    # can hold a handful of land cells.
    baseline_skill = baseline_metrics.get("koppen_map_skill") or {}
    current_skill = current_metrics.get("koppen_map_skill") or {}
    if baseline_skill and current_skill:
        for name, absolute_tolerance in (
            ("group_accuracy", 0.03),
            ("group_kappa", 0.03),
            ("group_share_mae_pp", 1.0),
            ("class_accuracy", 0.03),
            ("class_kappa", 0.03),
        ):
            if name not in baseline_skill or name not in current_skill:
                continue
            got = float(current_skill[name])
            want = float(baseline_skill[name])
            if abs(got - want) > absolute_tolerance:
                failures.append(
                    f"koppen_map_skill.{name}: {got:.6g} vs baseline {want:.6g} "
                    f"(allowed ±{absolute_tolerance:g})"
                )

    # Anchor-free land temperature skill (audit C1b). Gated on the two headline
    # accuracies only; the directional too_warm/too_cold splits and the per-zone
    # breakdown are diagnostic, and a change that moves land across a Köppen
    # threshold necessarily moves both halves of a split at once.
    baseline_thresholds = baseline_metrics.get("koppen_temperature_thresholds") or {}
    current_thresholds = current_metrics.get("koppen_temperature_thresholds") or {}
    if baseline_thresholds and current_thresholds:
        for name in ("coldest_month", "warmest_month"):
            want_entry = baseline_thresholds.get(name) or {}
            got_entry = current_thresholds.get(name) or {}
            if "accuracy" not in want_entry or "accuracy" not in got_entry:
                continue
            got = float(got_entry["accuracy"])
            want = float(want_entry["accuracy"])
            if abs(got - want) > 0.03:
                failures.append(
                    f"koppen_temperature_thresholds.{name}.accuracy: {got:.6g} "
                    f"vs baseline {want:.6g} (allowed ±0.03)"
                )

    for name, want_values in baseline_metrics["zonal"].items():
        got_values = current_metrics["zonal"].get(name)
        if got_values is None:
            failures.append(f"zonal.{name}: missing")
            continue
        for metric_name, allowed in (("temperature_c", 2.0), ("precip_mm_year", 100.0)):
            got = float(got_values[metric_name])
            want = float(want_values[metric_name])
            allowed_effective = max(allowed, abs(want) * relative_tolerance)
            if abs(got - want) > allowed_effective:
                failures.append(
                    f"zonal.{name}.{metric_name}: {got:.6g} vs baseline {want:.6g} "
                    f"(allowed ±{allowed_effective:.6g})"
                )
    return failures
