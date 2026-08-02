"""Deterministic real-terrain climate validation and regression scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from atmosphere import generate_precipitation
from masks import get_masks
from planet_params import EARTH, PlanetParams
from regional_validation import (
    EARTH_PRECIP_REGIONS,
    precipitation_by_region_mm_year,
    region_mean,
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
    generate_precipitation(
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
    budget_strategy = bool(pp.moisture_budget_precip_rescale)
    metrics: dict[str, float | str] = {
        "strategy": "moisture_budget" if budget_strategy else "legacy_multiplier",
        "mean": float(np.mean(scale)),
        "max": float(np.max(scale)),
        # Only the legacy strategy has a hard 5x multiplier ceiling. Large
        # effective ratios under the budget strategy occur when a near-zero raw
        # row receives bounded moisture and are not ceiling saturation.
        "saturated_fraction": (
            0.0 if budget_strategy else float(np.mean(scale >= 4.999))
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
    return metrics


def summarize_real_terrain_climate(
    state: PlanetState,
    *,
    mean_temperature_k: np.ndarray,
    mean_precipitation_mm_day: np.ndarray,
    mean_cloud_fraction: np.ndarray,
    mean_soil_moisture: np.ndarray,
    planet_params: PlanetParams,
) -> dict[str, Any]:
    """Compute stable global, zonal, and named-region validation metrics."""
    sea_mask, land_mask = get_masks(state.elevation, use_cache=False)
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
        "koppen_land_percent": _koppen_land_percentages(state, land_mask),
        "precip_rescale": _precip_rescale_metrics(state, planet_params),
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


def run_real_terrain_validation(
    config: RealTerrainValidationConfig = RealTerrainValidationConfig(),
    *,
    planet_params: PlanetParams = EARTH,
    initial_state_path: str | Path | None = None,
) -> tuple[PlanetState, dict[str, Any]]:
    """Run a deterministic real-DEM spinup/evaluation and return state plus report."""
    config.validate()
    mode = TimeScaleMode[config.time_scale]
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
        "precipitation": None,
        "cloud": None,
        "soil": None,
    }
    sampled_days = 0.0
    cycle_duration = cycle_days(mode, planet_params)
    for _ in range(evaluation_cycles):
        state = _advance_cycle(state, mode, planet_params, config)
        temperature = (
            state.air_temperature if state.air_temperature is not None else state.temperature
        )
        fields = {
            "temperature": temperature,
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
        sampled_days += cycle_duration

    if sampled_days <= 0.0:
        raise RuntimeError("validation collected no evaluation samples")
    means = {
        name: np.asarray(value, dtype=np.float64) / sampled_days
        for name, value in accumulators.items()
        if value is not None
    }
    metrics = summarize_real_terrain_climate(
        state,
        mean_temperature_k=means["temperature"],
        mean_precipitation_mm_day=means["precipitation"],
        mean_cloud_fraction=means["cloud"],
        mean_soil_moisture=means["soil"],
        planet_params=planet_params,
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
    current_planet = json.dumps(current.get("planet_params"), sort_keys=True)
    baseline_planet = json.dumps(baseline.get("planet_params"), sort_keys=True)
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
