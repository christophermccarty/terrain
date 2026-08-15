"""Attribute the fixed-depth boundary-layer candidate's regional heat budget."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundary_layer import mixed_layer_pressure_thickness  # noqa: E402
from masks import get_masks  # noqa: E402
from planet_params import EARTH  # noqa: E402
from real_terrain_validation import (  # noqa: E402
    RealTerrainValidationConfig,
    load_bundled_earth_dem,
)
from regional_validation import EARTH_PRECIP_REGIONS, region_mean  # noqa: E402
from simulate import clear_simulation_caches, create_initial_state, simulate_step  # noqa: E402
from simulation_state import TimeScaleMode  # noqa: E402
from time_policy import substeps_for_mode  # noqa: E402


PRIORITY_REGIONS = {
    region.name: region
    for region in EARTH_PRECIP_REGIONS
    if region.name in {"Central Europe", "East China", "S Japan", "SE US"}
}


def _mean(field, region, land):
    if field is None:
        return None
    return region_mean(np.asarray(field), region, cell_mask=land)


def _run(params, config, *, diagnose: bool):
    clear_simulation_caches()
    elevation = load_bundled_earth_dem(config.height, config.width)
    _, land = get_masks(elevation)
    state = create_initial_state(
        elevation,
        day_of_year=config.start_day,
        planet_params=params,
        block_size=config.block_size,
        wind_block_size=config.wind_block_size,
        precip_block_size=config.precip_block_size,
    )
    mode = TimeScaleMode[config.time_scale]
    for _ in range(12):
        for step_days, update_wind in substeps_for_mode(mode, params):
            state, _ = simulate_step(
                state, days=step_days, block_size=config.block_size,
                wind_block_size=config.wind_block_size,
                precip_block_size=config.precip_block_size,
                update_wind=update_wind, time_scale=mode,
                planet_params=params, track_components=False,
            )

    monthly = {name: [] for name in PRIORITY_REGIONS}
    for month in range(12):
        flux_samples = []
        for step_days, update_wind in substeps_for_mode(mode, params):
            state, components = simulate_step(
                state, days=step_days, block_size=config.block_size,
                wind_block_size=config.wind_block_size,
                precip_block_size=config.precip_block_size,
                update_wind=update_wind, time_scale=mode,
                planet_params=params, track_components=diagnose,
            )
            if diagnose:
                flux_samples.append(components)
        for name, region in PRIORITY_REGIONS.items():
            boundary = state.boundary_layer_temperature if diagnose else state.air_temperature
            row = {
                "month": month + 1,
                "surface_temperature_c": (
                    None if (v := _mean(state.temperature, region, land)) is None else v - 273.15
                ),
                "near_surface_temperature_c": (
                    None if (v := _mean(boundary, region, land)) is None else v - 273.15
                ),
                "free_air_temperature_c": (
                    None if (v := _mean(state.air_temperature, region, land)) is None else v - 273.15
                ),
                "precipitation_mm_day": _mean(state.precipitation, region, land),
            }
            if diagnose and flux_samples:
                for output_name, component_name in (
                    ("surface_sensible_gain_w_m2", "land_air_sensible_atmospheric_gain_w_m2"),
                    ("boundary_exchange_gain_w_m2", "boundary_layer_exchange_gain_w_m2"),
                    ("free_air_exchange_gain_w_m2", "free_air_exchange_gain_w_m2"),
                    ("resolved_convergence_w_m2", "atmospheric_heat_convergence_w_m2"),
                    ("boundary_horizontal_convergence_w_m2", "boundary_layer_horizontal_heat_convergence_w_m2"),
                    ("continuity_exchange_gain_w_m2", "boundary_layer_continuity_exchange_gain_w_m2"),
                ):
                    values = [
                        _mean(sample.get(component_name), region, land)
                        for sample in flux_samples
                    ]
                    valid = [value for value in values if value is not None]
                    row[output_name] = None if not valid else float(np.mean(valid))
            monthly[name].append(row)
    return state, monthly


def _summary(control, candidate):
    result = {}
    for name in PRIORITY_REGIONS:
        c_rows = control[name]
        b_rows = candidate[name]
        if all(row["near_surface_temperature_c"] is None for row in b_rows):
            result[name] = {"resolved": False}
            continue
        c_temp = np.array([row["near_surface_temperature_c"] for row in c_rows])
        b_temp = np.array([row["near_surface_temperature_c"] for row in b_rows])
        sensible = np.array([row["surface_sensible_gain_w_m2"] for row in b_rows])
        exchange = np.array([row["boundary_exchange_gain_w_m2"] for row in b_rows])
        horizontal = np.array([row["boundary_horizontal_convergence_w_m2"] for row in b_rows])
        continuity = np.array([row["continuity_exchange_gain_w_m2"] for row in b_rows])
        result[name] = {
            "resolved": True,
            "annual_near_surface_change_k": float(np.mean(b_temp - c_temp)),
            "seasonal_range_control_k": float(np.max(c_temp) - np.min(c_temp)),
            "seasonal_range_candidate_k": float(np.max(b_temp) - np.min(b_temp)),
            "coldest_month_change_k": float(np.min(b_temp) - np.min(c_temp)),
            "warmest_month_change_k": float(np.max(b_temp) - np.max(c_temp)),
            "annual_surface_sensible_gain_w_m2": float(np.mean(sensible)),
            "annual_boundary_exchange_gain_w_m2": float(np.mean(exchange)),
            "annual_horizontal_convergence_w_m2": float(np.mean(horizontal)),
            "annual_continuity_exchange_gain_w_m2": float(np.mean(continuity)),
            "cold_half_horizontal_convergence_w_m2": float(np.mean(horizontal[np.argsort(b_temp)[:6]])),
            "cold_half_exchange_gain_w_m2": float(np.mean(exchange[np.argsort(b_temp)[:6]])),
            "warm_half_exchange_gain_w_m2": float(np.mean(exchange[np.argsort(b_temp)[-6:]])),
            "annual_precipitation_change_mm_day": float(np.mean([
                b["precipitation_mm_day"] - c["precipitation_mm_day"]
                for b, c in zip(b_rows, c_rows)
            ])),
        }
    return result


def main() -> int:
    config = RealTerrainValidationConfig(height=32, width=64)
    common = dict(
        enable_force_restore_land=True,
        enable_force_restore_atmospheric_heat_convergence=True,
    )
    control_params = dataclasses.replace(EARTH, **common)
    candidate_params = dataclasses.replace(
        EARTH, **common, enable_force_restore_boundary_layer=True,
        enable_boundary_layer_stability_dependent_exchange=True,
        enable_boundary_layer_horizontal_transport=True,
    )
    _, control = _run(control_params, config, diagnose=False)
    _, candidate = _run(candidate_params, config, diagnose=True)
    delta_p = mixed_layer_pressure_thickness(
        surface_pressure_pa=float(EARTH.surface_pressure_pa),
        gravity_m_s2=float(EARTH.surface_gravity),
        gas_constant_j_kg_k=float(EARTH.gas_constant_dry),
        reference_temperature_k=288.15,
        mixed_layer_depth_m=float(candidate_params.boundary_layer_mixed_depth_m),
    )
    report = {
        "schema_version": 1,
        "grid": [config.height, config.width],
        "spinup_years": 1,
        "evaluation_years": 1,
        "contract": {
            "mixed_layer_depth_m": candidate_params.boundary_layer_mixed_depth_m,
            "pressure_thickness_pa": delta_p,
            "heat_capacity_j_m2_k": delta_p * EARTH.cp_dry / EARTH.surface_gravity,
            "entrainment_velocity_m_s": candidate_params.boundary_layer_entrainment_velocity_m_s,
        },
        "regional_summary": _summary(control, candidate),
        "control_monthly": control,
        "candidate_monthly": candidate,
    }
    output = ROOT / "temp" / "boundary_layer_budget_32x64.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report["contract"], "regions": report["regional_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
