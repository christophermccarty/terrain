"""Audit physical atmospheric storage and air--sea residuals for Phase 3."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundary_layer import (  # noqa: E402
    mixed_layer_pressure_thickness,
    overlying_layer_pressure_thickness,
)
from masks import get_masks  # noqa: E402
from planet_params import EARTH  # noqa: E402
from real_terrain_validation import RealTerrainValidationConfig, load_bundled_earth_dem  # noqa: E402
from simulate import (  # noqa: E402
    _ocean_mixed_layer_depth,
    clear_simulation_caches,
    create_initial_state,
    simulate_step,
)
from simulation_state import TimeScaleMode  # noqa: E402
from time_policy import substeps_for_mode  # noqa: E402


def _area_weights(shape):
    height, width = shape
    edges = np.linspace(np.pi / 2.0, -np.pi / 2.0, height + 1)
    rows = np.sin(edges[:-1]) - np.sin(edges[1:])
    weights = np.broadcast_to(rows[:, None], shape)
    return weights, float(width * np.sum(rows))


def _lat_2d(shape):
    height, width = shape
    edges = np.linspace(np.pi / 2.0, -np.pi / 2.0, height + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.broadcast_to(centers[:, None], shape)


def _surface_and_deep_energy_area_mean(state, params, land, sea, lat_2d, weights, denominator):
    """Land skin + ocean mixed layer + land-deep + deep-ocean energy, J m^-2 area mean.

    This is the reservoir energy the host owns *outside* the atmospheric column
    (as measured by ``_atmospheric_energy_area_mean``). Summing this with the
    atmospheric energy gives the full coupled-system storage, which must equal
    the time integral of TOA net radiation for the model to be conservative --
    any residual is an implicit (non-radiative) energy source or sink.
    """
    surface_temperature = np.asarray(state.temperature, dtype=np.float64)
    mld = _ocean_mixed_layer_depth(lat_2d, state.day_of_year, params)
    ocean_capacity = 4.186e6 * np.asarray(mld, dtype=np.float64)
    land_capacity = float(params.land_surface_heat_capacity_j_m2_k)
    energy = np.where(sea, ocean_capacity, land_capacity) * surface_temperature
    if state.land_deep_temperature is not None:
        land_deep_capacity = float(params.land_deep_heat_capacity_j_m2_k)
        energy = energy + np.where(
            land,
            land_deep_capacity * np.asarray(state.land_deep_temperature, dtype=np.float64),
            0.0,
        )
    if state.T_deep_ocean is not None:
        deep_ocean_capacity = ocean_capacity / float(params.deep_ocean_heat_capacity_ratio)
        energy = energy + np.where(
            sea,
            deep_ocean_capacity * np.asarray(state.T_deep_ocean, dtype=np.float64),
            0.0,
        )
    return float(np.sum(energy * weights) / denominator)


def _atmospheric_energy_area_mean(state, params, land, weights, denominator):
    air = np.asarray(state.air_temperature, dtype=np.float64)
    column_capacity = params.surface_pressure_pa / params.surface_gravity * params.cp_dry
    if (
        params.enable_coupled_two_layer_grey_radiation
        and state.midlevel_temperature is not None
        and state.upperlevel_temperature is not None
    ):
        fractions = (0.40, 0.35, 0.25)
        temperatures = (
            air,
            np.asarray(state.midlevel_temperature, dtype=np.float64),
            np.asarray(state.upperlevel_temperature, dtype=np.float64),
        )
        humidities = (
            np.zeros_like(air) if state.humidity is None else np.asarray(state.humidity),
            np.zeros_like(air) if state.midlevel_humidity is None else np.asarray(state.midlevel_humidity),
            np.zeros_like(air) if state.upperlevel_humidity is None else np.asarray(state.upperlevel_humidity),
        )
        column_mass = params.surface_pressure_pa / params.surface_gravity
        energy = sum(
            fraction * column_mass * (
                params.cp_dry * temperature + 2.5e6 * humidity
            )
            for fraction, temperature, humidity in zip(
                fractions, temperatures, humidities, strict=True
            )
        )
        if (
            params.enable_force_restore_boundary_layer
            and state.boundary_layer_temperature is not None
        ):
            delta_p = mixed_layer_pressure_thickness(
                surface_pressure_pa=params.surface_pressure_pa,
                gravity_m_s2=params.surface_gravity,
                gas_constant_j_kg_k=params.gas_constant_dry,
                reference_temperature_k=288.15,
                mixed_layer_depth_m=params.boundary_layer_mixed_depth_m,
            )
            energy = energy + np.where(
                land,
                delta_p / params.surface_gravity * params.cp_dry
                * (np.asarray(state.boundary_layer_temperature) - air),
                0.0,
            )
        return float(np.sum(energy * weights) / denominator)
    energy = column_capacity * air
    if (
        params.enable_force_restore_land
        and params.enable_force_restore_boundary_layer
        and state.boundary_layer_temperature is not None
    ):
        delta_p = mixed_layer_pressure_thickness(
            surface_pressure_pa=params.surface_pressure_pa,
            gravity_m_s2=params.surface_gravity,
            gas_constant_j_kg_k=params.gas_constant_dry,
            reference_temperature_k=288.15,
            mixed_layer_depth_m=params.boundary_layer_mixed_depth_m,
        )
        boundary_capacity = delta_p / params.surface_gravity * params.cp_dry
        land_energy = (
            boundary_capacity * np.asarray(state.boundary_layer_temperature)
            + (column_capacity - boundary_capacity) * air
        )
        if (
            params.enable_boundary_layer_interface_reservoir
            and state.boundary_layer_interface_temperature is not None
        ):
            interface_delta_p = overlying_layer_pressure_thickness(
                surface_pressure_pa=params.surface_pressure_pa,
                gravity_m_s2=params.surface_gravity,
                gas_constant_j_kg_k=params.gas_constant_dry,
                reference_temperature_k=288.15,
                layer_base_m=params.boundary_layer_mixed_depth_m,
                layer_depth_m=params.boundary_layer_mixed_depth_m,
            )
            interface_capacity = (
                interface_delta_p / params.surface_gravity * params.cp_dry
            )
            land_energy = (
                boundary_capacity * np.asarray(state.boundary_layer_temperature)
                + interface_capacity
                * np.asarray(state.boundary_layer_interface_temperature)
                + (column_capacity - boundary_capacity - interface_capacity) * air
            )
        energy = np.where(land, land_energy, energy)
    return float(np.sum(energy * weights) / denominator)


def _run(params, config):
    clear_simulation_caches()
    warmup_params = (
        dataclasses.replace(
            params,
            enable_coupled_two_layer_grey_radiation=False,
            enable_prognostic_column_water=False,
            enable_stability_aware_condensation=False,
            enable_two_layer_convective_adjustment=False,
            enable_three_level_pressure_column=False,
            enable_closed_three_level_thermodynamics=False,
        )
        if params.enable_coupled_two_layer_grey_radiation
        else params
    )
    elevation = load_bundled_earth_dem(config.height, config.width)
    sea, land = get_masks(elevation)
    weights, denominator = _area_weights(elevation.shape)
    lat_2d = _lat_2d(elevation.shape)
    state = create_initial_state(
        elevation, day_of_year=config.start_day, planet_params=warmup_params,
        block_size=config.block_size, wind_block_size=config.wind_block_size,
        precip_block_size=config.precip_block_size,
    )
    mode = TimeScaleMode[config.time_scale]
    schedule = substeps_for_mode(mode, params)
    for _ in range(12):
        for days, update_wind in schedule:
            state, _ = simulate_step(
                state, days=days, block_size=config.block_size,
                wind_block_size=config.wind_block_size,
                precip_block_size=config.precip_block_size,
                update_wind=update_wind, time_scale=mode,
                planet_params=warmup_params, track_components=False,
            )

    sums = {
        "atmospheric_storage_w_m2": 0.0,
        "surface_and_deep_storage_w_m2": 0.0,
        "full_system_storage_w_m2": 0.0,
        "airsea_atmospheric_gain_w_m2": 0.0,
        "airsea_ocean_gain_w_m2": 0.0,
        "airsea_physical_residual_w_m2": 0.0,
        "ocean_air_relaxation_unopposed_gain_w_m2": 0.0,
        "ocean_air_relaxation_ocean_gain_w_m2": 0.0,
        "ocean_air_relaxation_physical_residual_w_m2": 0.0,
        "net_radiation_w_m2": 0.0,
        "free_air_advection_gain_w_m2": 0.0,
        "free_air_diffusion_gain_w_m2": 0.0,
        "free_air_final_clamp_gain_w_m2": 0.0,
        "boundary_layer_final_clamp_gain_w_m2": 0.0,
        "surface_air_relaxation_gain_w_m2": 0.0,
        "boundary_layer_surface_sensible_gain_w_m2": 0.0,
        "cloud_shortwave_forcing_w_m2": 0.0,
        "cloud_longwave_forcing_w_m2": 0.0,
        "cloud_net_radiative_forcing_w_m2": 0.0,
        "cloud_specific_humidity": 0.0,
        "cloud_saturation_specific_humidity": 0.0,
        "cloud_relative_humidity": 0.0,
        "cloud_rh_above_065_area_fraction": 0.0,
        "land_cloud_relative_humidity": 0.0,
        "ocean_cloud_relative_humidity": 0.0,
        "cloud_rh_core": 0.0,
        "cloud_ascent_term": 0.0,
        "cloud_subsidence": 0.0,
        "cloud_orographic_driver": 0.0,
        "cloud_instantaneous": 0.0,
        "cloud_after_persistence": 0.0,
        "cloud_after_rainout": 0.0,
        "cloud_final": 0.0,
        "boundary_layer_exchange_gain_land_mean_w_m2": 0.0,
        "boundary_layer_effective_entrainment_land_mean_m_s": 0.0,
        "boundary_layer_bulk_ri_land_mean": 0.0,
        "boundary_layer_stable_area_fraction": 0.0,
        "boundary_layer_strongly_stable_area_fraction": 0.0,
        "boundary_layer_post_exchange_inversion_land_mean_k": 0.0,
        "boundary_layer_post_exchange_abs_inversion_land_mean_k": 0.0,
        "boundary_layer_mechanical_entrainment_land_mean_m_s": 0.0,
        "boundary_layer_convective_entrainment_land_mean_m_s": 0.0,
        "boundary_layer_surface_buoyancy_flux_land_mean_m2_s3": 0.0,
        "absorbed_shortwave_w_m2": 0.0,
        "outgoing_longwave_w_m2": 0.0,
        "effective_radiating_temperature_k": 0.0,
        "free_air_minus_effective_radiating_temperature_k": 0.0,
        "grey_emission_temperature_k": 0.0,
        "grey_upperlevel_temperature_k": 0.0,
        "free_air_minus_grey_emission_temperature_k": 0.0,
        "grey_surface_gain_w_m2": 0.0,
        "grey_midlevel_gain_w_m2": 0.0,
        "grey_upperlevel_gain_w_m2": 0.0,
        "grey_atmospheric_gain_w_m2": 0.0,
        "grey_downward_longwave_at_surface_w_m2": 0.0,
        "grey_outgoing_longwave_w_m2": 0.0,
        "grey_toa_net_radiation_w_m2": 0.0,
        "grey_total_optical_depth": 0.0,
        "grey_opaque_limited_area_fraction": 0.0,
        "grey_target_olr_residual_w_m2": 0.0,
    }
    seconds = 0.0
    start_energy = _atmospheric_energy_area_mean(
        state, params, land, weights, denominator
    )
    start_surface_energy = _surface_and_deep_energy_area_mean(
        state, params, land, sea, lat_2d, weights, denominator
    )
    evaluation_mode = (
        TimeScaleMode.DAILY
        if params.enable_coupled_two_layer_grey_radiation
        else mode
    )
    evaluation_schedule = (
        ((1.0, True),)
        if params.enable_coupled_two_layer_grey_radiation
        else schedule
    )
    evaluation_cycles = (
        int(round(params.orbital_period_days))
        if params.enable_coupled_two_layer_grey_radiation
        else 12
    )
    for _ in range(evaluation_cycles):
        for days, update_wind in evaluation_schedule:
            before = _atmospheric_energy_area_mean(
                state, params, land, weights, denominator
            )
            before_surface = _surface_and_deep_energy_area_mean(
                state, params, land, sea, lat_2d, weights, denominator
            )
            state, components = simulate_step(
                state, days=days, block_size=config.block_size,
                wind_block_size=config.wind_block_size,
                precip_block_size=config.precip_block_size,
                update_wind=update_wind, time_scale=evaluation_mode,
                planet_params=params, track_components=True,
            )
            dt = days * 86_400.0
            after = _atmospheric_energy_area_mean(
                state, params, land, weights, denominator
            )
            after_surface = _surface_and_deep_energy_area_mean(
                state, params, land, sea, lat_2d, weights, denominator
            )
            sums["atmospheric_storage_w_m2"] += (after - before)
            sums["surface_and_deep_storage_w_m2"] += (after_surface - before_surface)
            sums["full_system_storage_w_m2"] += (
                (after - before) + (after_surface - before_surface)
            )
            sums["airsea_atmospheric_gain_w_m2"] += (
                components["airsea_atmospheric_gain_area_mean_w_m2"] * dt
            )
            sums["airsea_ocean_gain_w_m2"] += (
                components["airsea_ocean_gain_area_mean_w_m2"] * dt
            )
            sums["airsea_physical_residual_w_m2"] += (
                components["airsea_physical_energy_residual_area_mean_w_m2"] * dt
            )
            sums["ocean_air_relaxation_unopposed_gain_w_m2"] += (
                components["ocean_air_relaxation_atmospheric_gain_area_mean_w_m2"] * dt
            )
            sums["ocean_air_relaxation_ocean_gain_w_m2"] += (
                components["ocean_air_relaxation_ocean_gain_area_mean_w_m2"] * dt
            )
            sums["ocean_air_relaxation_physical_residual_w_m2"] += (
                components[
                    "ocean_air_relaxation_physical_energy_residual_area_mean_w_m2"
                ] * dt
            )
            sums["net_radiation_w_m2"] += components["net_radiation_area_mean_w_m2"] * dt
            for key in (
                "free_air_advection_gain_w_m2",
                "free_air_diffusion_gain_w_m2",
                "free_air_final_clamp_gain_w_m2",
                "boundary_layer_final_clamp_gain_w_m2",
                "surface_air_relaxation_gain_w_m2",
                "cloud_shortwave_forcing_w_m2",
                "cloud_longwave_forcing_w_m2",
                "cloud_net_radiative_forcing_w_m2",
            ):
                component_key = key.replace("_w_m2", "_area_mean_w_m2")
                sums[key] += components[component_key] * dt
            sums["boundary_layer_surface_sensible_gain_w_m2"] += (
                components.get(
                    "boundary_layer_surface_sensible_gain_area_mean_w_m2", 0.0
                ) * dt
            )
            for target, component_key in (
                ("cloud_specific_humidity", "cloud_specific_humidity_area_mean"),
                (
                    "cloud_saturation_specific_humidity",
                    "cloud_saturation_specific_humidity_area_mean",
                ),
                ("cloud_relative_humidity", "cloud_relative_humidity_area_mean"),
                (
                    "cloud_rh_above_065_area_fraction",
                    "cloud_rh_above_065_area_fraction",
                ),
                (
                    "land_cloud_relative_humidity",
                    "land_cloud_relative_humidity_area_mean",
                ),
                (
                    "ocean_cloud_relative_humidity",
                    "ocean_cloud_relative_humidity_area_mean",
                ),
                ("cloud_rh_core", "cloud_rh_core_area_mean"),
                ("cloud_ascent_term", "cloud_ascent_term_area_mean"),
                ("cloud_subsidence", "cloud_subsidence_area_mean"),
                ("cloud_orographic_driver", "cloud_orographic_driver_area_mean"),
                ("cloud_instantaneous", "cloud_instantaneous_area_mean"),
                (
                    "cloud_after_persistence",
                    "cloud_after_persistence_area_mean",
                ),
                ("cloud_after_rainout", "cloud_after_rainout_area_mean"),
                ("cloud_final", "cloud_final_area_mean"),
                (
                    "boundary_layer_exchange_gain_land_mean_w_m2",
                    "boundary_layer_exchange_gain_land_mean_w_m2",
                ),
                (
                    "boundary_layer_effective_entrainment_land_mean_m_s",
                    "boundary_layer_effective_entrainment_land_mean_m_s",
                ),
                (
                    "boundary_layer_bulk_ri_land_mean",
                    "boundary_layer_bulk_ri_land_mean",
                ),
                (
                    "boundary_layer_stable_area_fraction",
                    "boundary_layer_stable_area_fraction",
                ),
                (
                    "boundary_layer_strongly_stable_area_fraction",
                    "boundary_layer_strongly_stable_area_fraction",
                ),
                (
                    "boundary_layer_post_exchange_inversion_land_mean_k",
                    "boundary_layer_post_exchange_inversion_land_mean_k",
                ),
                (
                    "boundary_layer_post_exchange_abs_inversion_land_mean_k",
                    "boundary_layer_post_exchange_abs_inversion_land_mean_k",
                ),
                (
                    "boundary_layer_mechanical_entrainment_land_mean_m_s",
                    "boundary_layer_mechanical_entrainment_land_mean_m_s",
                ),
                (
                    "boundary_layer_convective_entrainment_land_mean_m_s",
                    "boundary_layer_convective_entrainment_land_mean_m_s",
                ),
                (
                    "boundary_layer_surface_buoyancy_flux_land_mean_m2_s3",
                    "boundary_layer_surface_buoyancy_flux_land_mean_m2_s3",
                ),
                ("absorbed_shortwave_w_m2", "absorbed_shortwave_area_mean_w_m2"),
                ("outgoing_longwave_w_m2", "outgoing_longwave_area_mean_w_m2"),
                (
                    "effective_radiating_temperature_k",
                    "effective_radiating_temperature_area_mean_k",
                ),
                (
                    "free_air_minus_effective_radiating_temperature_k",
                    "free_air_minus_effective_radiating_temperature_area_mean_k",
                ),
                (
                    "grey_emission_temperature_k",
                    "grey_emission_temperature_area_mean_k",
                ),
                (
                    "grey_upperlevel_temperature_k",
                    "grey_upperlevel_temperature_area_mean_k",
                ),
                (
                    "free_air_minus_grey_emission_temperature_k",
                    "grey_free_air_minus_emission_temperature_area_mean_k",
                ),
                ("grey_surface_gain_w_m2", "grey_surface_gain_area_mean_w_m2"),
                ("grey_midlevel_gain_w_m2", "grey_midlevel_gain_area_mean_w_m2"),
                ("grey_upperlevel_gain_w_m2", "grey_upperlevel_gain_area_mean_w_m2"),
                (
                    "grey_atmospheric_gain_w_m2",
                    "grey_atmospheric_gain_area_mean_w_m2",
                ),
                (
                    "grey_downward_longwave_at_surface_w_m2",
                    "grey_downward_longwave_at_surface_area_mean_w_m2",
                ),
                (
                    "grey_outgoing_longwave_w_m2",
                    "grey_outgoing_longwave_area_mean_w_m2",
                ),
                (
                    "grey_toa_net_radiation_w_m2",
                    "grey_toa_net_radiation_area_mean_w_m2",
                ),
                ("grey_total_optical_depth", "grey_total_optical_depth_area_mean"),
                (
                    "grey_opaque_limited_area_fraction",
                    "grey_opaque_limited_area_fraction",
                ),
                (
                    "grey_target_olr_residual_w_m2",
                    "grey_target_olr_residual_area_mean_w_m2",
                ),
            ):
                sums[target] += components.get(component_key, 0.0) * dt
            seconds += dt
    end_energy = _atmospheric_energy_area_mean(state, params, land, weights, denominator)
    end_surface_energy = _surface_and_deep_energy_area_mean(
        state, params, land, sea, lat_2d, weights, denominator
    )
    # Storage was accumulated as J m-2; the other entries accumulated W m-2 s.
    _storage_keys = {
        "atmospheric_storage_w_m2",
        "surface_and_deep_storage_w_m2",
        "full_system_storage_w_m2",
    }
    for key in sums:
        if key not in _storage_keys:
            sums[key] /= seconds
    for key in _storage_keys:
        sums[key] /= seconds
    sums["storage_endpoint_check_w_m2"] = (end_energy - start_energy) / seconds
    sums["surface_storage_endpoint_check_w_m2"] = (
        (end_surface_energy - start_surface_energy) / seconds
    )
    sums["full_system_storage_endpoint_check_w_m2"] = (
        sums["storage_endpoint_check_w_m2"]
        + sums["surface_storage_endpoint_check_w_m2"]
    )
    sums["implicit_energy_source_w_m2"] = (
        sums["full_system_storage_endpoint_check_w_m2"]
        - sums["grey_toa_net_radiation_w_m2"]
    )
    sums["final_global_air_temperature_k"] = float(
        np.sum(np.asarray(state.air_temperature) * weights) / denominator
    )
    sums["final_global_surface_temperature_k"] = float(
        np.sum(np.asarray(state.temperature) * weights) / denominator
    )
    return sums


def main() -> int:
    config = RealTerrainValidationConfig(height=32, width=64)
    common = dict(
        enable_force_restore_land=True,
        enable_force_restore_atmospheric_heat_convergence=True,
    )
    control = dataclasses.replace(EARTH, **common)
    candidate = dataclasses.replace(
        EARTH, **common,
        enable_force_restore_boundary_layer=True,
        enable_boundary_layer_horizontal_transport=True,
        enable_boundary_layer_interface_reservoir=True,
        enable_boundary_layer_capacity_aware_airsea_exchange=True,
        enable_boundary_layer_capacity_aware_free_air_transport=True,
        enable_boundary_layer_near_surface_cloud_temperature=True,
        enable_boundary_layer_split_invariant_cloud_memory=True,
        enable_pressure_defined_radiative_temperature_profile=True,
        enable_coupled_two_layer_grey_radiation=True,
        enable_prognostic_column_water=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
        enable_closed_three_level_thermodynamics=True,
        enable_diabatic_interface_mass_flux=True,
    )
    report = {
        "schema_version": 1,
        "grid": [config.height, config.width],
        "control": _run(control, config),
        "candidate": _run(candidate, config),
    }
    report["candidate_minus_control"] = {
        key: report["candidate"][key] - report["control"][key]
        for key in report["control"]
    }
    grey_residual = (
        report["candidate"]["grey_surface_gain_w_m2"]
        + report["candidate"]["grey_atmospheric_gain_w_m2"]
        - report["candidate"]["grey_toa_net_radiation_w_m2"]
    )
    resolved_gap = report["candidate"][
        "free_air_minus_grey_emission_temperature_k"
    ]
    toa_net = report["candidate"]["grey_toa_net_radiation_w_m2"]
    olr_residual = (
        report["candidate"]["grey_outgoing_longwave_w_m2"]
        - report["candidate"]["outgoing_longwave_w_m2"]
    )
    air_shock = report["candidate_minus_control"]["final_global_air_temperature_k"]
    surface_shock = report["candidate_minus_control"]["final_global_surface_temperature_k"]
    admitted = all((
        abs(grey_residual) <= 1.0e-9,
        resolved_gap > 0.0,
        np.isfinite(report["candidate"]["grey_emission_temperature_k"]),
        abs(toa_net) <= 5.0,
        abs(olr_residual) <= 2.0,
        abs(air_shock) <= 5.0,
        abs(surface_shock) <= 5.0,
    ))
    report["radiative_profile_admission"] = {
        "grey_column_conservation_residual_w_m2": grey_residual,
        "grey_column_conservation_pass": abs(grey_residual) <= 1.0e-9,
        "resolved_emission_temperature_below_free_air": resolved_gap > 0.0,
        "resolved_emission_temperature_finite": bool(np.isfinite(
            report["candidate"]["grey_emission_temperature_k"]
        )),
        "grey_matches_existing_olr_residual_w_m2": olr_residual,
        "toa_balance_pass": abs(toa_net) <= 5.0,
        "olr_handoff_pass": abs(olr_residual) <= 2.0,
        "air_temperature_shock_k": air_shock,
        "surface_temperature_shock_k": surface_shock,
        "temperature_shock_pass": abs(air_shock) <= 5.0 and abs(surface_shock) <= 5.0,
        "coupling_status": "admitted" if admitted else "rejected",
    }
    output = ROOT / "temp" / "boundary_layer_column_energy_32x64.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
