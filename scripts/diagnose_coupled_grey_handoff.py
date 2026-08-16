"""Read-only diagnostic for the coupled two-layer grey-radiation handoff.

Context: ``docs/VERTICAL_THERMODYNAMIC_CLOSURE.md`` (2026-08-15) rejected the
``enable_coupled_two_layer_grey_radiation`` candidate at 32x64 because the
handoff bootstraps mid/upper-level temperatures from
``pressure_defined_temperature_profile`` -- an exact dry adiabat, i.e. zero
potential-temperature gradient by construction -- while
``diabatic_interface_mass_flux`` divides a heating anomaly by that gradient
with no floor.  The recorded failure trace is a vertical Courant number
roughly doubling daily (0.069 -> 0.804 -> 1.733 -> 3.48 over the first four
days after the handoff).

This script measures that pathway where it acts, without changing any
simulation code:

1. It reproduces the admission screen's warmup exactly (12 MONTHLY cycles at
   32x64 with the closed-column family off and the pressure-defined profile
   on), so the handoff state is the same one the screen produces.
2. At the handoff it records the bootstrapped state: the zonal
   potential-temperature stability at both interfaces, the area fraction near
   neutral, and the closed-form two-layer grey *radiative-equilibrium*
   mid/upper profile implied by the persisted optical depth (the target a
   grey-aware initialization would move toward).
3. It then runs the coupled gate DAILY and traces per-day temperatures,
   stability, the exactly recomputed diabatic omega/Courant response, grey
   budget components, and the full-system implicit energy source used by
   ``audit_boundary_layer_column_energy.py``.

It writes ``temp/coupled_grey_handoff_32x64.json`` and prints a day-by-day
table.  No simulation default or experimental gate is modified.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from audit_boundary_layer_column_energy import (  # noqa: E402
    _atmospheric_energy_area_mean,
    _lat_2d,
    _area_weights,
    _surface_and_deep_energy_area_mean,
)
from atmospheric_radiation import (  # noqa: E402
    STEFAN_BOLTZMANN,
    pressure_split_emissivities_from_optical_depth,
)
from masks import get_masks  # noqa: E402
from planet_params import EARTH  # noqa: E402
from pressure_circulation import (  # noqa: E402
    diabatic_interface_mass_flux,
    diabatic_interface_mass_flux_from_heating,
)
from real_terrain_validation import (  # noqa: E402
    RealTerrainValidationConfig,
    load_bundled_earth_dem,
)
from simulate import (  # noqa: E402
    clear_simulation_caches,
    create_initial_state,
    simulate_step,
)
from simulation_state import TimeScaleMode  # noqa: E402
from time_policy import substeps_for_mode  # noqa: E402

# The 0.40/0.35/0.25 pressure-mass partition is a fixed contract of the
# three-level closure family (see pressure_circulation.py).
_LAYER_MASS_FRACTIONS = (0.40, 0.35, 0.25)

# Diagnostic-only thresholds for "near neutral" static stability [K/Pa].
# A normally stratified tropical troposphere gives O(1e-4..1e-3) K/Pa between
# these layer centres; the dry-adiabatic bootstrap gives ~0 by construction.
_STABILITY_THRESHOLDS_K_PA = (1.0e-5, 5.0e-5, 1.0e-4)


def _candidate_params():
    """Mirror audit_boundary_layer_column_energy.py's candidate gate set."""
    common = dict(
        enable_force_restore_land=True,
        enable_force_restore_atmospheric_heat_convergence=True,
    )
    return dataclasses.replace(
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


def _warmup_params(params):
    """Mirror the audit's warmup: coupled family off, profile gate kept on."""
    return dataclasses.replace(
        params,
        enable_coupled_two_layer_grey_radiation=False,
        enable_prognostic_column_water=False,
        enable_stability_aware_condensation=False,
        enable_two_layer_convective_adjustment=False,
        enable_three_level_pressure_column=False,
        enable_closed_three_level_thermodynamics=False,
    )


def _zonal_interface_stability(lower, middle, upper, params):
    """Zonal potential-temperature gradient at both interfaces [K/Pa].

    Mirrors the denominator inside ``diabatic_interface_mass_flux_from_heating``
    (pressure_circulation.py): zonal means first, layer-centre pressures from
    the fixed 0.40/0.35/0.25 partition, kappa = R_d/c_p.  Pure diagnostic; any
    change to that operator's stability definition must be reflected here.
    """
    fractions = _LAYER_MASS_FRACTIONS
    p_s = float(params.surface_pressure_pa)
    edges = p_s * np.array((1.0, 1.0 - fractions[0], fractions[2], 0.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    kappa = 287.05 / float(params.cp_dry)

    def _theta(layer_temperature, pressure):
        zonal = np.mean(np.asarray(layer_temperature, dtype=np.float64), axis=1)
        return zonal * (p_s / pressure) ** kappa

    theta_lower = _theta(lower, centers[0])
    theta_middle = _theta(middle, centers[1])
    theta_upper = _theta(upper, centers[2])
    stability_lower_mid = (
        (theta_middle - theta_lower) / float(params.two_layer_pressure_depth_pa)
    )
    stability_mid_upper = (
        (theta_upper - theta_middle)
        / float(params.three_level_mid_upper_pressure_depth_pa)
    )
    return stability_lower_mid, stability_mid_upper


def _stability_stats(stability, row_weights):
    """Area-weighted percentiles and near-neutral fractions for a zonal field."""
    values = np.asarray(stability, dtype=np.float64)
    weights = np.asarray(row_weights, dtype=np.float64)
    stats = {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "area_weighted_mean": float(
            np.sum(values * weights) / np.sum(weights)
        ),
    }
    for threshold in _STABILITY_THRESHOLDS_K_PA:
        fraction = float(
            np.sum(weights[values < threshold]) / np.sum(weights)
        )
        stats[f"area_fraction_below_{threshold:.0e}"] = fraction
    return stats


def _grey_equilibrium_temperatures(surface_temperature_k, emissivity_mid, emissivity_upper):
    """Closed-form two-layer grey radiative-equilibrium mid/upper temperatures.

    Sets the model's own layer gains to zero (``two_layer_grey_radiation``):
    ``2 B_m = E_s + eps_u B_u`` and ``2 B_u = (1 - eps_m) E_s + eps_m B_m``
    with ``B = sigma T^4``.  Linear in the blackbody fluxes, so the solution is
    exact and per-cell.  Absorbed shortwave passes through both layers and does
    not enter the atmospheric gains, so it is not an input here.
    """
    surface_emission = STEFAN_BOLTZMANN * np.asarray(surface_temperature_k, dtype=np.float64) ** 4
    eps_m = np.asarray(emissivity_mid, dtype=np.float64)
    eps_u = np.asarray(emissivity_upper, dtype=np.float64)
    middle_blackbody = (
        surface_emission * (2.0 + eps_u - eps_u * eps_m) / (4.0 - eps_m * eps_u)
    )
    upper_blackbody = (
        (1.0 - eps_m) * surface_emission + eps_m * middle_blackbody
    ) / 2.0
    return (
        (middle_blackbody / STEFAN_BOLTZMANN) ** 0.25,
        (upper_blackbody / STEFAN_BOLTZMANN) ** 0.25,
    )


def _global_stats(field):
    values = np.asarray(field, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _diabatic_response(state, params):
    """Exactly recompute the live path's diabatic omega/Courant from state.

    Mirrors atmosphere.py's branch: with no prognostic heating reservoir the
    precipitation-anomaly variant is used; otherwise the heating variant.
    """
    kwargs = dict(
        dt_seconds=86_400.0,
        surface_pressure_pa=float(params.surface_pressure_pa),
        lower_mid_pressure_depth_pa=float(params.two_layer_pressure_depth_pa),
        mid_upper_pressure_depth_pa=float(params.three_level_mid_upper_pressure_depth_pa),
        gravity_m_s2=float(params.surface_gravity),
        cp_dry_j_kg_k=float(params.cp_dry),
    )
    heating = state.pressure_overturning_heating_w_m2
    if heating is None:
        step = diabatic_interface_mass_flux(
            state.precipitation,
            state.air_temperature,
            state.midlevel_temperature,
            state.upperlevel_temperature,
            **kwargs,
        )
        branch = "precipitation_anomaly"
    else:
        step = diabatic_interface_mass_flux_from_heating(
            heating,
            state.air_temperature,
            state.midlevel_temperature,
            state.upperlevel_temperature,
            **kwargs,
        )
        branch = "heating_reservoir"
    return {
        "branch": branch,
        "omega_lower_mid_abs_max_pa_s": float(np.max(np.abs(step.omega_lower_mid_pa_s))),
        "omega_mid_upper_abs_max_pa_s": float(np.max(np.abs(step.omega_mid_upper_pa_s))),
        "omega_lower_mid_rms_pa_s": float(np.sqrt(np.mean(step.omega_lower_mid_pa_s.astype(np.float64) ** 2))),
        "omega_mid_upper_rms_pa_s": float(np.sqrt(np.mean(step.omega_mid_upper_pa_s.astype(np.float64) ** 2))),
        "lower_mid_vertical_courant_max": float(step.lower_mid_vertical_courant_max),
        "mid_upper_vertical_courant_max": float(step.mid_upper_vertical_courant_max),
    }


def _component(components, key):
    value = components.get(key)
    return None if value is None else float(value)


def _record(state, params, row_weights, components=None):
    record = {
        "air_temperature_k": _global_stats(state.air_temperature),
        "midlevel_temperature_k": _global_stats(state.midlevel_temperature),
        "upperlevel_temperature_k": _global_stats(state.upperlevel_temperature),
        "surface_temperature_k": _global_stats(state.temperature),
        "precipitation_mm_day": _global_stats(state.precipitation),
    }
    stability_lm, stability_mu = _zonal_interface_stability(
        state.air_temperature,
        state.midlevel_temperature,
        state.upperlevel_temperature,
        params,
    )
    record["stability_lower_mid_k_pa"] = _stability_stats(stability_lm, row_weights)
    record["stability_mid_upper_k_pa"] = _stability_stats(stability_mu, row_weights)
    record["diabatic_response"] = _diabatic_response(state, params)

    optical_depth = state.grey_optical_depth
    if optical_depth is not None:
        split = pressure_split_emissivities_from_optical_depth(
            np.asarray(optical_depth, dtype=np.float64),
            float(params.two_layer_pressure_depth_pa),
            float(params.three_level_mid_upper_pressure_depth_pa),
        )
        eq_mid, eq_upper = _grey_equilibrium_temperatures(
            state.temperature, split.midlevel_emissivity, split.upperlevel_emissivity
        )
        eq_stab_lm, eq_stab_mu = _zonal_interface_stability(
            state.air_temperature, eq_mid, eq_upper, params
        )
        record["grey_equilibrium"] = {
            "midlevel_temperature_k": _global_stats(eq_mid),
            "upperlevel_temperature_k": _global_stats(eq_upper),
            "bootstrap_minus_equilibrium_midlevel_k": _global_stats(
                np.asarray(state.midlevel_temperature, dtype=np.float64) - eq_mid
            ),
            "bootstrap_minus_equilibrium_upperlevel_k": _global_stats(
                np.asarray(state.upperlevel_temperature, dtype=np.float64) - eq_upper
            ),
            "stability_lower_mid_k_pa": _stability_stats(eq_stab_lm, row_weights),
            "stability_mid_upper_k_pa": _stability_stats(eq_stab_mu, row_weights),
        }
    if components is not None:
        record["grey_components"] = {
            "surface_gain_w_m2": _component(components, "grey_surface_gain_area_mean_w_m2"),
            "midlevel_gain_w_m2": _component(components, "grey_midlevel_gain_area_mean_w_m2"),
            "upperlevel_gain_w_m2": _component(components, "grey_upperlevel_gain_area_mean_w_m2"),
            "outgoing_longwave_w_m2": _component(components, "grey_outgoing_longwave_area_mean_w_m2"),
            "target_olr_residual_w_m2": _component(components, "grey_target_olr_residual_area_mean_w_m2"),
            "toa_net_radiation_w_m2": _component(components, "grey_toa_net_radiation_area_mean_w_m2"),
            "total_optical_depth": _component(components, "grey_total_optical_depth_area_mean"),
            "opaque_limited_area_fraction": _component(components, "grey_opaque_limited_area_fraction"),
        }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument(
        "--days", type=int, default=14,
        help="coupled DAILY steps traced after the handoff (default 14)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="JSON output path (default temp/coupled_grey_handoff_<grid>.json)",
    )
    args = parser.parse_args()

    config = RealTerrainValidationConfig(height=args.height, width=args.width)
    params = _candidate_params()
    warmup = _warmup_params(params)

    clear_simulation_caches()
    elevation = load_bundled_earth_dem(config.height, config.width)
    sea, land = get_masks(elevation)
    weights, denominator = _area_weights(elevation.shape)
    row_weights = weights[:, 0]
    lat_2d = _lat_2d(elevation.shape)

    # Warmup: identical to the admission screen (12 MONTHLY cycles with the
    # closed-column family off), producing the persisted optical depth and the
    # dry-adiabatic mid/upper bootstrap state.
    state = create_initial_state(
        elevation, day_of_year=config.start_day, planet_params=warmup,
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
                planet_params=warmup, track_components=False,
            )

    report = {
        "schema_version": 1,
        "grid": [config.height, config.width],
        "warmup": {"cycles": 12, "time_scale": config.time_scale},
        "handoff": _record(state, params, row_weights),
        "daily": [],
    }

    print(
        "day  cour_lm  cour_mu  om_max_lm  om_max_mu  T_air    T_mid min/max   "
        "stab_p5_lm  implicit_w_m2"
    )
    cumulative_toa_j_m2 = 0.0
    cumulative_storage_j_m2 = 0.0
    seconds = 0.0
    for day in range(1, args.days + 1):
        before_atmos = _atmospheric_energy_area_mean(
            state, params, land, weights, denominator
        )
        before_surface = _surface_and_deep_energy_area_mean(
            state, params, land, sea, lat_2d, weights, denominator
        )
        state, components = simulate_step(
            state, days=1.0, block_size=config.block_size,
            wind_block_size=config.wind_block_size,
            precip_block_size=config.precip_block_size,
            update_wind=True, time_scale=TimeScaleMode.DAILY,
            planet_params=params, track_components=True,
        )
        after_atmos = _atmospheric_energy_area_mean(
            state, params, land, weights, denominator
        )
        after_surface = _surface_and_deep_energy_area_mean(
            state, params, land, sea, lat_2d, weights, denominator
        )
        dt = 86_400.0
        toa_net = _component(components, "grey_toa_net_radiation_area_mean_w_m2")
        storage_rate = (
            (after_atmos - before_atmos) + (after_surface - before_surface)
        ) / dt
        cumulative_storage_j_m2 += (
            (after_atmos - before_atmos) + (after_surface - before_surface)
        )
        cumulative_toa_j_m2 += (toa_net or 0.0) * dt
        seconds += dt

        record = _record(state, params, row_weights, components=components)
        record["day"] = day
        record["energy"] = {
            "storage_rate_w_m2": storage_rate,
            "toa_net_w_m2": toa_net,
            "implicit_source_w_m2": None if toa_net is None else storage_rate - toa_net,
            "cumulative_implicit_source_w_m2": (
                (cumulative_storage_j_m2 - cumulative_toa_j_m2) / seconds
            ),
        }
        report["daily"].append(record)

        response = record["diabatic_response"]
        print(
            f"{day:3d}  "
            f"{response['lower_mid_vertical_courant_max']:7.3f}  "
            f"{response['mid_upper_vertical_courant_max']:7.3f}  "
            f"{response['omega_lower_mid_abs_max_pa_s']:9.3f}  "
            f"{response['omega_mid_upper_abs_max_pa_s']:9.3f}  "
            f"{record['air_temperature_k']['mean']:7.2f}  "
            f"{record['midlevel_temperature_k']['min']:7.2f}/"
            f"{record['midlevel_temperature_k']['max']:7.2f}  "
            f"{record['stability_lower_mid_k_pa']['p05']:10.2e}  "
            f"{record['energy']['implicit_source_w_m2'] or float('nan'):13.3f}"
        )

    output = args.output
    if output is None:
        output = (
            ROOT / "temp"
            / f"coupled_grey_handoff_{config.height}x{config.width}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
