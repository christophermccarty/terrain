"""Short finite-state and timestep screen for the experimental mixed layer."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import EARTH  # noqa: E402
from simulate import clear_simulation_caches, create_initial_state, simulate_step  # noqa: E402


def _advance(initial, *, dt_days: float, total_days: float, params):
    state = initial
    max_surface_closure = 0.0
    max_exchange_closure = 0.0
    max_transport_area_mean = 0.0
    max_continuity_closure = 0.0
    for _ in range(int(round(total_days / dt_days))):
        state, diagnostics = simulate_step(
            state, days=dt_days, planet_params=params,
            track_components=True, update_wind=False,
        )
        surface = diagnostics["land_air_sensible_exchange_closure_w_m2"]
        exchange = (
            diagnostics["boundary_layer_exchange_gain_w_m2"]
            + diagnostics["free_air_exchange_gain_w_m2"]
        )
        max_surface_closure = max(max_surface_closure, float(np.max(np.abs(surface))))
        max_exchange_closure = max(max_exchange_closure, float(np.max(np.abs(exchange))))
        horizontal = diagnostics.get("boundary_layer_horizontal_heat_convergence_w_m2")
        if horizontal is not None:
            area_mean = diagnostics[
                "boundary_layer_horizontal_convergence_area_mean_w_m2"
            ]
            max_transport_area_mean = max(max_transport_area_mean, abs(float(area_mean)))
            continuity_closure = (
                diagnostics["boundary_layer_continuity_exchange_gain_w_m2"]
                + diagnostics["free_air_continuity_exchange_gain_w_m2"]
            )
            max_continuity_closure = max(
                max_continuity_closure, float(np.max(np.abs(continuity_closure)))
            )
    return (
        state, max_surface_closure, max_exchange_closure,
        max_transport_area_mean, max_continuity_closure,
    )


def main() -> int:
    height, width = 32, 64
    lat = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    lon = np.linspace(0.0, 2.0 * np.pi, width, endpoint=False, dtype=np.float32)[None, :]
    elevation = np.where(
        np.sin(2.0 * lon) + 0.35 * np.cos(3.0 * lon) - 0.2 * lat > 0.15,
        0.35 + 0.25 * np.cos(lat * np.pi),
        0.0,
    ).astype(np.float32)
    params = dataclasses.replace(
        EARTH,
        enable_force_restore_land=True,
        enable_force_restore_atmospheric_heat_convergence=True,
        enable_force_restore_boundary_layer=True,
        enable_boundary_layer_stability_dependent_exchange=True,
        enable_boundary_layer_horizontal_transport=True,
    )
    clear_simulation_caches()
    initial = create_initial_state(elevation, planet_params=params)
    duration_days = 1.0
    one_day, surface_1d, exchange_1d, transport_1d, continuity_1d = _advance(
        initial, dt_days=1.0, total_days=duration_days, params=params
    )
    clear_simulation_caches()
    half_day, surface_half, exchange_half, transport_half, continuity_half = _advance(
        initial, dt_days=0.5, total_days=duration_days, params=params
    )
    control_params = dataclasses.replace(params, enable_force_restore_boundary_layer=False)
    clear_simulation_caches()
    control_initial = create_initial_state(elevation, planet_params=control_params)
    control_full = control_initial
    for _ in range(int(round(duration_days))):
        control_full, _ = simulate_step(
            control_full, days=1.0, planet_params=control_params, update_wind=False
        )
    clear_simulation_caches()
    control_half = control_initial
    for _ in range(int(round(duration_days / 0.5))):
        control_half, _ = simulate_step(
            control_half, days=0.5, planet_params=control_params, update_wind=False
        )
    fields = (
        one_day.temperature, one_day.air_temperature,
        one_day.boundary_layer_temperature, half_day.temperature,
        half_day.air_temperature, half_day.boundary_layer_temperature,
    )
    finite = all(field is not None and np.all(np.isfinite(field)) for field in fields)
    land = elevation > 0.0
    split_rms = float(np.sqrt(np.mean(
        (one_day.boundary_layer_temperature[land]
         - half_day.boundary_layer_temperature[land]) ** 2
    )))
    control_split_rms = float(np.sqrt(np.mean(
        (control_full.air_temperature[land] - control_half.air_temperature[land]) ** 2
    )))
    extrema = {
        "surface_k": [float(np.min(one_day.temperature)), float(np.max(one_day.temperature))],
        "free_air_k": [float(np.min(one_day.air_temperature)), float(np.max(one_day.air_temperature))],
        "boundary_layer_k": [
            float(np.min(one_day.boundary_layer_temperature[land])),
            float(np.max(one_day.boundary_layer_temperature[land])),
        ],
    }
    passed = bool(
        finite
        and max(surface_1d, surface_half, exchange_1d, exchange_half) < 1e-5
        and max(transport_1d, transport_half, continuity_1d, continuity_half) < 1e-5
        and split_rms <= max(0.5, 1.25 * control_split_rms)
        and extrema["boundary_layer_k"][0] > 180.0
        and extrema["boundary_layer_k"][1] < 340.0
    )
    report = {
        "passed": passed,
        "grid": [height, width],
        "duration_days": duration_days,
        "finite": finite,
        "max_surface_exchange_closure_w_m2": max(surface_1d, surface_half),
        "max_internal_exchange_closure_w_m2": max(exchange_1d, exchange_half),
        "max_horizontal_transport_area_mean_w_m2": max(transport_1d, transport_half),
        "max_continuity_exchange_closure_w_m2": max(continuity_1d, continuity_half),
        "one_day_vs_half_day_boundary_layer_rms_k": split_rms,
        "resolved_control_one_day_vs_half_day_air_rms_k": control_split_rms,
        "one_day_extrema": extrema,
    }
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
