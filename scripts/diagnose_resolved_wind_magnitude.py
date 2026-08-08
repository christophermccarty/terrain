"""Measure the raw resolved lower/middle/upper meridional wind magnitude.

Section 15 of docs/PRIOR_ART_IMPLEMENTATION_PLAN.md found that
`close_upper_mass_flux`'s own local divergence-closing behavior is unaffected
by whether overturning heating is present, and reframed the open transport
problem: a zero-net-mass-throughflow closure is not mechanically expected to
zero net energy transport (a real Hadley cell is exactly that: mass-neutral,
energy-transporting). The real anomaly is that this experimental family's
cross-equatorial transport (-14 to -31 PW across Sections 11-15) sits
roughly 3-5x above Earth's real ~5-6 PW peak Hadley-cell energy transport.

This script reads the raw lower/middle/upper meridional wind (`state.wind_v`,
`state.midlevel_wind_v`, `state.wind_v_aloft`) directly from a heated run's
final PlanetState -- no closure or transport-diagnostic code involved -- and
compares the zonal-mean tropical-band magnitude at each level against the
literature range for the Hadley cell's zonal-mean meridional wind (weak,
typically on the order of 1-3 m/s despite carrying a large mass/energy flux
over a deep pressure layer), to check whether the branch speeds feeding the
transport diagnostic are simply too large for this configuration.
"""
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
from real_terrain_validation import (  # noqa: E402
    RealTerrainValidationConfig,
    run_real_terrain_validation,
)

BASE_KERNEL = dict(
    enable_prognostic_column_water=True,
    enable_prognostic_condensate=True,
    column_water_use_bulk_condensate_rainfall=True,
    enable_stability_aware_condensation=True,
    enable_two_layer_convective_adjustment=True,
    enable_three_level_pressure_column=True,
)

VARIANTS: dict[str, dict] = {
    # The base free-tropospheric momentum kernel alone: no mass-flux closure,
    # no overturning heating, no energy-limited evaporation. Isolates whether
    # evolve_wind_aloft's own forcing/damping balance already produces
    # excessive mid/upper wind before any of Sections 8-15's additions.
    "kernel_only": dict(BASE_KERNEL),
    # The full Section 15 heated configuration, for direct comparison.
    "full_heated": dict(
        BASE_KERNEL,
        enable_three_level_horizontal_mass_flux_closure=True,
        enable_energy_limited_evaporation=True,
        evaporation_downwelling_longwave_w_m2=25.0,
        enable_native_balanced_moist_static_energy_overturning=True,
        native_balanced_mse_radiative_relaxation_days=10.0,
        native_balanced_mse_use_toa_radiative_target=True,
    ),
}

HADLEY_EDGE_DEG = 24.0


def _tropical_stats(field: np.ndarray, latitude: np.ndarray) -> dict:
    tropical = np.abs(latitude) <= HADLEY_EDGE_DEG
    cos = np.cos(np.radians(latitude))[:, None]
    zonal_mean = np.mean(field, axis=1)
    return {
        "zonal_mean_tropical_mean_abs_m_s": float(np.mean(np.abs(zonal_mean[tropical]))),
        "zonal_mean_tropical_max_abs_m_s": float(np.max(np.abs(zonal_mean[tropical]))),
        "area_weighted_tropical_rms_m_s": float(
            np.sqrt(
                np.sum((field[tropical] ** 2) * cos[tropical])
                / np.sum(np.broadcast_to(cos, field.shape)[tropical])
            )
        ),
    }


def _run_one(overrides: dict) -> dict:
    config = RealTerrainValidationConfig(
        height=32, width=64, spinup_years=1.0, evaluation_years=1.0,
    )
    pp = dataclasses.replace(EARTH, **overrides)
    state, report = run_real_terrain_validation(config, planet_params=pp)

    h = state.wind_v.shape[0]
    latitude = 90.0 - (np.arange(h, dtype=np.float64) + 0.5) * 180.0 / h

    results = {
        "lower_v": _tropical_stats(np.asarray(state.wind_v, dtype=np.float64), latitude),
        "reference_earth_hadley_zonal_mean_v_m_s": "~1-3 (literature, zonal-mean, all levels)",
    }
    if state.midlevel_wind_v is not None:
        results["midlevel_v"] = _tropical_stats(
            np.asarray(state.midlevel_wind_v, dtype=np.float64), latitude
        )
    if state.wind_v_aloft is not None:
        results["upper_v"] = _tropical_stats(
            np.asarray(state.wind_v_aloft, dtype=np.float64), latitude
        )
    results["circulation"] = report["metrics"].get("circulation")
    return results


def main() -> int:
    all_results = {}
    for name, overrides in VARIANTS.items():
        all_results[name] = _run_one(overrides)
        print(f"--- {name} ---")
        print(json.dumps(all_results[name], indent=2, default=str))

    output_path = ROOT / "scripts" / "resolved_wind_magnitude_diagnostic.json"
    output_path.write_text(
        json.dumps(all_results, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
