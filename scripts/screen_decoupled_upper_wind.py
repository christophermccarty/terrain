"""Section 17 compact screen: does decoupling the upper wind move transport?

PRIOR_ART_IMPLEMENTATION_PLAN.md Section 16 found the three-level path's
excess cross-equatorial transport traces back to state.wind_u_aloft/
wind_v_aloft -- the shared, always-on jet-stream kernel -- and framed a
design fork: decouple the three-level path's upper wind from that kernel, or
accept it as a known limitation. Section 17 implements the decoupling
(PlanetState.upperlevel_wind_u/v, evolved by its own
three_level_upper_wind_pgf_fraction/three_level_upper_wind_damping).

This script re-runs the same 32x64, one-year spin-up/one-year evaluation
protocol used throughout Sections 11-15 (the SHARED configuration below is
copied from scripts/sweep_mass_flux_cap_with_overturning.py, the most recent
compact-screen script in that family) with the new decoupled upper wind, at
a few three_level_upper_wind_damping values, and reports whether Köppen
group/class skill and cross-equatorial transport move toward Earth's real
~5-6 PW peak Hadley-cell magnitude.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import EARTH  # noqa: E402
from real_terrain_validation import (  # noqa: E402
    RealTerrainValidationConfig,
    run_real_terrain_validation,
)

# Identical to scripts/sweep_mass_flux_cap_with_overturning.py's SHARED dict
# (Section 14's screen), the most recent full "retained candidate"
# configuration documented in the plan -- column-water/condensate/two-layer/
# three-level pipeline, the horizontal mass-flux closure, energy-limited
# evaporation at the Section 3 longwave constant, and the MSE overturning
# closure at its 10-day relaxation.
SHARED = dict(
    enable_prognostic_column_water=True,
    enable_prognostic_condensate=True,
    column_water_use_bulk_condensate_rainfall=True,
    enable_stability_aware_condensation=True,
    enable_two_layer_convective_adjustment=True,
    enable_three_level_pressure_column=True,
    enable_three_level_horizontal_mass_flux_closure=True,
    enable_energy_limited_evaporation=True,
    evaporation_downwelling_longwave_w_m2=25.0,
    enable_native_balanced_moist_static_energy_overturning=True,
    native_balanced_mse_radiative_relaxation_days=10.0,
)

# Section 11-15's default cap/target: TOA-only radiative target at 10-day
# relaxation was the row directly re-measured throughout Sections 12-15
# (native_balanced_mse_use_toa_radiative_target defaults False, i.e. the
# ocean-transport target; both are screened here for completeness against
# the new decoupled upper wind).
TARGETS = {"ocean": False, "toa": True}
DAMPING_VALUES = (0.05, 0.08, 0.15, 0.25, 0.40)

# Follow-up sweep: transport stayed noisy in the same -15 to -28 PW band at
# every damping value above (non-monotonic, no clean lever). PGF fraction
# scales the forcing itself, before each substep's Coriolis rotation acts on
# it, so it is the more direct lever on the equilibrium wind magnitude per
# evolve_wind_aloft's own mechanism. Held at a fixed, reasonable damping
# (0.08, matching the independent middle wind's own precedent) while
# sweeping the fraction; 0.55 mirrors the middle wind's own fraction.
PGF_FRACTION_VALUES = (0.1, 0.25, 0.4, 0.55, 0.7, 1.0)
PGF_FRACTION_FIXED_DAMPING = 0.08


def _extract(metrics: dict) -> dict:
    circulation = metrics.get("circulation") or {}
    transport = circulation.get("meridional_transport") or {}
    koppen = metrics.get("koppen_map_skill") or {}
    return {
        "global_precip_mm_day": metrics["global"]["precip_mm_day"],
        "koppen_group_accuracy": koppen.get("group_accuracy"),
        "koppen_class_accuracy": koppen.get("class_accuracy"),
        "reference_error_score": metrics["reference_error_score"],
        "cross_equatorial_total_energy_transport_pw": transport.get(
            "cross_equatorial_total_energy_transport_pw"
        ),
        "upper_jet_latitude_deg": circulation.get("upper_jet_latitude_deg"),
        "upper_hadley_edge_deg": circulation.get("upper_hadley_edge_deg"),
    }


def main() -> int:
    config = RealTerrainValidationConfig(
        height=32, width=64, spinup_years=1.0, evaluation_years=1.0,
    )
    results: dict[str, dict] = {}
    for target_name, use_toa in TARGETS.items():
        for damping in DAMPING_VALUES:
            key = f"{target_name}_damping{damping}"
            pp = dataclasses.replace(
                EARTH, **SHARED,
                native_balanced_mse_use_toa_radiative_target=use_toa,
                three_level_upper_wind_damping=damping,
            )
            _, report = run_real_terrain_validation(config, planet_params=pp)
            results[key] = _extract(report["metrics"])
            print(f"--- {key} ---")
            print(json.dumps(results[key], indent=2, default=str))

    for target_name, use_toa in TARGETS.items():
        for pgf_fraction in PGF_FRACTION_VALUES:
            key = f"{target_name}_pgf_fraction{pgf_fraction}"
            pp = dataclasses.replace(
                EARTH, **SHARED,
                native_balanced_mse_use_toa_radiative_target=use_toa,
                three_level_upper_wind_pgf_fraction=pgf_fraction,
                three_level_upper_wind_damping=PGF_FRACTION_FIXED_DAMPING,
            )
            _, report = run_real_terrain_validation(config, planet_params=pp)
            results[key] = _extract(report["metrics"])
            print(f"--- {key} ---")
            print(json.dumps(results[key], indent=2, default=str))

    output_path = ROOT / "scripts" / "decoupled_upper_wind_screen_result.json"
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
