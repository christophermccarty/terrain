"""Isolate the horizontal mass-flux closure's own cross-equatorial transport.

Section 11 of docs/PRIOR_ART_IMPLEMENTATION_PLAN.md falsified the
"borrowed hemispheric asymmetry" hypothesis for the MSE overturning closure's
excess cross-equatorial transport (-18 to -31 PW across four ocean/TOA
radiative-target variants). The remaining suspect is the interaction between
`thermally_direct_overturning`'s fixed 0.40/0.35/0.25 mass-conserving branch
structure and `enable_three_level_horizontal_mass_flux_closure`'s own
divergence correction.

This script isolates that by running the identical shared configuration with
the overturning closure's heating pinned to exactly zero (both overturning
gates left off, so `thermally_direct_overturning` is never invoked at all),
holding everything else -- including the mass-flux closure -- fixed. It also
reproduces one of Section 11's four rows (ocean-target, 10-day relaxation) in
this script's own harness as a sanity check that the two are apples-to-apples
before trusting the new zero-heating number.
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
)

VARIANTS: dict[str, dict] = {
    # Sanity check: should land close to Section 11's "ocean-transport target,
    # 10-day relaxation" row (2.61 mm/day, 0.545/0.227, 1.112, -25.10 PW, 0.181).
    "reproduce_section11_ocean_10day": dict(
        SHARED,
        enable_native_balanced_moist_static_energy_overturning=True,
        native_balanced_mse_use_toa_radiative_target=False,
        native_balanced_mse_radiative_relaxation_days=10.0,
    ),
    # The new diagnostic: overturning heating pinned to zero (both overturning
    # gates off), mass-flux closure still on. Isolates the closure's own
    # contribution to cross-equatorial transport.
    "mass_flux_closure_only_zero_heating": dict(SHARED),
    # Control: neither the overturning closure nor the mass-flux closure.
    "neither_closure": {
        k: v for k, v in SHARED.items()
        if k not in ("enable_three_level_horizontal_mass_flux_closure",)
    },
}


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
        "cross_equatorial_dry_static_energy_transport_pw": transport.get(
            "cross_equatorial_dry_static_energy_transport_pw"
        ),
        "cross_equatorial_latent_energy_transport_pw": transport.get(
            "cross_equatorial_latent_energy_transport_pw"
        ),
    }


def main() -> int:
    config = RealTerrainValidationConfig(
        height=32, width=64, spinup_years=1.0, evaluation_years=1.0,
    )
    results: dict[str, dict] = {}
    for name, overrides in VARIANTS.items():
        pp = dataclasses.replace(EARTH, **overrides)
        _, report = run_real_terrain_validation(config, planet_params=pp)
        results[name] = _extract(report["metrics"])
        print(f"--- {name} ---")
        print(json.dumps(results[name], indent=2))

    output_path = ROOT / "scripts" / "mass_flux_transport_isolation_result.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
