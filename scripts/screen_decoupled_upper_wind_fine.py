"""Section 17 follow-up: finer bounded sweep around the pgf_fraction=0.1 candidate.

Section 17 (docs/PRIOR_ART_IMPLEMENTATION_PLAN.md) found that decoupling the
three-level path's upper wind from the shared jet-stream kernel and setting
`three_level_upper_wind_pgf_fraction=0.1` (damping=0.08) improved
cross-equatorial transport from the -14 to -31 PW band down to -9.36 PW (TOA
target) while also improving Köppen group accuracy -- but that was one point
on a coarse six-value fraction sweep (0.1/0.25/0.4/0.55/0.7/1.0) crossed with
a coarse damping sweep (0.05/0.08/0.15/0.25/0.40), never run jointly, and
explicitly flagged as not a promotion candidate on that evidence alone.

This script runs the finer, *joint* sweep the plan calls for next: fraction
in (0.05, 0.1, 0.15, 0.2) crossed with damping in (0.04, 0.06, 0.08, 0.10,
0.12) -- a finer grid centered on the coarse sweep's 0.08 default -- at both
radiative targets, on the identical 32x64 one-year spin-up/one-year
evaluation protocol used throughout Sections 11-17. This checks whether
fraction=0.1 is a real local optimum or just the edge of the coarse grid,
and whether some damping off the 0.08 precedent does better once crossed
with fraction rather than swept independently.
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

# Identical to scripts/screen_decoupled_upper_wind.py's SHARED dict.
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

TARGETS = {"ocean": False, "toa": True}
PGF_FRACTION_VALUES = (0.05, 0.1, 0.15, 0.2)
DAMPING_VALUES = (0.04, 0.06, 0.08, 0.10, 0.12)


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
    total = len(TARGETS) * len(PGF_FRACTION_VALUES) * len(DAMPING_VALUES)
    done = 0
    for target_name, use_toa in TARGETS.items():
        for pgf_fraction in PGF_FRACTION_VALUES:
            for damping in DAMPING_VALUES:
                key = f"{target_name}_frac{pgf_fraction}_damp{damping}"
                pp = dataclasses.replace(
                    EARTH, **SHARED,
                    native_balanced_mse_use_toa_radiative_target=use_toa,
                    three_level_upper_wind_pgf_fraction=pgf_fraction,
                    three_level_upper_wind_damping=damping,
                )
                _, report = run_real_terrain_validation(config, planet_params=pp)
                results[key] = _extract(report["metrics"])
                done += 1
                print(f"--- [{done}/{total}] {key} ---")
                print(json.dumps(results[key], indent=2, default=str))

    output_path = ROOT / "scripts" / "decoupled_upper_wind_fine_screen_result.json"
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
