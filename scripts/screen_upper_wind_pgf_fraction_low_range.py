"""Section 19 follow-up: extend the pgf_fraction axis below 0.05.

Section 19 (docs/PRIOR_ART_IMPLEMENTATION_PLAN.md) proved cross-equatorial
transport carries no usable signal at the compact 32x64/1yr protocol
(phase-shift alone swings it more than the whole physics grid), but Koppen
group accuracy is robust to both the physics-parameter axis and the
phase axis, and tracked `three_level_upper_wind_pgf_fraction` cleanly and
monotonically in Section 18's fine sweep, best at 0.05 -- the lowest value
tested there, with no sign of plateauing.

This script extends the fraction axis down to and including 0.0 (zero PGF
forcing on the decoupled upper wind -- the floor case) to find where group
accuracy actually plateaus or turns over, holding
`three_level_upper_wind_damping=0.08` fixed throughout (Section 18 showed
damping barely perturbs accuracy: row stdev <0.01) at both radiative
targets, same 32x64 one-year spin-up/one-year evaluation protocol.
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
    enable_native_balanced_moist_static_energy_overturning=True,
    native_balanced_mse_radiative_relaxation_days=10.0,
    three_level_upper_wind_damping=0.08,
)

TARGETS = {"ocean": False, "toa": True}
# 0.05/0.1 repeat Section 18's fine-sweep points as an in-run consistency
# check; 0.0/0.01/0.025/0.075 are the new points extending the axis.
PGF_FRACTION_VALUES = (0.0, 0.01, 0.025, 0.05, 0.075, 0.1)


def _extract(metrics: dict) -> dict:
    circulation = metrics.get("circulation") or {}
    transport = circulation.get("meridional_transport") or {}
    koppen = metrics.get("koppen_map_skill") or {}
    return {
        "koppen_group_accuracy": koppen.get("group_accuracy"),
        "koppen_class_accuracy": koppen.get("class_accuracy"),
        "reference_error_score": metrics["reference_error_score"],
        "global_precip_mm_day": metrics["global"]["precip_mm_day"],
        "cross_equatorial_total_energy_transport_pw": transport.get(
            "cross_equatorial_total_energy_transport_pw"
        ),
    }


def main() -> int:
    results: dict[str, dict] = {}
    for target_name, use_toa in TARGETS.items():
        for pgf_fraction in PGF_FRACTION_VALUES:
            key = f"{target_name}_frac{pgf_fraction}"
            pp = dataclasses.replace(
                EARTH, **SHARED,
                native_balanced_mse_use_toa_radiative_target=use_toa,
                three_level_upper_wind_pgf_fraction=pgf_fraction,
            )
            config = RealTerrainValidationConfig(
                height=32, width=64, spinup_years=1.0, evaluation_years=1.0,
            )
            _, report = run_real_terrain_validation(config, planet_params=pp)
            results[key] = _extract(report["metrics"])
            print(f"--- {key} ---")
            print(json.dumps(results[key], indent=2, default=str))

    output_path = ROOT / "scripts" / "upper_wind_pgf_fraction_low_range_result.json"
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
