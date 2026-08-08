"""Sweep the eddy-correction speed cap with overturning heating restored.

Section 13 of docs/PRIOR_ART_IMPLEMENTATION_PLAN.md found, on the *zero-heating*
configuration, that `three_level_horizontal_mass_flux_max_speed_m_s=12.0`
(the current default) clips the closure's eddy/Fourier-inverted correction to
only 6% divergence closure, and that raising the cap improves Köppen skill
monotonically but does not improve cross-equatorial transport monotonically
(best at 120 m/s, worse again at 1000 m/s). That scan intentionally excluded
any diabatic heating to isolate the closure's own behavior.

This script re-runs the same cap values against the full retained
configuration from Section 11 -- both the ocean-transport and TOA-only
radiative targets at their 10-day relaxation -- to see whether the
zero-heating finding still holds once `thermally_direct_overturning` is
actually contributing meridional flow. Per this project's standing rule, a
short compact-grid result here is a screen, not a promotion: any winner still
needs the full 128x256 five-year spin-up/five-year evaluation gate.
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
)

CAP_VALUES_M_S = (20.0, 40.0, 60.0, 80.0, 120.0)
TARGETS = {"ocean": False, "toa": True}


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
    }


def main() -> int:
    config = RealTerrainValidationConfig(
        height=32, width=64, spinup_years=1.0, evaluation_years=1.0,
    )
    results: dict[str, dict] = {}
    for target_name, use_toa in TARGETS.items():
        for cap in CAP_VALUES_M_S:
            key = f"{target_name}_cap{int(cap)}"
            pp = dataclasses.replace(
                EARTH, **SHARED,
                native_balanced_mse_use_toa_radiative_target=use_toa,
                three_level_horizontal_mass_flux_max_speed_m_s=cap,
            )
            _, report = run_real_terrain_validation(config, planet_params=pp)
            results[key] = _extract(report["metrics"])
            print(f"--- {key} ---")
            print(json.dumps(results[key], indent=2))

    output_path = ROOT / "scripts" / "mass_flux_cap_overturning_sweep_result.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
