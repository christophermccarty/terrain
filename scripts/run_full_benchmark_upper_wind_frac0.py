"""Section 20 candidate: the mandatory full-resolution CRU benchmark.

Section 19 proved cross-equatorial transport carries no signal at the
compact 32x64/1yr protocol, and Section 20 selected
`three_level_upper_wind_pgf_fraction=0.0`/`three_level_upper_wind_damping=
0.08`/TOA radiative target as the single candidate worth carrying forward,
by Koppen group accuracy (the one metric shown robust to both the
physics-parameter axis and the evaluation-window-phase axis) -- but
explicitly not a promotion on compact-screen evidence alone.

This runs the standard 128x256 five-year spin-up/five-year evaluation CRU
benchmark this whole experimental family (Sections 8-20) has deferred to:
the untouched baseline (no three-level/closure/overturning gates at all)
and the Section 20 candidate, back to back, at full resolution/duration.
"""
from __future__ import annotations

import dataclasses
import json
import time
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

CANDIDATE = dict(
    **SHARED,
    native_balanced_mse_use_toa_radiative_target=True,
    three_level_upper_wind_pgf_fraction=0.0,
    three_level_upper_wind_damping=0.08,
)

CONFIGS = {
    "baseline_untouched": EARTH,
    "candidate_frac0_damp0.08_toa": dataclasses.replace(EARTH, **CANDIDATE),
}


def _extract(metrics: dict) -> dict:
    circulation = metrics.get("circulation") or {}
    transport = circulation.get("meridional_transport") or {}
    koppen = metrics.get("koppen_map_skill") or {}
    return {
        "koppen_group_accuracy": koppen.get("group_accuracy"),
        "koppen_class_accuracy": koppen.get("class_accuracy"),
        "reference_error_score": metrics["reference_error_score"],
        "global_precip_mm_day": metrics["global"]["precip_mm_day"],
        "global_temp_c": metrics["global"].get("temp_c"),
        "cross_equatorial_total_energy_transport_pw": transport.get(
            "cross_equatorial_total_energy_transport_pw"
        ),
        "peak_northward_energy_transport_pw": transport.get(
            "peak_northward_energy_transport_pw"
        ),
        "peak_southward_energy_transport_pw": transport.get(
            "peak_southward_energy_transport_pw"
        ),
    }


def main() -> int:
    config = RealTerrainValidationConfig(
        height=128, width=256, spinup_years=5.0, evaluation_years=5.0,
    )
    results: dict[str, dict] = {}
    for name, planet_params in CONFIGS.items():
        print(f"=== running {name} (128x256, 5yr spin-up / 5yr evaluation) ===")
        t0 = time.time()
        _, report = run_real_terrain_validation(config, planet_params=planet_params)
        elapsed = time.time() - t0
        results[name] = _extract(report["metrics"])
        results[name]["elapsed_seconds"] = elapsed
        print(f"--- {name} ({elapsed:.1f}s) ---")
        print(json.dumps(results[name], indent=2, default=str))

    output_path = ROOT / "scripts" / "full_benchmark_upper_wind_frac0_result.json"
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
