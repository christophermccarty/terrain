"""Run a 32x64 attribution matrix for the experimental boundary layer."""
from __future__ import annotations

import argparse
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


def _summary(report: dict) -> dict:
    metrics = report["metrics"]
    global_metrics = metrics["global"]
    koppen = metrics["koppen_map_skill"]
    thresholds = metrics["koppen_temperature_thresholds"]
    return {
        "global_temperature_k": global_metrics["temperature_k"],
        "ocean_temperature_k": global_metrics["ocean_temperature_k"],
        "global_precipitation_mm_day": global_metrics["precip_mm_day"],
        "seasonal_cycle_error": metrics["land_seasonal_cycle"]["cycle_error_score"],
        "koppen_group_accuracy": koppen["group_accuracy"],
        "koppen_class_accuracy": koppen["class_accuracy"],
        "koppen_group_share_mae_pp": koppen["group_share_mae_pp"],
        "coldest_month_accuracy": thresholds["coldest_month"]["accuracy"],
        "warmest_month_accuracy": thresholds["warmest_month"]["accuracy"],
        "regional_precipitation_mm_year": {
            name: metrics["regional_precip_mm_year"].get(name)
            for name in ("Central Europe", "East China", "S Japan", "SE US")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "temp" / "boundary_layer_ablation_32x64.json",
    )
    args = parser.parse_args()
    config = RealTerrainValidationConfig(height=32, width=64)
    common = {
        "enable_force_restore_land": True,
        "enable_force_restore_atmospheric_heat_convergence": True,
    }
    variants = {
        "resolved_control": {},
        "reservoir_only": {
            "enable_force_restore_boundary_layer": True,
            "enable_boundary_layer_capacity_aware_airsea_exchange": True,
            "enable_boundary_layer_capacity_aware_free_air_transport": True,
            "enable_boundary_layer_near_surface_cloud_temperature": True,
            "enable_boundary_layer_split_invariant_cloud_memory": True,
        },
        "stability_only": {
            "enable_force_restore_boundary_layer": True,
            "enable_boundary_layer_capacity_aware_airsea_exchange": True,
            "enable_boundary_layer_capacity_aware_free_air_transport": True,
            "enable_boundary_layer_near_surface_cloud_temperature": True,
            "enable_boundary_layer_split_invariant_cloud_memory": True,
            "enable_boundary_layer_stability_dependent_exchange": True,
        },
        "transport_only": {
            "enable_force_restore_boundary_layer": True,
            "enable_boundary_layer_capacity_aware_airsea_exchange": True,
            "enable_boundary_layer_capacity_aware_free_air_transport": True,
            "enable_boundary_layer_near_surface_cloud_temperature": True,
            "enable_boundary_layer_split_invariant_cloud_memory": True,
            "enable_boundary_layer_horizontal_transport": True,
        },
        "stability_and_transport": {
            "enable_force_restore_boundary_layer": True,
            "enable_boundary_layer_capacity_aware_airsea_exchange": True,
            "enable_boundary_layer_capacity_aware_free_air_transport": True,
            "enable_boundary_layer_near_surface_cloud_temperature": True,
            "enable_boundary_layer_split_invariant_cloud_memory": True,
            "enable_boundary_layer_stability_dependent_exchange": True,
            "enable_boundary_layer_horizontal_transport": True,
        },
    }
    reports = {}
    summaries = {}
    for name, overrides in variants.items():
        _, report = run_real_terrain_validation(
            config,
            planet_params=dataclasses.replace(EARTH, **common, **overrides),
            track_phase3_heat_convergence=True,
        )
        reports[name] = report
        summaries[name] = _summary(report)

    control = summaries["resolved_control"]
    deltas = {
        name: {
            key: value - control[key]
            for key, value in summary.items()
            if isinstance(value, (int, float))
        }
        for name, summary in summaries.items()
        if name != "resolved_control"
    }
    payload = {
        "schema_version": 1,
        "config": dataclasses.asdict(config),
        "summaries": summaries,
        "deltas_from_resolved_control": deltas,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "summaries": summaries,
        "deltas_from_resolved_control": deltas,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
