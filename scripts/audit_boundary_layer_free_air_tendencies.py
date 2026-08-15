"""Attribute atmospheric storage to temperature operators at 32x64."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import EARTH  # noqa: E402
from real_terrain_validation import RealTerrainValidationConfig  # noqa: E402
from scripts.audit_boundary_layer_column_energy import _run  # noqa: E402


def main() -> int:
    config = RealTerrainValidationConfig(height=32, width=64)
    common = dict(
        enable_force_restore_land=True,
        enable_force_restore_atmospheric_heat_convergence=True,
    )
    variants = {
        "resolved_control": dataclasses.replace(EARTH, **common),
        "reservoir_only": dataclasses.replace(
            EARTH,
            **common,
            enable_force_restore_boundary_layer=True,
            enable_boundary_layer_capacity_aware_airsea_exchange=True,
            enable_boundary_layer_capacity_aware_free_air_transport=True,
            enable_boundary_layer_near_surface_cloud_temperature=True,
            enable_boundary_layer_split_invariant_cloud_memory=True,
        ),
        "stability_only": dataclasses.replace(
            EARTH,
            **common,
            enable_force_restore_boundary_layer=True,
            enable_boundary_layer_capacity_aware_airsea_exchange=True,
            enable_boundary_layer_capacity_aware_free_air_transport=True,
            enable_boundary_layer_near_surface_cloud_temperature=True,
            enable_boundary_layer_split_invariant_cloud_memory=True,
            enable_boundary_layer_stability_dependent_exchange=True,
        ),
    }
    report = {
        "schema_version": 1,
        "grid": [config.height, config.width],
        "variants": {name: _run(params, config) for name, params in variants.items()},
    }
    output = ROOT / "temp" / "boundary_layer_free_air_tendencies_32x64.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
