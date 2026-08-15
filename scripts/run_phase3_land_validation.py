"""Run the bounded Phase 3 land-replacement admission matrix."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climate_acceptance import evaluate_land_candidate, evaluate_phase3_candidate  # noqa: E402
from planet_params import EARTH  # noqa: E402
from real_terrain_validation import RealTerrainValidationConfig, run_real_terrain_validation  # noqa: E402


def phase3_candidate_overrides() -> list[tuple[str, dict[str, float | bool]]]:
    """Five bounded, physically interpretable Phase 3 candidates."""
    common: dict[str, float | bool] = {
        "enable_force_restore_land": True,
        "enable_force_restore_atmospheric_heat_convergence": True,
    }
    return [
        ("resolved_default", dict(common)),
        ("fast_restore_15d", {**common, "land_force_restore_days": 15.0}),
        ("slow_restore_60d", {**common, "land_force_restore_days": 60.0}),
        ("shallower_deep_store", {**common, "land_deep_heat_capacity_j_m2_k": 6_000_000.0}),
        ("lower_dry_resistance", {**common, "land_surface_resistance_dry_s_m": 1_000.0}),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-climatology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--spinup-years", type=float, default=1.0)
    parser.add_argument("--evaluation-years", type=float, default=1.0)
    args = parser.parse_args()
    config = RealTerrainValidationConfig(
        height=args.height,
        width=args.width,
        spinup_years=args.spinup_years,
        evaluation_years=args.evaluation_years,
    )
    _, supported = run_real_terrain_validation(
        config, monthly_climatology_path=args.monthly_climatology
    )
    _, force_restore = run_real_terrain_validation(
        config,
        planet_params=dataclasses.replace(EARTH, enable_force_restore_land=True),
        monthly_climatology_path=args.monthly_climatology,
    )
    rows = []
    for name, overrides in phase3_candidate_overrides():
        _, report = run_real_terrain_validation(
            config,
            planet_params=dataclasses.replace(EARTH, **overrides),
            monthly_climatology_path=args.monthly_climatology,
            track_phase3_heat_convergence=True,
        )
        rows.append(
            {
                "name": name,
                "parameters": overrides,
                "against_supported": evaluate_phase3_candidate(report, supported),
                "against_force_restore": evaluate_land_candidate(report, force_restore),
                "metrics": report["metrics"],
            }
        )
    payload = {
        "schema_version": 1,
        "config": dataclasses.asdict(config),
        "monthly_climatology": str(args.monthly_climatology),
        "supported_metrics": supported["metrics"],
        "force_restore_metrics": force_restore["metrics"],
        "candidates": rows,
        "supported_accepted_count": sum(row["against_supported"]["accepted"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "supported_accepted_count": payload["supported_accepted_count"],
        "candidates": [
            {
                "name": row["name"],
                "against_supported": row["against_supported"],
                "against_force_restore": row["against_force_restore"],
                "heat_convergence": row["metrics"].get("atmospheric_heat_convergence"),
            }
            for row in rows
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
