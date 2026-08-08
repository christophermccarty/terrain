"""Run a bounded CRU-scored calibration matrix for the force-restore land path."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climate_acceptance import evaluate_land_candidate  # noqa: E402
from planet_params import EARTH  # noqa: E402
from real_terrain_validation import (  # noqa: E402
    RealTerrainValidationConfig,
    load_validation_report,
    run_real_terrain_validation,
)
from scripts.sweep_condensate_closure import parse_values  # noqa: E402


def candidate_overrides(
    restore_days: tuple[float, ...],
    deep_heat_capacities: tuple[float, ...],
    dry_resistances: tuple[float, ...],
):
    """Yield a bounded Cartesian matrix of physically interpretable controls."""
    for restore in restore_days:
        for capacity in deep_heat_capacities:
            for resistance in dry_resistances:
                yield {
                    "enable_force_restore_land": True,
                    "land_force_restore_days": restore,
                    "land_deep_heat_capacity_j_m2_k": capacity,
                    "land_surface_resistance_dry_s_m": resistance,
                }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-climatology", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--spinup-years", type=float, default=1.0)
    parser.add_argument("--evaluation-years", type=float, default=1.0)
    parser.add_argument("--restore-days", type=parse_values, default=(15.0, 30.0, 60.0))
    parser.add_argument("--deep-heat-capacities", type=parse_values,
                        default=(6_000_000.0, 12_000_000.0))
    parser.add_argument("--dry-resistances", type=parse_values, default=(1_000.0, 2_000.0))
    args = parser.parse_args()
    baseline = load_validation_report(args.baseline_report)
    config = RealTerrainValidationConfig(
        height=args.height, width=args.width,
        spinup_years=args.spinup_years, evaluation_years=args.evaluation_years,
    )
    rows = []
    for overrides in candidate_overrides(
        args.restore_days, args.deep_heat_capacities, args.dry_resistances
    ):
        _, report = run_real_terrain_validation(
            config,
            planet_params=dataclasses.replace(EARTH, **overrides),
            monthly_climatology_path=args.monthly_climatology,
        )
        rows.append({"parameters": overrides, "decision": evaluate_land_candidate(report, baseline)})
    payload = {
        "schema_version": 1,
        "baseline_report": str(args.baseline_report),
        "monthly_climatology": str(args.monthly_climatology),
        "config": dataclasses.asdict(config),
        "accepted_count": sum(row["decision"]["accepted"] for row in rows),
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
