"""Run a bounded CRU-scored calibration matrix for the condensate closure."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climate_acceptance import evaluate_precipitation_candidate  # noqa: E402
from planet_params import EARTH  # noqa: E402
from real_terrain_validation import (  # noqa: E402
    RealTerrainValidationConfig,
    load_validation_report,
    run_real_terrain_validation,
)


def parse_values(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("expected one or more positive comma-separated values")
    return values


def candidate_overrides(
    condensation_timescales: tuple[float, ...],
    fallout_timescales: tuple[float, ...],
    transport_scales: tuple[float, ...],
    *,
    raw_column_water: bool = False,
    stability_aware: bool = False,
    stability_critical_rh: float = 0.70,
    stability_cape_scale_j_kg: float = 50.0,
):
    """Yield a deterministic, bounded Cartesian calibration matrix."""
    for condensation in condensation_timescales:
        for fallout in fallout_timescales:
            for transport in transport_scales:
                overrides = {
                    "enable_prognostic_condensate": True,
                    "condensate_condensation_timescale_days": condensation,
                    "condensate_fallout_timescale_days": fallout,
                    "condensate_transport_scale": transport,
                }
                if raw_column_water:
                    overrides.update(
                        enable_prognostic_column_water=True,
                        column_water_use_bulk_condensate_rainfall=True,
                    )
                if stability_aware:
                    overrides.update(
                        enable_stability_aware_condensation=True,
                        stability_condensation_critical_rh=stability_critical_rh,
                        stability_condensation_cape_scale_j_kg=stability_cape_scale_j_kg,
                    )
                yield overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-climatology", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condensation-days", type=parse_values, default=(3.0, 6.0, 9.0))
    parser.add_argument("--fallout-days", type=parse_values, default=(1.0,))
    parser.add_argument("--transport-scales", type=parse_values, default=(0.0, 1.0))
    parser.add_argument("--raw-column-water", action="store_true")
    parser.add_argument("--stability-aware", action="store_true")
    parser.add_argument("--stability-critical-rhs", type=parse_values, default=(0.70,))
    parser.add_argument("--stability-cape-scales", type=parse_values, default=(50.0,))
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--spinup-years", type=float, default=5.0)
    parser.add_argument("--evaluation-years", type=float, default=5.0)
    args = parser.parse_args()
    if any(value >= 1.0 for value in args.stability_critical_rhs):
        parser.error("--stability-critical-rhs values must be below 1")
    baseline = load_validation_report(args.baseline_report)
    config = RealTerrainValidationConfig(
        height=args.height,
        width=args.width,
        spinup_years=args.spinup_years,
        evaluation_years=args.evaluation_years,
    )
    rows = []
    for critical_rh in args.stability_critical_rhs:
        for cape_scale in args.stability_cape_scales:
            for overrides in candidate_overrides(
                args.condensation_days,
                args.fallout_days,
                args.transport_scales,
                raw_column_water=args.raw_column_water,
                stability_aware=args.stability_aware,
                stability_critical_rh=critical_rh,
                stability_cape_scale_j_kg=cape_scale,
            ):
                planet = dataclasses.replace(EARTH, **overrides)
                _, report = run_real_terrain_validation(
                    config,
                    planet_params=planet,
                    monthly_climatology_path=args.monthly_climatology,
                )
                rows.append({"parameters": overrides, "decision": evaluate_precipitation_candidate(report, baseline)})
    payload = {
        "schema_version": 1,
        "baseline_report": str(args.baseline_report),
        "monthly_climatology": str(args.monthly_climatology),
        "accepted_count": sum(row["decision"]["accepted"] for row in rows),
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
