"""Run a bounded CRU-scored calibration matrix for land energy parameters."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-climatology", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thermal-inertia-days", type=parse_values, default=(15.0, 30.0, 60.0))
    parser.add_argument("--transport-gains", type=parse_values, default=(0.35, 0.4, 0.45))
    args = parser.parse_args()
    baseline = load_validation_report(args.baseline_report)
    rows = []
    for inertia in args.thermal_inertia_days:
        for transport_gain in args.transport_gains:
            parameters = {
                "land_thermal_inertia_days": inertia,
                "land_transport_gain": transport_gain,
            }
            _, report = run_real_terrain_validation(
                RealTerrainValidationConfig(),
                planet_params=dataclasses.replace(EARTH, **parameters),
                monthly_climatology_path=args.monthly_climatology,
            )
            rows.append({"parameters": parameters, "decision": evaluate_land_candidate(report, baseline)})
    payload = {
        "schema_version": 1,
        "accepted_count": sum(row["decision"]["accepted"] for row in rows),
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
