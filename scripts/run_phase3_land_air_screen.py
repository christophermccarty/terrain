"""Screen the distinct Phase 3 boundary layer at directional resolution."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climate_acceptance import evaluate_phase3_candidate  # noqa: E402
from planet_params import EARTH  # noqa: E402
from real_terrain_validation import RealTerrainValidationConfig, run_real_terrain_validation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monthly-climatology", type=Path,
        help="optional CRU-derived monthly climatology NPZ",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--spinup-years", type=float, default=1.0)
    parser.add_argument("--evaluation-years", type=float, default=1.0)
    args = parser.parse_args()
    config = RealTerrainValidationConfig(
        height=args.height,
        width=args.width,
        spinup_years=args.spinup_years,
        evaluation_years=args.evaluation_years,
    )
    common = {
        "enable_force_restore_land": True,
        "enable_force_restore_atmospheric_heat_convergence": True,
    }
    _, control = run_real_terrain_validation(
        config,
        planet_params=dataclasses.replace(EARTH, **common),
        monthly_climatology_path=args.monthly_climatology,
        track_phase3_heat_convergence=True,
    )
    _, candidate = run_real_terrain_validation(
        config,
        planet_params=dataclasses.replace(
            EARTH,
            **common,
            enable_force_restore_boundary_layer=True,
            enable_boundary_layer_stability_dependent_exchange=True,
            enable_boundary_layer_horizontal_transport=True,
        ),
        monthly_climatology_path=args.monthly_climatology,
        track_phase3_heat_convergence=True,
    )
    try:
        decision = evaluate_phase3_candidate(candidate, control)
    except ValueError as exc:
        # A run without the optional CRU monthly archive still produces the
        # built-in regional, threshold, precipitation and Koppen diagnostics,
        # but cannot pass the formal gridded acceptance scorecard.
        decision = {"accepted": False, "incomplete": True, "reason": str(exc)}
    payload = {
        "schema_version": 1,
        "config": dataclasses.asdict(config),
        "control": control,
        "candidate": candidate,
        "decision_against_resolved_control": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "decision": payload["decision_against_resolved_control"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
