"""Gate a long climate candidate before it can change a default parameter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climate_acceptance import evaluate_precipitation_candidate  # noqa: E402
from real_terrain_validation import load_validation_report  # noqa: E402


def promotion_decision(candidate: dict, baseline: dict) -> dict:
    """Require matching long-run configuration plus the short-run skill gate."""
    config = candidate.get("config", {})
    baseline_config = baseline.get("config", {})
    configuration_gates = {
        "configuration_matches_baseline": config == baseline_config,
        "resolution_at_least_128x256": int(config.get("height", 0)) >= 128
        and int(config.get("width", 0)) >= 256,
        "spinup_at_least_5_years": float(config.get("spinup_years", 0.0)) >= 5.0,
        "evaluation_at_least_5_years": float(config.get("evaluation_years", 0.0)) >= 5.0,
    }
    skill = evaluate_precipitation_candidate(candidate, baseline)
    return {
        "configuration_gates": configuration_gates,
        "skill": skill,
        "promoted": all(configuration_gates.values()) and skill["accepted"],
        "required_follow_up": (
            "Run `python -m pytest testing -m slow` before changing a default."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    args = parser.parse_args()
    decision = promotion_decision(
        load_validation_report(args.candidate_report),
        load_validation_report(args.baseline_report),
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["promoted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
