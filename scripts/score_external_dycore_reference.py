"""Score a normalized external-dycore monthly artifact against local CRU/Köppen.

The output JSON deliberately uses the same ``metrics`` names as native
real-terrain validation, allowing like-for-like comparison without making the
external GCM a dependency of PlanetSim's interactive runtime.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_dycore import score_external_dycore_against_cru  # noqa: E402
from monthly_climatology import load_monthly_climatology  # noqa: E402


def _json_safe(value):
    """Convert NumPy scalars and undefined correlations into strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly", type=Path, required=True, help="normalized external monthly NPZ")
    parser.add_argument(
        "--cru",
        type=Path,
        default=ROOT / "testing" / "reference_data" / "cru_ts_v4.10_1991_2020.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-land-fraction", type=float, default=0.5)
    args = parser.parse_args()
    model = load_monthly_climatology(args.monthly)
    cru = load_monthly_climatology(args.cru)
    report = {
        "kind": "external_dycore_cru_reference",
        "model_artifact": str(args.monthly),
        "cru_artifact": str(args.cru),
        "metrics": score_external_dycore_against_cru(
            model, cru, minimum_land_fraction=args.minimum_land_fraction
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = _json_safe(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True, allow_nan=False))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
