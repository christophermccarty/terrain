"""Run the deterministic real-terrain climate validation benchmark."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import EARTH  # noqa: E402
from real_terrain_validation import (  # noqa: E402
    DEFAULT_BASELINE_PATH,
    RealTerrainValidationConfig,
    compare_validation_reports,
    load_validation_report,
    run_real_terrain_validation,
    save_validation_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible regional/zonal validation on the bundled Earth DEM."
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--spinup-years", type=float, default=1.0)
    parser.add_argument("--evaluation-years", type=float, default=1.0)
    parser.add_argument(
        "--time-scale",
        choices=("DAILY", "WEEKLY", "MONTHLY", "ANNUAL"),
        default="MONTHLY",
    )
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--wind-block-size", type=int, default=4)
    parser.add_argument("--precip-block-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="PlanetParams numeric override (repeatable).",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="?",
        const=DEFAULT_BASELINE_PATH,
        help="Compare against a baseline (default: tracked compact baseline).",
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        nargs="?",
        const=DEFAULT_BASELINE_PATH,
        help="Write this result as a new versioned baseline.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = RealTerrainValidationConfig(
        height=args.height,
        width=args.width,
        spinup_years=args.spinup_years,
        evaluation_years=args.evaluation_years,
        time_scale=args.time_scale,
        block_size=args.block_size,
        wind_block_size=args.wind_block_size,
        precip_block_size=args.precip_block_size,
    )
    overrides: dict[str, object] = {}
    field_names = {field.name for field in dataclasses.fields(EARTH)}
    for item in args.param:
        name, separator, raw_value = item.partition("=")
        if not separator or not name:
            raise SystemExit(f"Invalid --param {item!r}; expected NAME=VALUE")
        if name not in field_names:
            raise SystemExit(f"Unknown PlanetParams field {name!r}")
        default = getattr(EARTH, name)
        if isinstance(default, bool):
            normalized = raw_value.strip().lower()
            if normalized not in {"0", "1", "false", "true", "no", "yes"}:
                raise SystemExit(
                    f"Boolean --param {name} expects true/false or 1/0, got {raw_value!r}"
                )
            overrides[name] = normalized in {"1", "true", "yes"}
        elif isinstance(default, int):
            overrides[name] = int(raw_value)
        elif isinstance(default, float):
            overrides[name] = float(raw_value)
        else:
            raise SystemExit(f"--param does not support non-scalar field {name!r}")
    planet = dataclasses.replace(EARTH, **overrides) if overrides else EARTH

    _, report = run_real_terrain_validation(
        config,
        planet_params=planet,
        initial_state_path=args.initial_state,
    )
    if args.output:
        save_validation_report(report, args.output)
    if args.write_baseline:
        save_validation_report(report, args.write_baseline)

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.compare:
        baseline = load_validation_report(args.compare)
        failures = compare_validation_reports(report, baseline)
        if failures:
            print("\nREGRESSION FAILURES:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1
        print("\nBaseline comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
