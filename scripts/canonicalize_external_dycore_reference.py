"""Normalize a preserved ExoPlaSim archive without rerunning its GCM."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_dycore import average_exoplasim_archives, canonicalize_exoplasim_output  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--archive-glob", help="evaluation archives to monthly-average before canonicalizing")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature-key", default="tas")
    parser.add_argument("--precipitation-key", default="pr")
    parser.add_argument(
        "--precipitation-units", choices=("mm_day", "kg_m2_s", "m_s"), default="m_s"
    )
    parser.add_argument("--ncpus", type=int, default=None)
    parser.add_argument("--evaluation-years", type=int, default=None)
    args = parser.parse_args()
    if (args.raw is None) == (args.archive_glob is None):
        parser.error("provide exactly one of --raw or --archive-glob")
    provenance = {"canonicalized_without_rerun": True}
    if args.ncpus is not None:
        provenance["ncpus"] = args.ncpus
    raw = args.raw
    if args.archive_glob is not None:
        temperature, precipitation, archive_years = average_exoplasim_archives(
            sorted(glob.glob(args.archive_glob)),
            temperature_key=args.temperature_key,
            precipitation_key=args.precipitation_key,
        )
        raw = args.output.with_suffix(".external_mean.npz")
        np.savez_compressed(raw, **{args.temperature_key: temperature, args.precipitation_key: precipitation})
        provenance["evaluation_years_actual"] = archive_years
    if args.evaluation_years is not None:
        provenance["evaluation_years_actual"] = args.evaluation_years
    canonicalize_exoplasim_output(
        raw,
        args.request,
        args.output,
        temperature_key=args.temperature_key,
        precipitation_key=args.precipitation_key,
        precipitation_units=args.precipitation_units,
        runner_provenance=provenance,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
