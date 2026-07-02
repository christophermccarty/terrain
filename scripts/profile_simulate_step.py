"""profile_simulate_step.py — cProfile pass on simulate_step (PLAN.md Phase 5, task 1).

Profiles a representative run of `simulate_step` and reports the top CPU
consumers by cumulative time (call-graph hotspots) and by internal/"tottime"
(where the CPU actually spends cycles, excluding callees).

Numba JIT-compiles on first call, so a warmup pass runs un-profiled first —
otherwise the profile is dominated by one-time compilation cost rather than
steady-state performance.

Usage
-----
    python scripts/profile_simulate_step.py [--mode DAILY|WEEKLY|MONTHLY|ANNUAL]
                                             [--steps N] [--size H W] [--top N]
                                             [--out FILE.prof]

Example
-------
    python scripts/profile_simulate_step.py --mode DAILY --steps 30 --size 60 120
    python -m snakeviz profile_simulate_step.prof   # optional flame-graph viewer
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _step_days_for_mode(mode: str) -> float:
    return {"DAILY": 1.0, "WEEKLY": 7.0, "MONTHLY": 30.0, "ANNUAL": 364.0}[mode]


def main() -> None:
    parser = argparse.ArgumentParser(description="cProfile a run of simulate_step")
    parser.add_argument("--mode", choices=["DAILY", "WEEKLY", "MONTHLY", "ANNUAL"], default="DAILY")
    parser.add_argument("--steps", type=int, default=30, help="Number of simulate_step calls to profile")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Un-profiled steps to warm up Numba JIT")
    parser.add_argument("--size", type=int, nargs=2, default=[60, 120], metavar=("H", "W"))
    parser.add_argument("--block-size", type=int, default=3, help="Temperature/precip downsample factor (main.py default: 3)")
    parser.add_argument("--wind-block-size", type=int, default=8, help="Wind downsample factor (main.py default: 8)")
    parser.add_argument("--top", type=int, default=25, help="Number of functions to print per sort order")
    parser.add_argument("--out", type=str, default="scripts/profile_simulate_step.prof",
                         help="Where to save the raw .prof file (for snakeviz etc.)")
    args = parser.parse_args()

    from optimizer.headless import _make_default_elevation
    from planet_params import EARTH
    from simulate import create_initial_state, simulate_step

    H, W = args.size
    elevation = _make_default_elevation(H, W)
    state = create_initial_state(elevation, day_of_year=80.0)
    step_days = _step_days_for_mode(args.mode)
    do_wind = args.mode == "DAILY"

    step_kwargs = dict(
        update_wind=do_wind,
        planet_params=EARTH,
        block_size=args.block_size,
        wind_block_size=args.wind_block_size,
    )

    print(f"Warming up Numba JIT ({args.warmup_steps} un-profiled {args.mode} steps at {H}x{W}, "
          f"block_size={args.block_size}, wind_block_size={args.wind_block_size})...")
    for _ in range(args.warmup_steps):
        state, _ = simulate_step(state, days=step_days, **step_kwargs)

    print(f"Profiling {args.steps} {args.mode} steps ({step_days}d each) at {H}x{W}...")
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.steps):
        state, _ = simulate_step(state, days=step_days, **step_kwargs)
    profiler.disable()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(out_path))
    print(f"\nRaw profile saved to {out_path} (view with `python -m snakeviz {out_path}`)\n")

    stats = pstats.Stats(profiler)

    print("=" * 90)
    print(f"TOP {args.top} BY CUMULATIVE TIME (call-graph hotspots, includes callees)")
    print("=" * 90)
    stats.sort_stats("cumulative").print_stats(args.top)

    print("=" * 90)
    print(f"TOP {args.top} BY INTERNAL TIME (tottime — where CPU cycles are actually spent)")
    print("=" * 90)
    stats.sort_stats("tottime").print_stats(args.top)


if __name__ == "__main__":
    main()
