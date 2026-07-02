"""benchmark_headless.py — Formal headless performance benchmark (PLAN.md Phase 5).

Measures wall-clock cost per simulated year for each TimeScaleMode, at both the
historically-documented "reference" resolution (60x120) and the actual
`main.py` production default (512x1024) — the two badly diverge (see
scripts/profile_simulate_step.py's findings), so both are tracked separately
rather than assuming one represents the other.

Appends a timestamped record to scripts/benchmark_results.json on every run,
so results can be tracked over time (e.g. to catch a future regression).

Usage
-----
    python scripts/benchmark_headless.py                     # both sizes, all modes
    python scripts/benchmark_headless.py --size 512 1024     # production only
    python scripts/benchmark_headless.py --modes DAILY WEEKLY
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_RESULTS_PATH = ROOT / "scripts" / "benchmark_results.json"

# Historically documented benchmark target: <90s to simulate 1 year, headless.
_TARGET_S_PER_YEAR = 90.0


def _seconds_per_year(mode_name: str, H: int, W: int) -> float:
    from optimizer.headless import _advance_one_cycle, _make_default_elevation, _DAYS_PER_CYCLE
    from planet_params import EARTH
    from simulate import create_initial_state, TimeScaleMode

    mode = TimeScaleMode[mode_name]
    elevation = _make_default_elevation(H, W)
    state = create_initial_state(elevation, day_of_year=80.0)

    # Warm up Numba JIT with one cycle, un-timed.
    state = _advance_one_cycle(state, mode, planet_params=EARTH)

    cycle_days = _DAYS_PER_CYCLE[mode]
    n_cycles = max(1, round(EARTH.orbital_period_days / cycle_days))

    t0 = time.perf_counter()
    for _ in range(n_cycles):
        state = _advance_one_cycle(state, mode, planet_params=EARTH)
    elapsed = time.perf_counter() - t0
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal headless performance benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[60, 120, 512, 1024],
                         help="Grid size pairs, e.g. --sizes 60 120 512 1024 (must be even count)")
    parser.add_argument("--modes", type=str, nargs="+",
                         default=["DAILY", "WEEKLY", "MONTHLY", "ANNUAL"],
                         help="TimeScaleMode names to benchmark")
    parser.add_argument("--no-save", action="store_true", help="Don't append to benchmark_results.json")
    args = parser.parse_args()

    if len(args.sizes) % 2 != 0:
        parser.error("--sizes must be an even number of values (H W pairs)")
    size_pairs = [(args.sizes[i], args.sizes[i + 1]) for i in range(0, len(args.sizes), 2)]

    print(f"Headless benchmark — target: <{_TARGET_S_PER_YEAR:.0f}s/simulated-year\n")
    record: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_s_per_year": _TARGET_S_PER_YEAR,
        "results": [],
    }

    header = f"{'Size':>12}  {'Mode':<8}  {'s/year':>10}  {'vs target':>10}"
    print(header)
    print("-" * len(header))
    for H, W in size_pairs:
        for mode_name in args.modes:
            elapsed = _seconds_per_year(mode_name, H, W)
            status = "OK" if elapsed < _TARGET_S_PER_YEAR else "OVER"
            print(f"{H:>5}x{W:<6}  {mode_name:<8}  {elapsed:>9.2f}s  {status:>10}")
            record["results"].append({
                "H": H, "W": W, "mode": mode_name, "seconds_per_year": elapsed, "status": status,
            })

    if not args.no_save:
        history = []
        if _RESULTS_PATH.exists():
            try:
                history = json.loads(_RESULTS_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                history = []
        history.append(record)
        _RESULTS_PATH.write_text(json.dumps(history, indent=2))
        print(f"\nAppended to {_RESULTS_PATH} ({len(history)} recorded run(s) total)")


if __name__ == "__main__":
    main()
