"""run_amoc_sweep.py — Sweep amoc_bonus_far to find values that close the NH gradient gap.

The model's NH equator-to-pole gradient is ~22 K (target 40–65 K).  The AMOC
bonus over-warms the NH polar region.  This script runs a Latin Hypercube sweep
over amoc_bonus_far (and optionally amoc_bonus_near) to find values that bring
the gradient closer to the observed 40–65 K without breaking other diagnostics.

Usage
-----
    python scripts/run_amoc_sweep.py [--trials N] [--years Y] [--size H W]

Results are printed as a table sorted by NH gradient.

Example
-------
    python scripts/run_amoc_sweep.py --trials 30 --years 1 --size 32 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _lhs(n_trials: int, bounds: list[tuple[float, float]], seed: int = 42):
    """Latin Hypercube Sampling over [lo, hi] ranges."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n_dims = len(bounds)
    result = np.zeros((n_trials, n_dims))
    for d, (lo, hi) in enumerate(bounds):
        perm = rng.permutation(n_trials)
        result[:, d] = lo + (hi - lo) * (perm + rng.random(n_trials)) / n_trials
    return result


def _run_trial(amoc_near: float, amoc_far: float,
               years: float, H: int, W: int) -> dict:
    from optimizer.headless import run_simulation
    from planet_params import PlanetParams
    from simulate import TimeScaleMode

    pp = PlanetParams(
        amoc_bonus_near=amoc_near,
        amoc_bonus_far=amoc_far,
    )
    state, metrics = run_simulation(
        pp,
        spinup_years=years,
        eval_years=0.25,
        H=H, W=W,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
    )
    return {
        "amoc_near": amoc_near,
        "amoc_far":  amoc_far,
        "gradient_nh": metrics.gradient_nh,
        "gradient_sh": metrics.gradient_sh,
        "global_mean_t": metrics.global_mean_t,
        "ice_nh": metrics.ice_frac_nh,
        "has_nan": metrics.has_nan,
    }


def main():
    parser = argparse.ArgumentParser(description="AMOC bonus sweep")
    parser.add_argument("--trials", type=int, default=30, help="Number of LHS trials")
    parser.add_argument("--years",  type=float, default=1.5, help="Spinup years per trial")
    parser.add_argument("--size",   type=int, nargs=2, default=[32, 64], metavar=("H", "W"))
    parser.add_argument("--near-range", type=float, nargs=2, default=[1.0, 6.0],
                        metavar=("LO", "HI"), help="amoc_bonus_near sweep range [K]")
    parser.add_argument("--far-range",  type=float, nargs=2, default=[4.0, 18.0],
                        metavar=("LO", "HI"), help="amoc_bonus_far sweep range [K]")
    args = parser.parse_args()

    H, W = args.size
    bounds = [tuple(args.near_range), tuple(args.far_range)]
    samples = _lhs(args.trials, bounds)

    print(f"AMOC sweep: {args.trials} trials, {args.years}yr spinup, {H}×{W} grid")
    print(f"  amoc_bonus_near: {args.near_range[0]:.1f}–{args.near_range[1]:.1f} K")
    print(f"  amoc_bonus_far:  {args.far_range[0]:.1f}–{args.far_range[1]:.1f} K")
    print()

    results = []
    for i, (near, far) in enumerate(samples):
        print(f"  [{i+1:3d}/{args.trials}] near={near:.2f}K  far={far:.2f}K  ...", end="", flush=True)
        try:
            r = _run_trial(float(near), float(far), args.years, H, W)
            results.append(r)
            flag = "NaN!" if r["has_nan"] else ""
            print(f"  gradient_nh={r['gradient_nh']:.1f}K  T_mean={r['global_mean_t']:.1f}K  {flag}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Sort by NH gradient descending
    results.sort(key=lambda r: r["gradient_nh"], reverse=True)

    print()
    print("=" * 75)
    print(f"{'near':>6}  {'far':>6}  {'grad_nh':>8}  {'grad_sh':>8}  {'T_mean':>8}  {'ice_nh':>7}")
    print("-" * 75)
    for r in results:
        marker = " <-- target" if r["gradient_nh"] >= 30.0 else ""
        print(
            f"{r['amoc_near']:6.2f}  {r['amoc_far']:6.2f}  "
            f"{r['gradient_nh']:8.1f}  {r['gradient_sh']:8.1f}  "
            f"{r['global_mean_t']:8.1f}  {r['ice_nh']:7.4f}"
            f"{marker}"
        )
    print("=" * 75)

    good = [r for r in results if r["gradient_nh"] >= 30.0 and not r["has_nan"]]
    print(f"\n{len(good)}/{len(results)} trials achieved NH gradient ≥ 30 K")
    if good:
        best = good[0]
        print(f"Best: amoc_near={best['amoc_near']:.2f}K, amoc_far={best['amoc_far']:.2f}K "
              f"→ gradient_nh={best['gradient_nh']:.1f}K")


if __name__ == "__main__":
    main()
