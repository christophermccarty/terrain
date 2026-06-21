"""run_gradient_sweep.py — LHS sweep targeting the NH equator-pole temperature gradient.

The Earth score's biggest gap is gradient_nh (~22 K measured vs 40–65 K target).
This script sweeps the four parameters that most directly control that gradient:

  thermal_diffusion     — lower values let poles cool more freely
  ice_albedo_strength   — higher values make ice colder (ice-albedo feedback)
  polar_cooling_scale   — higher values add direct radiative polar cooling
  ocean_transport_coeff — lower values reduce poleward ocean heat flux

Results are written to optimizer/results/gradient_sweep.csv.
The top-5 configurations by gradient_nh contribution are printed at the end.

Usage
-----
    python scripts/run_gradient_sweep.py [--trials N] [--jobs J]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.sweep import random_search, ParamSpace
from optimizer.scoring import EARTH_REFERENCE, ReferenceClimate
from planet_params import EARTH

PARAM_SPACE: ParamSpace = {
    "thermal_diffusion":    (0.005, 0.045),
    "ice_albedo_strength":  (0.25,  0.65),
    "polar_cooling_scale":  (0.20,  0.70),
    "ocean_transport_coeff":(0.10,  0.48),
}

# Tighter reference so gradient gets full weight even at moderate values
GRADIENT_REF = ReferenceClimate(
    gradient_nh=(38.0, 65.0, 2.0),   # up-weight gradient
    gradient_sh=(36.0, 62.0, 1.5),
    global_mean_t=(284.0, 292.0, 1.5),
    ice_frac_nh=(0.02, 0.12, 1.0),
    ice_frac_sh=(0.02, 0.12, 0.8),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=60, help="Number of LHS samples")
    parser.add_argument("--jobs",   type=int, default=4,  help="Parallel worker processes")
    parser.add_argument("--out",    type=str,
                        default=str(ROOT / "optimizer" / "results" / "gradient_sweep.csv"))
    args = parser.parse_args()

    print(f"Running {args.trials} trials across {args.jobs} workers …")
    print(f"Output -> {args.out}\n")

    results = random_search(
        PARAM_SPACE,
        n_samples=args.trials,
        n_jobs=args.jobs,
        output_csv=args.out,
        planet_params=EARTH,
        reference=GRADIENT_REF,
        seed=7,
        H=32, W=64,
        spinup_years=2.0,
        eval_years=0.5,
    )

    # --- Print results ---
    if hasattr(results, "to_dict"):
        rows = results.to_dict("records")
    else:
        rows = results

    errors = [r for r in rows if r.get("score", 0) < 0]
    if errors:
        print(f"WARNING: {len(errors)} trial(s) errored:")
        for e in errors[:3]:
            print(f"  {e.get('error','?')}")

    ok = [r for r in rows if r.get("score", -1) >= 0]
    ok.sort(key=lambda r: r.get("metric_gradient_nh", 0.0), reverse=True)

    print(f"\nTop-5 by gradient_nh (of {len(ok)} successful trials):\n")
    print(f"{'#':>3}  {'score':>6}  {'grad_nh':>8}  {'grad_sh':>8}  "
          f"{'T_mean':>7}  {'td':>6}  {'ia':>6}  {'pc':>6}  {'oc':>6}")
    print("-" * 72)
    for i, r in enumerate(ok[:5], 1):
        print(
            f"{i:>3}  {r.get('score',0):>6.1f}  "
            f"{r.get('metric_gradient_nh',0):>8.2f}  "
            f"{r.get('metric_gradient_sh',0):>8.2f}  "
            f"{r.get('metric_global_mean_t',0):>7.2f}  "
            f"{r.get('param_thermal_diffusion',0):>6.4f}  "
            f"{r.get('param_ice_albedo_strength',0):>6.3f}  "
            f"{r.get('param_polar_cooling_scale',0):>6.3f}  "
            f"{r.get('param_ocean_transport_coeff',0):>6.3f}"
        )

    if ok:
        best = ok[0]
        print("\nBest configuration for gradient_nh:")
        for k in PARAM_SPACE:
            print(f"  {k}: {best.get(f'param_{k}', '?'):.4f}")
        print(f"  gradient_nh    = {best.get('metric_gradient_nh', 0):.2f} K")
        print(f"  global_mean_t  = {best.get('metric_global_mean_t', 0):.2f} K")
        print(f"  score          = {best.get('score', 0):.1f}")

    print(f"\nFull results: {args.out}")


if __name__ == "__main__":
    main()
