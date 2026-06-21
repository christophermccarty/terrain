"""run_ecs_experiment.py — Measure the model's Equilibrium Climate Sensitivity (ECS).

ECS is the global mean temperature change after doubling atmospheric CO2 from
pre-industrial levels (280 ppm -> 560 ppm) once the climate reaches equilibrium.

Earth's observed ECS is approximately 2.5-4.0 K/doubling (IPCC AR6 best estimate: 3.0 K).

Method
------
1. Spinup 3yr MONTHLY + 50yr ANNUAL at 280 ppm (pre-industrial baseline).
2. Spinup 3yr MONTHLY + 50yr ANNUAL at 560 ppm (2x CO2).
3. ECS = mean(T_final_560) - mean(T_final_280), averaged over the last 10yr of each run.

Usage
-----
    python scripts/run_ecs_experiment.py [--years N] [--size H W]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run(co2_ppm: float, years: int, H: int, W: int, spinup: float) -> list[dict]:
    from optimizer.headless import run_long_simulation
    from planet_params import PlanetParams
    pp = PlanetParams(co2_initial_ppm=co2_ppm)
    _, records = run_long_simulation(
        pp,
        years=years,
        H=H, W=W,
        spinup_years=spinup,
        sample_every=5,
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years",   type=int, default=50, help="ANNUAL years after spinup")
    parser.add_argument("--spinup",  type=float, default=3.0, help="MONTHLY spinup years")
    parser.add_argument("--H",       type=int, default=32, help="Grid rows")
    parser.add_argument("--W",       type=int, default=64, help="Grid cols")
    args = parser.parse_args()

    print(f"ECS experiment: {args.years}yr ANNUAL, {args.spinup}yr MONTHLY spinup, "
          f"{args.H}x{args.W} grid")
    print("Running 280 ppm (pre-industrial) ...")
    rec_280 = _run(280.0, args.years, args.H, args.W, args.spinup)

    print("Running 560 ppm (2x CO2) ...")
    rec_560 = _run(560.0, args.years, args.H, args.W, args.spinup)

    if not rec_280 or not rec_560:
        print("ERROR: no records returned")
        sys.exit(1)

    # Year-by-year temperature table
    print(f"\n{'Year':>5}  {'T_280 (K)':>10}  {'T_560 (K)':>10}  {'dT (K)':>8}")
    print("-" * 42)
    for r280, r560 in zip(rec_280, rec_560):
        yr = int(r280.get("year", 0))
        T280 = r280.get("global_mean_t", float("nan"))
        T560 = r560.get("global_mean_t", float("nan"))
        dT = T560 - T280
        print(f"{yr:>5}  {T280:>10.3f}  {T560:>10.3f}  {dT:>8.3f}")

    # ECS from last 10yr average
    tail = max(2, len(rec_280) // 5)
    T280_mean = sum(r.get("global_mean_t", 0.0) for r in rec_280[-tail:]) / tail
    T560_mean = sum(r.get("global_mean_t", 0.0) for r in rec_560[-tail:]) / tail
    ecs = T560_mean - T280_mean

    print(f"\n--- ECS Result ---")
    print(f"280 ppm mean T (last {tail} records): {T280_mean:.3f} K")
    print(f"560 ppm mean T (last {tail} records): {T560_mean:.3f} K")
    print(f"ECS = {ecs:.3f} K / CO2-doubling")

    if 2.0 <= ecs <= 5.5:
        verdict = "PLAUSIBLE  (Earth range: 2.5-4.0 K; IPCC AR6 best: 3.0 K)"
    elif ecs < 2.0:
        verdict = "TOO LOW  (model under-sensitive to CO2 forcing)"
    else:
        verdict = "TOO HIGH  (model over-sensitive to CO2 forcing)"
    print(f"Verdict: {verdict}")

    # CO2 drift check
    co2_280_final = rec_280[-1].get("co2_ppm", float("nan"))
    co2_560_final = rec_560[-1].get("co2_ppm", float("nan"))
    print(f"\nCO2 drift: 280 ppm start -> {co2_280_final:.1f} ppm final; "
          f"560 ppm start -> {co2_560_final:.1f} ppm final")


if __name__ == "__main__":
    main()
