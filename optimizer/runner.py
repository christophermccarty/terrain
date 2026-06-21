"""CLI entry point for the PlanetSim optimizer.

Runs parameter sweeps, Bayesian optimisation, or a single scored simulation
from the command line — no GUI required.

Usage
-----
# Single scored run (Earth defaults)
python optimizer/runner.py --mode single

# Random sweep (50 samples, 4 parallel workers, LHS sampling)
python optimizer/runner.py --mode sweep --config optimizer/configs/sweep_wind.json \\
    --output results/wind_sweep.csv --jobs 4 --samples 50

# Bayesian optimisation (100 trials)
python optimizer/runner.py --mode bayes --config optimizer/configs/sweep_wind.json \\
    --output results/bayes_wind.csv --trials 100

# Can also be called as a module:
python -m optimizer.runner --mode single
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import EARTH
from simulate import TimeScaleMode
from optimizer.scoring import ClimateScore, EARTH_REFERENCE
from optimizer.headless import run_simulation


_MODE_CHOICES = ("single", "sweep", "bayes")

_SCALE_MAP = {
    "daily":   TimeScaleMode.DAILY,
    "weekly":  TimeScaleMode.WEEKLY,
    "monthly": TimeScaleMode.MONTHLY,
    "annual":  TimeScaleMode.ANNUAL,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="optimizer",
        description="PlanetSim headless parameter optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=_MODE_CHOICES, default="single",
                   help="Run mode")
    p.add_argument("--config", type=Path, default=None,
                   help="JSON file defining the parameter space (required for sweep/bayes)")
    p.add_argument("--output", type=Path, default=Path("results/optimizer_out.csv"),
                   help="Output CSV path for sweep/bayes results")
    p.add_argument("--jobs", type=int, default=1,
                   help="Number of parallel worker processes (sweep/bayes only)")
    p.add_argument("--samples", type=int, default=50,
                   help="Number of samples for random sweep")
    p.add_argument("--trials", type=int, default=100,
                   help="Number of trials for Bayesian optimisation")
    p.add_argument("--spinup-years", type=float, default=2.0,
                   help="Spinup duration in years")
    p.add_argument("--eval-years", type=float, default=1.0,
                   help="Evaluation window in years")
    p.add_argument("--spinup-scale", choices=list(_SCALE_MAP), default="monthly",
                   help="Time-scale mode for spinup phase")
    p.add_argument("--eval-scale", choices=list(_SCALE_MAP), default="daily",
                   help="Time-scale mode for eval phase")
    p.add_argument("--H", type=int, default=60, help="Grid height (rows)")
    p.add_argument("--W", type=int, default=120, help="Grid width (columns)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--no-lhs", action="store_true",
                   help="Use uniform random instead of latin hypercube sampling")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _run_single(args: argparse.Namespace) -> None:
    """Score a single simulation with Earth defaults (or config overrides)."""
    physics_kwargs: dict = {}
    if args.config is not None:
        data = json.loads(args.config.read_text())
        physics_kwargs = data.get("fixed_params", data)

    print("Running single scored simulation …")
    t0 = time.perf_counter()
    _state, metrics = run_simulation(
        planet_params=EARTH,
        spinup_years=args.spinup_years,
        eval_years=args.eval_years,
        H=args.H,
        W=args.W,
        spinup_time_scale=_SCALE_MAP[args.spinup_scale],
        eval_time_scale=_SCALE_MAP[args.eval_scale],
        **physics_kwargs,
    )
    elapsed = time.perf_counter() - t0

    score_fn = ClimateScore(EARTH_REFERENCE)
    score = score_fn.score(metrics)
    bd = score_fn.breakdown(metrics)

    print(f"\n{'-'*50}")
    print(f"  Overall score: {score:.1f} / 100   ({elapsed:.1f}s)")
    print(f"{'-'*50}")
    print(f"  {'Metric':<28} {'Value':>10}  {'Contrib':>8}")
    print(f"  {'------':<28} {'-----':>10}  {'-------':>8}")

    metric_display = {
        "global_mean_t":         (f"{metrics.global_mean_t:.1f} K",),
        "gradient_nh":           (f"{metrics.gradient_nh:.1f} K",),
        "gradient_sh":           (f"{metrics.gradient_sh:.1f} K",),
        "ice_frac_nh":           (f"{metrics.ice_frac_nh*100:.1f} %",),
        "ice_frac_sh":           (f"{metrics.ice_frac_sh*100:.1f} %",),
        "mean_precip":           (f"{metrics.mean_precip:.2f} mm/d",),
        "wind_trade_mean":       (f"{metrics.wind_trade_mean:.1f} m/s",),
        "wind_midlat_mean":      (f"{metrics.wind_midlat_mean:.1f} m/s",),
        "wind_itcz_conv":        (f"{metrics.wind_itcz_conv:.3f}",),
        "seasonal_amplitude_nh": (f"{metrics.seasonal_amplitude_nh:.1f} K",),
    }

    for name, (val_str,) in metric_display.items():
        contrib = bd.get(name, 0.0)
        print(f"  {name:<28} {val_str:>10}  {contrib:>7.2f}")

    print(f"{'-'*50}")
    if metrics.has_nan:
        print("  WARNING: simulation produced NaN values.")
    if metrics.has_inf:
        print("  WARNING: simulation produced Inf values.")
    print()


def _run_sweep(args: argparse.Namespace) -> None:
    from optimizer.sweep import ParamSpace, random_search
    from optimizer.results import summarize, top_n

    if args.config is None:
        print("ERROR: --config is required for sweep mode.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(args.config.read_text())
    param_space: ParamSpace = {
        name: tuple(bounds)  # type: ignore[misc]
        for name, bounds in data.get("param_space", data).items()
    }
    fixed_params: dict = data.get("fixed_params", {})

    print(f"Sweep: {len(param_space)} parameters, {args.samples} samples, {args.jobs} jobs")
    print(f"  Spinup: {args.spinup_years}yr @ {args.spinup_scale}")
    print(f"  Eval:   {args.eval_years}yr @ {args.eval_scale}")
    print(f"  Output: {args.output}")
    print()

    run_kwargs = {
        "spinup_years": args.spinup_years,
        "eval_years": args.eval_years,
        "H": args.H,
        "W": args.W,
        "spinup_time_scale": _SCALE_MAP[args.spinup_scale],
        "eval_time_scale": _SCALE_MAP[args.eval_scale],
        **fixed_params,
    }

    results = random_search(
        param_space,
        n_samples=args.samples,
        n_jobs=args.jobs,
        output_csv=args.output,
        seed=args.seed,
        use_lhs=not args.no_lhs,
        **run_kwargs,
    )

    summary = summarize(results)
    print(f"\nSweep complete: {summary['n']} trials")
    print(f"  Score range: {summary['min']:.1f} – {summary['max']:.1f}")
    print(f"  Mean ± std:  {summary['mean']:.1f} ± {summary['std']:.1f}")
    print(f"  Above 65:    {summary['n_above_65']}")
    print(f"  Above 80:    {summary['n_above_80']}")
    print(f"\nTop-5 configurations:")
    for row in list(top_n(results, n=5)):
        if isinstance(row, dict):
            score = row.get("score", -1.0)
            params = {k[6:]: v for k, v in row.items() if k.startswith("param_")}
            print(f"  {float(score):>6.1f}  {params}")


def _run_bayes(args: argparse.Namespace) -> None:
    from optimizer.bayesian import bayesian_search
    from optimizer.results import summarize, top_n

    if args.config is None:
        print("ERROR: --config is required for bayes mode.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(args.config.read_text())
    param_space = {name: tuple(bounds) for name, bounds in data.get("param_space", data).items()}
    fixed_params: dict = data.get("fixed_params", {})

    print(f"Bayesian search: {len(param_space)} parameters, {args.trials} trials")

    run_kwargs = {
        "spinup_years": args.spinup_years,
        "eval_years": args.eval_years,
        "H": args.H,
        "W": args.W,
        "spinup_time_scale": _SCALE_MAP[args.spinup_scale],
        "eval_time_scale": _SCALE_MAP[args.eval_scale],
        **fixed_params,
    }

    results = bayesian_search(
        param_space,
        n_trials=args.trials,
        n_jobs=args.jobs,
        output_csv=args.output,
        seed=args.seed,
        **run_kwargs,
    )

    summary = summarize(results)
    print(f"\nBayes complete: {summary['n']} trials, best={summary['max']:.1f}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode == "single":
        _run_single(args)
    elif args.mode == "sweep":
        _run_sweep(args)
    elif args.mode == "bayes":
        _run_bayes(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
