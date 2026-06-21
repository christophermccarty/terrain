"""Parameter sweep utilities: random search + latin hypercube sampling.

A ``ParamSpace`` maps parameter names to ``(lo, hi)`` bounds. The sweep
functions sample from that space and score each configuration using
``headless.run_simulation`` + ``scoring.ClimateScore``.

Results are written incrementally to a CSV file so a sweep can be resumed
or inspected while it is still running.

Usage
-----
from optimizer.sweep import ParamSpace, random_search

space: ParamSpace = {
    "thermal_diffusion":   (0.01, 0.10),
    "ice_albedo_strength": (0.10, 0.50),
    "wind_damping":        (0.25, 0.75),
}
df = random_search(space, n_samples=50, n_jobs=4, output_csv="results/sweep.csv")
"""
from __future__ import annotations

import csv
import itertools
import multiprocessing
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import PlanetParams, EARTH
from simulate import TimeScaleMode
from optimizer.scoring import ClimateScore, ClimateMetrics, EARTH_REFERENCE, ReferenceClimate
from optimizer.headless import run_simulation

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ParamSpace = dict[str, tuple[float, float]]
"""Map of parameter name → (lower_bound, upper_bound)."""


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def latin_hypercube_sample(
    param_space: ParamSpace,
    n_samples: int,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Generate n_samples parameter dicts using latin hypercube sampling.

    Falls back to uniform random if scipy is not available.
    """
    names = list(param_space.keys())
    k = len(names)

    try:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=k, seed=seed)
        unit_samples = sampler.random(n=n_samples)  # shape (n_samples, k)
    except ImportError:
        warnings.warn(
            "scipy not installed; falling back to uniform random sampling.",
            stacklevel=2,
        )
        rng = np.random.default_rng(seed)
        unit_samples = rng.random((n_samples, k))

    configs: list[dict[str, float]] = []
    for row in unit_samples:
        config = {}
        for i, name in enumerate(names):
            lo, hi = param_space[name]
            config[name] = float(lo + row[i] * (hi - lo))
        configs.append(config)
    return configs


def random_uniform_sample(
    param_space: ParamSpace,
    n_samples: int,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Generate n_samples parameter dicts using independent uniform draws."""
    rng = np.random.default_rng(seed)
    names = list(param_space.keys())
    configs: list[dict[str, float]] = []
    for _ in range(n_samples):
        config = {}
        for name in names:
            lo, hi = param_space[name]
            config[name] = float(rng.uniform(lo, hi))
        configs.append(config)
    return configs


# ---------------------------------------------------------------------------
# Worker function (module-level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> dict[str, Any]:
    """Single-sample evaluation worker. Returns a result dict."""
    (trial_id, config, planet_params_kwargs, run_kwargs, ref_kwargs) = args
    try:
        pp = PlanetParams(**planet_params_kwargs) if planet_params_kwargs else EARTH
        ref = ReferenceClimate(**ref_kwargs) if ref_kwargs else EARTH_REFERENCE
        score_fn = ClimateScore(ref)

        t0 = time.perf_counter()
        _state, metrics = run_simulation(planet_params=pp, **run_kwargs, **config)
        elapsed = time.perf_counter() - t0

        score = score_fn.score(metrics)
        bd = score_fn.breakdown(metrics)

        result = {"trial_id": trial_id, "score": score, "elapsed_s": elapsed}
        result.update({f"param_{k}": v for k, v in config.items()})
        result.update({f"metric_{k}": getattr(metrics, k) for k in vars(metrics)})
        result.update({f"contrib_{k}": v for k, v in bd.items() if k != "total"})
        return result
    except Exception as exc:
        return {
            "trial_id": trial_id,
            "score": -1.0,
            "elapsed_s": 0.0,
            "error": str(exc),
            **{f"param_{k}": v for k, v in config.items()},
        }


# ---------------------------------------------------------------------------
# Public search functions
# ---------------------------------------------------------------------------

def random_search(
    param_space: ParamSpace,
    n_samples: int = 50,
    *,
    n_jobs: int = 1,
    output_csv: str | Path | None = None,
    planet_params: PlanetParams = EARTH,
    reference: ReferenceClimate = EARTH_REFERENCE,
    seed: int = 0,
    use_lhs: bool = True,
    **run_kwargs: Any,
) -> "list[dict] | pd.DataFrame":
    """Sample n_samples configurations from param_space and score each.

    Parameters
    ----------
    param_space:
        Dict of ``{param_name: (lo, hi)}``. Only numeric simulate_step kwargs.
    n_samples:
        Number of configurations to evaluate.
    n_jobs:
        Number of parallel worker processes. Set to 1 to disable multiprocessing
        (required when calling from an interactive session or Jupyter).
    output_csv:
        Path to write results incrementally (created / appended). If None,
        no file is written.
    planet_params:
        PlanetParams instance passed to every worker.
    reference:
        ReferenceClimate to use for scoring.
    seed:
        RNG seed for reproducible sampling.
    use_lhs:
        Use latin hypercube sampling (True) or pure uniform random (False).
    **run_kwargs:
        Extra keyword arguments forwarded to ``run_simulation`` (e.g.
        ``spinup_years``, ``eval_years``, ``H``, ``W``).

    Returns
    -------
    pandas.DataFrame if pandas is installed; list[dict] otherwise.
    Sorted by score descending.
    """
    if use_lhs:
        configs = latin_hypercube_sample(param_space, n_samples, seed=seed)
    else:
        configs = random_uniform_sample(param_space, n_samples, seed=seed)

    # Serialise PlanetParams + ReferenceClimate for multiprocessing
    pp_kwargs = {
        f.name: getattr(planet_params, f.name)
        for f in planet_params.__dataclass_fields__.values()  # type: ignore[attr-defined]
    } if planet_params is not EARTH else {}

    ref_kwargs: dict = {}  # EARTH_REFERENCE = default; no need to serialise

    worker_args = [
        (i, cfg, pp_kwargs, run_kwargs, ref_kwargs)
        for i, cfg in enumerate(configs)
    ]

    csv_writer = None
    csv_file = None
    fieldnames: list[str] | None = None

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = output_csv.open("w", newline="")
        csv_writer = None  # initialised on first result

    results: list[dict] = []

    def _handle_result(res: dict) -> None:
        nonlocal csv_writer, fieldnames
        results.append(res)
        if csv_writer is None and csv_file is not None:
            fieldnames = list(res.keys())
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
            csv_writer.writeheader()
        if csv_writer is not None:
            csv_writer.writerow({k: res.get(k, "") for k in (fieldnames or [])})
            csv_file.flush()  # type: ignore[union-attr]

    try:
        if n_jobs == 1:
            for args in worker_args:
                _handle_result(_worker(args))
        else:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=n_jobs) as pool:
                for res in pool.imap_unordered(_worker, worker_args):
                    _handle_result(res)
    finally:
        if csv_file is not None:
            csv_file.close()

    results.sort(key=lambda r: float(r.get("score", -1.0)), reverse=True)

    if _PANDAS:
        import pandas as pd  # noqa: PLC0415
        return pd.DataFrame(results)
    return results


def grid_search(
    param_grid: dict[str, list[float]],
    *,
    n_jobs: int = 1,
    output_csv: str | Path | None = None,
    planet_params: PlanetParams = EARTH,
    reference: ReferenceClimate = EARTH_REFERENCE,
    **run_kwargs: Any,
) -> "list[dict] | pd.DataFrame":
    """Exhaustive grid search over all combinations in param_grid.

    Parameters
    ----------
    param_grid:
        Dict of ``{param_name: [value1, value2, ...]}``. All combinations
        are evaluated.
    """
    names = list(param_grid.keys())
    value_lists = [param_grid[n] for n in names]
    configs = [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]

    pp_kwargs: dict = {}
    ref_kwargs: dict = {}

    worker_args = [
        (i, cfg, pp_kwargs, run_kwargs, ref_kwargs)
        for i, cfg in enumerate(configs)
    ]

    results: list[dict] = []
    if n_jobs == 1:
        for args in worker_args:
            results.append(_worker(args))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            results = list(pool.imap(_worker, worker_args))

    if output_csv is not None:
        _write_csv(results, Path(output_csv))

    results.sort(key=lambda r: float(r.get("score", -1.0)), reverse=True)
    if _PANDAS:
        import pandas as pd  # noqa: PLC0415
        return pd.DataFrame(results)
    return results


def _write_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
