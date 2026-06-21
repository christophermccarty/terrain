"""Bayesian hyperparameter optimisation using optuna (optional dependency).

If optuna is not installed, all functions silently fall back to
``sweep.random_search`` with latin hypercube sampling.

Install optuna:
    pip install optuna

Usage
-----
from optimizer.bayesian import bayesian_search
from optimizer.sweep import ParamSpace

space: ParamSpace = {
    "thermal_diffusion":   (0.01, 0.10),
    "ice_albedo_strength": (0.10, 0.50),
}
df = bayesian_search(space, n_trials=100, n_jobs=1)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from planet_params import PlanetParams, EARTH
from optimizer.scoring import ClimateScore, EARTH_REFERENCE, ReferenceClimate
from optimizer.sweep import ParamSpace, random_search, _worker
from optimizer.headless import run_simulation

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False


def _is_optuna_available() -> bool:
    return _OPTUNA


def bayesian_search(
    param_space: ParamSpace,
    n_trials: int = 100,
    *,
    n_jobs: int = 1,
    output_csv: str | Path | None = None,
    planet_params: PlanetParams = EARTH,
    reference: ReferenceClimate = EARTH_REFERENCE,
    seed: int = 0,
    pruner_startup_trials: int = 10,
    pruner_warmup_steps: int = 5,
    **run_kwargs: Any,
) -> Any:
    """Run Bayesian optimisation over param_space.

    Uses optuna's TPESampler with a median pruner (stops runs that diverge
    early). Falls back to random_search if optuna is not installed.

    Parameters
    ----------
    param_space:
        Dict of ``{param_name: (lo, hi)}``.
    n_trials:
        Total number of trials to run.
    n_jobs:
        Parallel jobs. Optuna supports parallel studies with shared storage;
        for in-memory search (default), set n_jobs=1.
    output_csv:
        Optional path to write results incrementally.
    planet_params:
        PlanetParams for all trials.
    reference:
        Reference climate for scoring.
    seed:
        RNG seed for the sampler.
    pruner_startup_trials:
        Number of trials before the median pruner activates.
    pruner_warmup_steps:
        Pruning steps before each trial's intermediate reports start counting.
    **run_kwargs:
        Forwarded to ``run_simulation``.

    Returns
    -------
    pandas.DataFrame or list[dict] sorted by score descending.
    """
    if not _OPTUNA:
        warnings.warn(
            "optuna is not installed. Falling back to random_search with LHS sampling.\n"
            "Install with: pip install optuna",
            stacklevel=2,
        )
        return random_search(
            param_space,
            n_samples=n_trials,
            n_jobs=n_jobs,
            output_csv=output_csv,
            planet_params=planet_params,
            reference=reference,
            seed=seed,
            use_lhs=True,
            **run_kwargs,
        )

    score_fn = ClimateScore(reference)

    def objective(trial: "optuna.Trial") -> float:
        config = {
            name: trial.suggest_float(name, lo, hi)
            for name, (lo, hi) in param_space.items()
        }
        try:
            _state, metrics = run_simulation(
                planet_params=planet_params, **run_kwargs, **config
            )
            s = score_fn.score(metrics)
            return float(s)
        except Exception:
            return 0.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_startup_trials,
        n_warmup_steps=pruner_warmup_steps,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

    # Collect all trial results
    results: list[dict] = []
    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue
        row: dict[str, Any] = {
            "trial_id": trial.number,
            "score": trial.value or 0.0,
        }
        row.update({f"param_{k}": v for k, v in trial.params.items()})
        results.append(row)

    results.sort(key=lambda r: float(r.get("score", -1.0)), reverse=True)

    if output_csv is not None:
        from optimizer.results import save_results
        save_results(results, output_csv, metadata={"n_trials": n_trials, "method": "bayesian"})

    try:
        import pandas as pd
        return pd.DataFrame(results)
    except ImportError:
        return results
