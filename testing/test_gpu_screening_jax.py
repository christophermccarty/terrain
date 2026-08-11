"""Physics-level regression tests for optimizer/jax_screening.py and
optimizer/sweep.py::gpu_random_search.

Requires jax[cuda12] and (for reasonable runtime) a GPU -- skipped entirely
via `pytest.importorskip` on this project's normal Windows dev venv. See
project memory gpu-sweep-screening-phase4-anticorrelated-2026-08-10 for the
WSL2 setup this backend actually runs in, and
testing/test_gpu_screening.py for the always-run half of this test suite
(split out because `pytest.importorskip` at module scope skips the whole
module's collection, not just tests after it).

These follow the same "does a swept parameter actually change the output"
philosophy as test_param_wiring.py -- the exact failure mode that sank v1
of this model (three swept parameters that barely moved the real CPU
model's score at all) is precisely what
test_swept_params_actually_change_output guards against; a silently inert
parameter would otherwise pass every other check here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

jax = pytest.importorskip("jax")


def _default_params(n: int) -> dict[str, np.ndarray]:
    """SUPPORTED_PARAMS at their jax_screening.DEFAULT_PARAMS values."""
    from optimizer import jax_screening
    return {k: np.full(n, v, dtype=np.float32) for k, v in jax_screening.DEFAULT_PARAMS.items()}


def test_supported_params_matches_sweep_wind_json():
    """jax_screening.SUPPORTED_PARAMS must stay in sync with
    optimizer/configs/sweep_wind.json's param_space -- that config is the
    real production sweep this model was recalibrated against. If the two
    drift apart, gpu_random_search's ValueError guard silently stops
    meaning what its docstring says it means."""
    import json
    from optimizer import jax_screening

    config = json.loads((ROOT / "optimizer" / "configs" / "sweep_wind.json").read_text())
    json_params = set(config["param_space"].keys())
    assert json_params == jax_screening.SUPPORTED_PARAMS


def test_wind_baroclinic_jet_amp_range_does_not_saturate_wind():
    """Physics-level half of the stale-range regression guard (see
    testing/test_gpu_screening.py for the config-file half): confirms the
    CURRENT sweep_wind.json upper bound doesn't pin wind speed at
    VMAX_CLIP the way the old ~1e6-scale value did."""
    import json
    from optimizer import jax_screening

    config = json.loads((ROOT / "optimizer" / "configs" / "sweep_wind.json").read_text())
    hi = config["param_space"]["wind_baroclinic_jet_amp"][1]

    params = _default_params(1)
    params["wind_baroclinic_jet_amp"] = np.array([hi], dtype=np.float32)
    metrics = jax_screening.run_batch(params)
    assert metrics["wind_trade_mean"][0] < jax_screening.VMAX_CLIP * 0.5, (
        "wind_trade_mean is pinned near VMAX_CLIP -- likely a reintroduced "
        "baroclinic-mixing blowup"
    )


def test_run_batch_produces_physically_sane_output():
    """Broad sanity bounds, not exact values (those are Phase-4 tuning
    targets and expected to keep moving) -- this just catches "the model
    now silently produces NaN/Inf/nonsense" before a real sweep does."""
    from optimizer import jax_screening

    metrics = jax_screening.run_batch(_default_params(4))

    for field in jax_screening.METRICS_FIELDS:
        values = metrics[field]
        assert np.all(np.isfinite(values)), f"{field} produced non-finite values"

    assert np.all((metrics["global_mean_t"] > 150.0) & (metrics["global_mean_t"] < 400.0))
    assert np.all((metrics["ice_frac_nh"] >= 0.0) & (metrics["ice_frac_nh"] <= 1.0))
    assert np.all((metrics["ice_frac_sh"] >= 0.0) & (metrics["ice_frac_sh"] <= 1.0))
    assert np.all(metrics["mean_precip"] >= 0.0)
    assert np.all(metrics["wind_trade_mean"] >= 0.0)
    assert np.all(metrics["wind_midlat_mean"] >= 0.0)


def test_swept_params_actually_change_output():
    """Same philosophy as test_param_wiring.py: run at each parameter's
    default vs. a substantially perturbed value and assert the output
    differs beyond a noise floor -- the exact failure mode that sank v1 of
    this model (see module docstring)."""
    from optimizer import jax_screening

    lo_params = _default_params(1)
    hi_params = _default_params(1)
    for name, (lo, hi) in {
        "wind_damping": (0.25, 0.75),
        "wind_baroclinic_jet_amp": (0.3, 3.0),
        "wind_pgf_temp_scale": (200.0, 800.0),
        "wind_cell_relax_days": (1.5, 6.0),
    }.items():
        lo_params[name] = np.array([lo], dtype=np.float32)
        hi_params[name] = np.array([hi], dtype=np.float32)

    lo_metrics = jax_screening.run_batch(lo_params)
    hi_metrics = jax_screening.run_batch(hi_params)

    changed = any(
        abs(float(lo_metrics[f][0]) - float(hi_metrics[f][0])) > 1e-3
        for f in jax_screening.METRICS_FIELDS
    )
    assert changed, "sweeping every parameter simultaneously produced no measurable change"


def test_gpu_random_search_rejects_unsupported_param():
    from optimizer.sweep import gpu_random_search

    with pytest.raises(ValueError, match="does not support sweeping"):
        gpu_random_search({"moisture_advection_scale": (0.0, 1.0)}, n_samples=2)


def test_gpu_random_search_allows_partial_param_space():
    """Found by this test suite 2026-08-11: gpu_random_search used to
    require every SUPPORTED_PARAMS name to be present in param_space and
    KeyError'd otherwise. CPU-backend random_search allows sweeping a
    subset of simulate_step's kwargs (the rest keep their own defaults);
    gpu_random_search needs the same behavior -- fixed via
    jax_screening.DEFAULT_PARAMS filling in whatever isn't swept."""
    from optimizer.sweep import gpu_random_search

    results = gpu_random_search({"wind_damping": (0.25, 0.75)}, n_samples=2, seed=0)
    rows = results.to_dict("records") if hasattr(results, "to_dict") else results
    assert len(rows) == 2


def test_gpu_random_search_output_matches_cpu_backend_shape():
    """Same result-row shape as random_search's CPU backend (trial_id,
    score, elapsed_s, param_*, metric_*, contrib_*) -- the documented
    contract that lets CSVs from the two backends be compared/appended."""
    from optimizer.sweep import gpu_random_search
    from optimizer.scoring import ClimateMetrics

    space = {"wind_damping": (0.25, 0.75), "wind_pgf_temp_scale": (200.0, 800.0)}
    results = gpu_random_search(space, n_samples=3, seed=0)
    rows = results.to_dict("records") if hasattr(results, "to_dict") else results
    assert len(rows) == 3

    row = rows[0]
    for key in ("trial_id", "score", "elapsed_s"):
        assert key in row
    for name in space:
        assert f"param_{name}" in row
    for field in vars(ClimateMetrics()):
        assert f"metric_{field}" in row
