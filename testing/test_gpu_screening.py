"""Always-run regression tests for the GPU-batched optimizer-sweep
screening backend (optimizer/jax_screening.py, optimizer/sweep.py::
gpu_random_search, optimizer/scoring.py::WIND_SCREENING_REFERENCE) that
need no jax/GPU dependency.

See testing/test_gpu_screening_jax.py for the physics-level tests, which
require jax[cuda12] and are skipped on this project's normal Windows dev
venv (see project memory gpu-sweep-screening-phase4-anticorrelated-
2026-08-10 for the WSL2 setup this backend actually runs in). Kept in a
separate file because `pytest.importorskip` skips a module's entire
collection, not just the tests after it -- these tests need to survive on
a machine with no jax installed at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_sweep_module_importable_without_jax():
    """optimizer.sweep must stay importable on a machine with no jax/GPU.

    gpu_random_search's `from optimizer import jax_screening` has to stay
    inside the function body, not module level -- otherwise every caller
    of optimizer.sweep (including the CPU-only random_search/grid_search
    paths) would break without jax installed.
    """
    from optimizer import sweep
    assert hasattr(sweep, "gpu_random_search")
    assert hasattr(sweep, "random_search")  # CPU backend must be unaffected


def test_wind_screening_reference_weights():
    """Lock in which metrics WIND_SCREENING_REFERENCE trusts and which it
    doesn't -- see that constant's docstring (optimizer/scoring.py) for why
    each one is zeroed or kept. A silent change here would re-open the
    ceiling-tie / wrong-signal bugs that constant was built to fix."""
    from optimizer.scoring import WIND_SCREENING_REFERENCE

    zero_weight_fields = [
        "global_mean_t", "gradient_nh", "gradient_sh",
        "ice_frac_nh", "ice_frac_sh", "mean_precip", "seasonal_amplitude_nh",
    ]
    for field in zero_weight_fields:
        _, _, weight = getattr(WIND_SCREENING_REFERENCE, field)
        assert weight == 0.0, f"{field} should carry zero weight in WIND_SCREENING_REFERENCE"

    nonzero_weight_fields = ["wind_trade_mean", "wind_midlat_mean", "wind_itcz_conv"]
    for field in nonzero_weight_fields:
        _, _, weight = getattr(WIND_SCREENING_REFERENCE, field)
        assert weight > 0.0, f"{field} should carry nonzero weight in WIND_SCREENING_REFERENCE"


def test_gpu_random_search_default_reference_is_earth():
    """gpu_random_search defaults to EARTH_REFERENCE (validated at Spearman
    0.708 against the real CPU model -- see project memory
    gpu-sweep-screening-phase4-anticorrelated-2026-08-10), NOT
    WIND_SCREENING_REFERENCE (an earlier, weaker fallback at 0.338). A
    silent default change here would quietly downgrade every future
    GPU-backend sweep's score quality."""
    import inspect
    from optimizer.sweep import gpu_random_search
    from optimizer.scoring import EARTH_REFERENCE

    default = inspect.signature(gpu_random_search).parameters["reference"].default
    assert default is EARTH_REFERENCE


def test_sweep_wind_json_baroclinic_jet_amp_not_stale():
    """Regression guard for the stale-range bug found 2026-08-11:
    optimizer/configs/sweep_wind.json used to sweep wind_baroclinic_jet_amp
    over [200000, 3000000], left over from before a 2026-07-04 semantic
    rescale (see testing/test_param_wiring.py:71) that turned it into a
    dimensionless coupling-strength multiplier (current simulate_step
    default: 1.0). No jax needed for this half of the check -- see
    test_gpu_screening_jax.py::test_wind_baroclinic_jet_amp_range_does_not_
    saturate_wind for the physics-level half."""
    import json

    config = json.loads((ROOT / "optimizer" / "configs" / "sweep_wind.json").read_text())
    lo, hi = config["param_space"]["wind_baroclinic_jet_amp"]
    assert hi < 100.0, (
        "wind_baroclinic_jet_amp's upper bound looks like the pre-2026-07-04 "
        "~1e6 scale again -- see this test's docstring"
    )


def test_earth_params_json_baroclinic_jet_amp_not_stale():
    """Same bug, second location: optimizer/configs/earth_params.json
    claims to capture simulate_step's actual current defaults but had
    wind_baroclinic_jet_amp=1000000.0, predating the same 2026-07-04
    rescale. Current simulate_step default is 1.0 (simulate.py:1026)."""
    import json

    config = json.loads((ROOT / "optimizer" / "configs" / "earth_params.json").read_text())
    value = config["fixed_params"]["wind_baroclinic_jet_amp"]
    assert value < 100.0, (
        "earth_params.json's wind_baroclinic_jet_amp looks like the "
        "pre-2026-07-04 ~1e6 scale again -- see this test's docstring"
    )
