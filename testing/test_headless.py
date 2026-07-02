"""test_headless.py — Headless-vs-threaded step parity (PLAN.md Phase 4/6 gap).

Verifies that `optimizer.headless`'s `_advance_one_cycle` (the headless call path)
and `main.py`'s `SimulationThread.run()` (the GUI call path) assemble the same
effective `simulate_step` calls and therefore produce identical results, given
the same initial state and explicit parameters.

Scope note: Python threading itself cannot introduce numerical divergence here —
`SimulationThread` is a single writer with no concurrent access to shared mutable
state, so there is nothing for real OS-thread scheduling to perturb. The actual
risk this test guards against is the two *code paths* (main.py vs
optimizer/headless.py) silently drifting apart in which kwargs they pass to
`simulate_step` (e.g. one path forgetting `wind_block_size`) — so this test
replicates `SimulationThread.run()`'s exact substep/kwargs logic directly rather
than spinning a real background thread, which would just add timing-based
flakiness without testing anything Python's own `threading` module doesn't
already guarantee.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _threaded_style_steps(state, mode, wind_block_size: int, n_cycles: int):
    """Replicates `SimulationThread.run()`'s inner loop exactly (see main.py)."""
    from simulate import simulate_step, TimeScaleMode

    if mode == TimeScaleMode.WEEKLY:
        substeps = [(1.0, True)] * 7
    elif mode == TimeScaleMode.MONTHLY:
        substeps = [(6.0, False)] * 5
    elif mode == TimeScaleMode.ANNUAL:
        substeps = [(7.0, False)] * 52
    else:
        substeps = [(1.0, True)]

    for _ in range(n_cycles):
        for step_days, do_wind in substeps:
            state, _ = simulate_step(
                state,
                days=step_days,
                wind_block_size=wind_block_size,
                update_wind=do_wind,
                debug_log=False,
                track_components=False,
                time_scale=mode,
            )
    return state


def _headless_style_steps(state, mode, wind_block_size: int, n_cycles: int):
    """Replicates `optimizer.headless._advance_one_cycle`'s call pattern exactly."""
    from optimizer.headless import _advance_one_cycle
    from planet_params import EARTH

    for _ in range(n_cycles):
        state = _advance_one_cycle(
            state, mode, planet_params=EARTH, wind_block_size=wind_block_size,
        )
    return state


@pytest.mark.parametrize("mode_name", ["DAILY", "WEEKLY", "MONTHLY", "ANNUAL"])
def test_headless_matches_threaded_call_pattern(mode_name):
    """Headless and threaded call paths must produce identical state for identical params."""
    from simulate import create_initial_state, TimeScaleMode
    from optimizer.headless import _make_default_elevation

    mode = TimeScaleMode[mode_name]
    H, W = 24, 48
    wind_block_size = 4
    n_cycles = 3
    elevation = _make_default_elevation(H, W)

    state_headless = create_initial_state(elevation, day_of_year=80.0)
    state_headless = _headless_style_steps(state_headless, mode, wind_block_size, n_cycles)

    state_threaded = create_initial_state(elevation, day_of_year=80.0)
    state_threaded = _threaded_style_steps(state_threaded, mode, wind_block_size, n_cycles)

    assert state_headless.total_days == pytest.approx(state_threaded.total_days), (
        f"{mode_name}: total_days diverged: "
        f"headless={state_headless.total_days} threaded={state_threaded.total_days}"
    )
    np.testing.assert_allclose(
        state_headless.temperature, state_threaded.temperature, atol=1e-4, rtol=0,
        err_msg=f"{mode_name}: temperature diverged between headless and threaded call paths",
    )
    if state_headless.wind_u is not None and state_threaded.wind_u is not None:
        np.testing.assert_allclose(
            state_headless.wind_u, state_threaded.wind_u, atol=1e-4, rtol=0,
            err_msg=f"{mode_name}: wind_u diverged between headless and threaded call paths",
        )


def test_real_simulation_thread_produces_valid_state():
    """Smoke test: the actual `SimulationThread` class (real background thread,
    real Queue/Event synchronization) runs without error and produces a valid
    state — catches import/threading-setup issues the call-pattern comparison
    above can't (it never touches the real Thread machinery)."""
    import time
    from simulate import create_initial_state, TimeScaleMode
    from optimizer.headless import _make_default_elevation
    from main import SimulationThread

    H, W = 24, 48
    elevation = _make_default_elevation(H, W)
    state_init = create_initial_state(elevation, day_of_year=80.0)

    thread = SimulationThread(state_init, wind_block_size=4, time_scale_mode=TimeScaleMode.DAILY)
    thread.start()
    thread.resume()
    try:
        deadline = time.time() + 15.0
        while thread.state.total_days <= state_init.total_days and time.time() < deadline:
            time.sleep(0.02)
    finally:
        thread.pause()
        thread.stop()
        thread.join(timeout=5.0)

    assert thread.state.total_days > state_init.total_days, "SimulationThread never advanced"
    assert not np.any(np.isnan(thread.state.temperature)), "SimulationThread produced NaN"
