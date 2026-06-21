"""test_performance.py — Performance regression tests.

Measures simulation throughput and flags regressions.
Set PLANETSIM_PERF_THRESHOLD env var to override the default threshold.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Minimum acceptable throughput: sim-days per wall-clock second.
# Override with env var for CI environments with slower machines.
_DEFAULT_THRESHOLD = float(os.environ.get("PLANETSIM_PERF_THRESHOLD", "30"))


def _measure_throughput(state, n_steps: int = 50, **step_kwargs) -> float:
    """Return sim-days/sec for ``n_steps`` calls to simulate_step."""
    from simulate import simulate_step
    t0 = time.perf_counter()
    for _ in range(n_steps):
        state, _ = simulate_step(state, days=1.0, **step_kwargs)
    elapsed = time.perf_counter() - t0
    return n_steps / elapsed


# ---------------------------------------------------------------------------
# Throughput benchmarks
# ---------------------------------------------------------------------------

def test_throughput_coarse(flat_ocean_state):
    """block_size=4 should sustain ≥ THRESHOLD sim-days/sec."""
    throughput = _measure_throughput(
        flat_ocean_state, n_steps=50,
        block_size=4, wind_block_size=8,
        track_components=False, enable_carbon_cycle=False,
    )
    assert throughput >= _DEFAULT_THRESHOLD, (
        f"Performance regression: {throughput:.1f} sim-days/sec "
        f"(threshold: {_DEFAULT_THRESHOLD:.0f})"
    )


def test_track_components_overhead(flat_ocean_state):
    """track_components=True should add at most 3× overhead vs False."""
    from simulate import simulate_step

    state = flat_ocean_state
    kwargs = dict(block_size=4, wind_block_size=8, enable_carbon_cycle=False)

    t0 = time.perf_counter()
    for _ in range(20):
        state, _ = simulate_step(state, days=1.0, track_components=False, **kwargs)
    t_false = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):
        state, _ = simulate_step(state, days=1.0, track_components=True, **kwargs)
    t_true = time.perf_counter() - t0

    overhead = t_true / t_false
    assert overhead < 3.0, (
        f"track_components=True is {overhead:.1f}× slower than False (expected < 3×)"
    )


def test_carbon_cycle_overhead(flat_ocean_state):
    """Carbon cycle should add at most 5× overhead vs disabled."""
    from simulate import simulate_step

    state = flat_ocean_state
    kwargs = dict(block_size=4, wind_block_size=8, track_components=False)

    t0 = time.perf_counter()
    for _ in range(20):
        state, _ = simulate_step(state, days=1.0, enable_carbon_cycle=False, **kwargs)
    t_off = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):
        state, _ = simulate_step(state, days=1.0, enable_carbon_cycle=True, **kwargs)
    t_on = time.perf_counter() - t0

    overhead = t_on / t_off
    assert overhead < 5.0, (
        f"Carbon cycle is {overhead:.1f}× slower when enabled (expected < 5×)"
    )
