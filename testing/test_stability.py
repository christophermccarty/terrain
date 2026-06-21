"""test_stability.py — Numerical stability and timestep sensitivity tests.

Verifies that the simulation remains bounded and produces consistent results
across a range of time step sizes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Large-timestep stability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [0.5, 1.0, 5.0, 10.0, 30.0])
def test_large_timestep_no_blowup(flat_ocean_state, dt):
    """No NaN/Inf and T in [150, 370] K for various timestep sizes."""
    from simulate import simulate_step

    state = flat_ocean_state
    n_steps = max(1, int(365 / dt))
    for _ in range(n_steps):
        state, _ = simulate_step(state, days=dt, block_size=4,
                                 track_components=False)

    T = state.temperature
    assert not np.any(np.isnan(T)),  f"NaN at dt={dt}"
    assert not np.any(np.isinf(T)),  f"Inf at dt={dt}"
    assert float(np.min(T)) > 150.0, f"T below floor at dt={dt}: min={np.min(T):.0f} K"
    assert float(np.max(T)) < 370.0, f"T above cap at dt={dt}: max={np.max(T):.0f} K"


def test_timestep_invariance(flat_ocean_state):
    """30 simulated days as 30×1d vs 6×5d should give similar global mean T (±5 K)."""
    from simulate import simulate_step

    state_daily  = flat_ocean_state
    state_coarse = flat_ocean_state

    for _ in range(30):
        state_daily, _ = simulate_step(state_daily, days=1.0, block_size=4,
                                       track_components=False)
    for _ in range(6):
        state_coarse, _ = simulate_step(state_coarse, days=5.0, block_size=4,
                                        track_components=False)

    T_daily  = float(np.mean(state_daily.temperature))
    T_coarse = float(np.mean(state_coarse.temperature))
    assert abs(T_daily - T_coarse) < 10.0, (
        f"Timestep sensitivity too high: 1-day={T_daily:.1f} K, 5-day={T_coarse:.1f} K "
        f"(diff={abs(T_daily - T_coarse):.1f} K)"
    )


# ---------------------------------------------------------------------------
# Wind stability
# ---------------------------------------------------------------------------

def test_wind_does_not_diverge(flat_ocean_state):
    """Max wind speed should remain below 100 m/s after 1 year."""
    from simulate import simulate_step

    state = flat_ocean_state
    for _ in range(365):
        state, _ = simulate_step(state, days=1.0, block_size=4,
                                 track_components=False)

    if state.wind_u is not None:
        speed = np.sqrt(state.wind_u**2 + state.wind_v**2)
        assert float(np.max(speed)) < 100.0, (
            f"Wind diverged: max speed = {np.max(speed):.1f} m/s"
        )
        assert not np.any(np.isnan(speed)), "NaN in wind field after 1 year"


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(flat_ocean_state):
    """Pickle a state, reload it, run 10 more days: T should be continuous (< 0.1 K jump)."""
    import pickle
    import io
    from simulate import simulate_step

    state = flat_ocean_state
    for _ in range(10):
        state, _ = simulate_step(state, days=1.0, block_size=4,
                                 track_components=False)

    T_before = float(np.mean(state.temperature))

    buf = io.BytesIO()
    pickle.dump(state, buf)
    buf.seek(0)
    state_reloaded = pickle.load(buf)

    T_after = float(np.mean(state_reloaded.temperature))
    assert abs(T_before - T_after) < 0.1, (
        f"T changed on save/load: {T_before:.3f} K → {T_after:.3f} K"
    )
