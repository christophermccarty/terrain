"""Generate/regenerate the golden-state regression fixture.

Run this manually whenever a deliberate physics change alters simulate_step's
behavior:

    python scripts/generate_golden_state.py

This overwrites testing/fixtures/golden_state_reference.pkl, which
testing/test_golden_state.py compares every future run against. Do NOT run
this to "fix" a failing golden-state test without first confirming the state
change is an intentional physics change, not a regression -- that's exactly
the bug class this fixture exists to catch (see test_golden_state.py).

Uses the same deterministic entry point (optimizer.headless.run_simulation)
and config as the test, at a small/fast resolution so both stay quick.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer.headless import run_simulation
from planet_params import EARTH
from simulate import TimeScaleMode, save_state

# Kept in one place so the test module can import and reuse these exact values.
GOLDEN_STATE_CONFIG = dict(
    planet_params=EARTH,
    spinup_years=0.1,
    eval_years=0.05,
    H=32,
    W=64,
    spinup_time_scale=TimeScaleMode.MONTHLY,
    eval_time_scale=TimeScaleMode.DAILY,
)

FIXTURE_PATH = ROOT / "testing" / "fixtures" / "golden_state_reference.pkl"


def main() -> None:
    state, _ = run_simulation(**GOLDEN_STATE_CONFIG)
    save_state(state, FIXTURE_PATH)


if __name__ == "__main__":
    main()
