## Testing Layout

This folder is a standard pytest suite (`pytest.ini` + `conftest.py` fixtures at the repo
root/`testing/`), not a collection of standalone scripts. As of 2026-07, it holds 29
`test_*.py` files (~230 collected tests).

### Run Commands

```powershell
python -m pytest testing/                 # full suite (includes @slow tests; can take ~10+ min)
python -m pytest testing/ -m "not slow"    # fast subset — use this for routine iteration
python -m pytest testing/test_foo.py -q    # a single file
```

Long-running benchmark/integration tests are marked `@pytest.mark.slow` (see `pytest.ini`) and
excluded by `-m "not slow"`.

### Notable non-pytest files

A few files under `testing/` are named `test_*.py` for historical reasons but are **standalone
diagnostic scripts**, not pytest suites — pytest collects zero tests from them:

- `test_high_lat_diagnostic.py` — run directly (`python testing/test_high_lat_diagnostic.py`) to
  print/export a high-latitude temperature diagnostic using `high_latitude_diagnostics.py`
- `test_ocean_temps.py` — run directly to compare simulated SST against real-world data across
  the year

`test_physics.py`, despite the similar name, *is* a real pytest file (6 tests).

### Supporting files/directories

- `conftest.py` — shared fixtures (e.g. `mixed_initial_state`)
- `high_latitude_diagnostics.py` — helper used by `test_high_lat_diagnostic.py` and by
  `test_circulation_strength.py`/`test_earth_benchmark.py`/`test_polar_balance.py` for their
  gate assertions
- `results/high_latitude/` — generated JSON outputs from diagnostic runs
- `reports/` — test and diagnostic writeups

### Rule Going Forward

Place new test-only scripts, generated test outputs, and diagnostic reports under `testing/`
instead of the workspace root. New assertion-based tests should be real pytest files
(`def test_...`), not standalone scripts, so they run as part of `python -m pytest testing/`.
