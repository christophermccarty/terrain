## Testing Layout

This folder is a standard pytest suite (`pytest.ini` + `conftest.py` fixtures at the repo
root/`testing/`), not a collection of standalone scripts. As of 2026-08-12, it holds
93 `test_*.py` modules and collects 802 tests (630 routine, 172 slow). Use pytest collection as the
authoritative count because the suite changes frequently:

```powershell
.\.venv\Scripts\python.exe -m pytest --collect-only -q
```

### Run Commands

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest testing/                 # full suite (includes @slow tests; can take ~10+ min)
.\.venv\Scripts\python.exe -m pytest testing/ -m "not slow"    # fast subset — use this for routine iteration
.\.venv\Scripts\python.exe -m pytest testing/test_foo.py -q    # a single file
.\.venv\Scripts\python.exe scripts\run_real_terrain_validation.py --compare
```

Long-running benchmark/integration tests are marked `@pytest.mark.slow` (see `pytest.ini`) and
excluded by `-m "not slow"`.

Pytest's `--basetemp` is set to `testing/.pytest-tmp/` in `pytest.ini`. This
avoids user-profile temp-directory permissions on Windows and keeps temporary
test files inside the ignored workspace location. Do not run concurrent pytest
processes using the default configuration: they would share that temporary
directory. Supply a distinct `--basetemp` for each parallel process.

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
- `fixtures/real_terrain_validation_baseline.json` — compact real-DEM regression baseline;
  it guards reviewed behavior while the separate reference score measures realism

### Rule Going Forward

Place new test-only scripts, generated test outputs, and diagnostic reports under `testing/`
instead of the workspace root. New assertion-based tests should be real pytest files
(`def test_...`), not standalone scripts, so they run as part of `python -m pytest testing/`.
