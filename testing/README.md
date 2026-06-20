## Testing Layout

This folder contains test runners, diagnostic helpers, generated test results, and test-focused reports.

### Contents

- `test_physics.py`: unit and integration sanity checks for core physics.
- `test_high_lat_diagnostic.py`: targeted high-latitude climate diagnostic runner.
- `test_ocean_temps.py`: ocean temperature comparison script.
- `high_latitude_diagnostics.py`: helper used by the high-latitude diagnostic workflow.
- `results/high_latitude/`: generated JSON outputs from high-latitude diagnostic runs.
- `reports/`: test and diagnostic writeups.

### Run Commands

```powershell
py testing/test_physics.py
py testing/test_physics.py --fast
py testing/test_high_lat_diagnostic.py
py testing/test_ocean_temps.py
```

### Rule Going Forward

Place new test-only scripts, generated test outputs, and diagnostic reports under `testing/` instead of the workspace root.
