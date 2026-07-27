# Real-terrain validation

`real_terrain_validation.py` is the canonical deterministic validation path for
Earth physics changes. It starts from the tracked 16-bit Earth DEM, runs the
same centralized time-scale policy as the GUI and headless optimizer, averages
the evaluation window, and emits JSON-safe global, zonal, and named-region
metrics.

## Routine use

```powershell
# Run the compact benchmark and compare it with the tracked regression baseline
.\.venv\Scripts\python.exe scripts\run_real_terrain_validation.py --compare

# Save a report for an experiment
.\.venv\Scripts\python.exe scripts\run_real_terrain_validation.py `
  --param spherical_metric_precip=1 `
  --output testing\results\spherical-precip.json

# Higher-resolution or longer measurement
.\.venv\Scripts\python.exe scripts\run_real_terrain_validation.py `
  --height 128 --width 256 `
  --spinup-years 10 --evaluation-years 3 `
  --output testing\results\long-real-terrain.json
```

The default compact configuration is 64×128, one year of MONTHLY spinup, then
one year of time-averaged evaluation. It is intentionally small enough for a
repeatable regression test. Calibration decisions should also be checked with a
longer and/or higher-resolution run.

## What is measured

- area-weighted global temperature, precipitation, cloud cover, and soil water;
- land and Northern Hemisphere mid-latitude soil-floor saturation;
- precipitation and soil moisture in the Sahara, Kalahari, Atacama, Canadian
  Prairies, US Midwest, and Central Europe;
- continental-minus-desert precipitation separation;
- temperature bias and precipitation ratio in six ERA5/CRU-inspired zonal
  reference bands;
- Köppen land fractions;
- precipitation rescale mean, maximum, and saturation fraction;
- a composite reference-error score.

## Baseline versus realism

`testing/fixtures/real_terrain_validation_baseline.json` is a regression
baseline, not a declaration that the current climate is realistic. The
comparison gate detects material drift from a reviewed state. The reference
targets and `reference_error_score` independently show whether a change moves
toward observations.

The initial baseline exposes the known structural issues clearly:

- 25% of latitude rows hit the precipitation rescale ceiling;
- Atacama and Sahara are too wet;
- the US Midwest is too dry;
- tropical precipitation is too strong;
- mean cloud fraction is too low.

This is why the next physics work should address the precipitation
production/rescale mechanism rather than merely retuning regional constants.

## Updating the baseline

Only update the baseline for an intentional, explained climate change:

```powershell
.\.venv\Scripts\python.exe scripts\run_real_terrain_validation.py `
  --write-baseline testing\fixtures\real_terrain_validation_baseline.json
```

Before accepting it, compare the old and new reference-error scores and inspect
every regional and zonal metric. A green unit suite alone is not sufficient.
