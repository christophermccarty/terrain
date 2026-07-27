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
- area-weighted precipitation poleward of 70°;
- Köppen land fractions;
- precipitation strategy, effective correction, moisture-capacity limitation,
  unmet target, and legacy-ceiling saturation;
- a composite reference-error score.

## Baseline versus realism

`testing/fixtures/real_terrain_validation_baseline.json` is a regression
baseline, not a declaration that the current climate is realistic. The
comparison gate detects material drift from a reviewed state. The reference
targets and `reference_error_score` independently show whether a change moves
toward observations.

The production precipitation correction is moisture-budget bounded. Unlike the
legacy row multiplier, it:

- scales excess precipitation down but treats deficient zonal targets as
  aspirational;
- allocates added condensation preferentially to existing rain systems;
- protects dynamically subsiding dry-belt land from artificial target filling;
- caps added rainout at 15% of local specific humidity per call and total
  rainout at 85%;
- reports capacity-limited rows and unmet target precipitation explicitly.

On the compact two-year real-terrain gate, this reduced the composite reference
error from 0.438 to 0.376. On a five-year spinup plus three-year evaluation it
reduced error from 0.419 to 0.349. Tropical zonal rain moved from roughly
3,700-3,800 mm/year to 1,969 mm/year, cloud fraction increased, and US Midwest
rain improved. Sahara remains too wet, so the new mechanism removes the old
multiplier-ceiling failure but does not claim to complete regional calibration.
Set `moisture_budget_precip_rescale=False` only for legacy comparison runs.

Precipitation convergence and cloud ascent/subsidence now use spherical flux
divergence by default. Their normalization is scale-invariant, so the physical
per-second divergence is not suppressed by a dimensionless epsilon. The
compact and 5+3-year gates showed only a 2-3% reference-score cost, accepted in
exchange for correct latitude metrics, nonzero polar rows, and globally
conservative divergence. Set `spherical_metric_precip=False` or
`spherical_metric_clouds=False` only for legacy A/B runs.

## Updating the baseline

Only update the baseline for an intentional, explained climate change:

```powershell
.\.venv\Scripts\python.exe scripts\run_real_terrain_validation.py `
  --write-baseline testing\fixtures\real_terrain_validation_baseline.json
```

Before accepting it, compare the old and new reference-error scores and inspect
every regional and zonal metric. A green unit suite alone is not sufficient.
