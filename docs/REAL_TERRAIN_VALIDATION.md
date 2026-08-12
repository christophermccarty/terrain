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

# Score against real climatology references (CRU TS v4.10 temperature/precip,
# NCEP/NCAR Reanalysis 1 wind) instead of only the built-in zonal-band targets
.\.venv\Scripts\python.exe scripts\run_real_terrain_validation.py `
  --monthly-climatology path\to\climatology.npz `
  --wind-climatology path\to\ncep_ncar_wind_1991_2020.npz
```

The default compact configuration is 64×128, one year of MONTHLY spinup, then
one year of time-averaged evaluation. It is intentionally small enough for a
repeatable regression test. Calibration decisions should also be checked with a
longer and/or higher-resolution run.

`--monthly-climatology` and `--wind-climatology` are optional and independent
of each other — see `docs/MONTHLY_CLIMATOLOGY_REFERENCE.md` for how to build
the reference files and what metrics each one populates.

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
- seasonal named-region precipitation-pathway diagnostics: final and raw
  rainout, the post-raw allocator adjustment, local evaporation, humidity,
  physical specific-humidity-flux convergence, lower wind, ascent/convection
  and storm-track drivers, plus named adjacent-ocean source temperature,
  humidity, cloud/inversion, evaporation, and source-to-land wind proxies for
  Atacama, East China, South Japan, and Central Europe, and the model's
  upwind SST anomaly even while SST coupling is default-off;
- seasonal lower- and upper-level jet core latitude, strength, migration span,
  and NH/SH asymmetry;
- a composite reference-error score.

`metrics.regional_moisture_budget` is a diagnostic decomposition, not a claim
that the default path closes a resolved horizontal column-water budget. It is
the duration-weighted mean of one-day, native-grid precipitation probes made at
each evaluation state; the probes do not alter simulation state. In particular,
`lower_wind_convergence_proxy` is the rainfall scheme's grid-scale wind
divergence signal, while `moisture_flux_convergence_driver` is its normalized
specific-humidity-flux driver. Neither should be reported as a physical
moisture-flux convergence in SI units. `post_raw_precip_adjustment_mm_day`
separates the calibrated row-target allocator from raw rain production, so a
regional improvement cannot be mistaken for new local moisture supply.

`metrics.seasonal_regional_moisture_budget` groups the same native-grid probes
into astronomical DJF/MAM/JJA/SON seasons relative to the configured vernal
equinox. It adds a true spherical divergence of the lower specific-humidity
flux (`physical_moisture_flux_convergence_q_s`) for diagnosis; this is a
snapshot tendency proxy, not a claimed prognostic column-water budget. The
four named ocean controls are context for the coastal/monsoon and Atlantic
pathways, not additional target regions. A positive
`source_to_region_lower_wind_m_s` means the regional mean lower wind points
from that ocean control toward the land box.

`metrics.seasonal_jet` samples each evaluation state and reports the
zonal-mean westerly core in both hemispheres. It is appropriate for diagnosing
the model's own upper-wind placement and seasonal movement. NCEP/NCAR's local
reference is near-surface wind speed only, so it remains a global speed-map
sanity check and must not be misused as an upper-tropospheric jet-latitude
target.

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

**2026-07-28 baseline refresh -- flagged, not fully explained.** The tracked
baseline was stale against a large pile of same-day uncommitted work (the
razor-sharp-biome-line precip-target smoothing, `wind_prognostic_substep_days`
defaulting on, coastal-fog suppression, and this session's own
`_moisture_budget_precip_rescale` per-cell-ceiling fix -- see
`test_climate_drift.py`'s module docstring for that one). Refreshing it was
in scope; explaining the score move was not, and it is a real regression worth
follow-up: composite reference-error score moved 0.384 -> 0.617, driven mostly
by desert regions getting substantially wetter (Sahara 386 -> 715 mm/yr,
Kalahari 68 -> 355 mm/yr, Atacama 177 -> 286 mm/yr), plus a milder continental
-interior wettening (Canadian Prairies 323 -> 550, US Midwest 566 -> 716,
Central Europe 582 -> 672 mm/yr). Isolated A/B (this session's own fix held
constant, everything else in the pile toggled): the fix itself *improves* the
score (0.851 without it vs. 0.617 with it, both against the same otherwise-
unchanged pile), so the desert-wetting driver is somewhere in the other,
already-shipped-earlier-today changes, not introduced by this session. Root-
causing which one is unexamined -- worth a dedicated pass before trusting
desert calibration again.

**2026-07-29 follow-up -- root-caused and fixed.** Isolated via `--param`
overrides on the same fixture: setting `wind_prognostic_substep_days=0.0`
(reverting only that flip) dropped the score from 0.617 to 0.455, while
disabling `coastal_upwelling_fog_strength` instead made it *worse* (0.692) --
confirming the wind-substep default flip was the actual driver, not the
target-smoothing or the fog gate. Direct field measurement (debug_fields probe,
annual-mean over the full evaluation year) found local desert `div` values
essentially unchanged between the diagnostic and prognostic wind pathways
(e.g. Kalahari ~0.229 both ways), but `atmosphere.py`'s early
`subsidence_suppression` block normalizes by the **global** mean of positive
divergence (`np.mean(_div_pos_early)`), and the prognostic solver's field has
a much heavier tail (p90 0.222->0.281, p99 0.243->0.429) that inflates that
global mean ~37% even though the bulk of the field (median, p75) barely moved.
Since a spatially-smoothing filter cannot change a field's global mean by
construction, the tropical-speckle-fix session's existing smoothing passes
were structurally unable to fix this -- it's a magnitude, not a spatial-noise,
problem. Fixed by capping each cell's contribution to that normalizer's
reference mean at a fixed 0.02 (real desert-local values run 0.16-0.28, well
above it, so they still saturate the ratio distinctly from near-zero
ocean/calm cells) -- see the code comment above `_div_pos_norm_ref` in
`atmosphere.py`. Recovered the baseline to 0.457 (Sahara 550, Kalahari 248,
Atacama 207 mm/yr) without touching `wind_prognostic_substep_days`'s default,
which stays load-bearing for the razor-sharp-biome-line fix. Full fast suite
(365 passed) plus `test_climate_drift.py` (8/8) green after the fix; both
fixtures regenerated. Desert boxes are still well above their informal
150-200 mm/yr targets in this baseline -- that's the long-standing, separate,
already-known desert-vs-continental gap (see known-physics-gaps.md item 3),
not this regression; lowering the 0.02 cap further closes more of that older
gap but increasingly costs already-under-target continental interior (US
Midwest), so it wasn't pushed past the point of recovering this session's own
regression.
