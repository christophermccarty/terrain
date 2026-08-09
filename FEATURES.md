# PlanetSim — Feature Backlog (Part 2)

> Created: 2026-07-25
> Companion to `ROADMAP.md` (long-horizon idea backlog) and `PLAN.md` (live task tracking).
>
> This file is specifically the **"what's missing that we should build"** list, as distinct
> from the **"what's broken that we should fix"** list tracked in the memory notes and
> summarized in ROADMAP.md. Work here begins **after** the Part 1 findings (open physics
> gaps, inert-gate decisions, test debt) are resolved.
>
> Ordered by recommendation strength. Effort estimates are rough and assume familiarity
> with the module being touched.

---

## Tier 1 — Recommended next

### 1. Surface hydrology: runoff routing, rivers, and lakes -- **Shipped 2026-07-27, default off**
**Effort: large. Touches: `hydrology.py`, `simulate.py`, `masks.py`.**

`PlanetParams.enable_surface_hydrology` (default `False`) adds a D8 flow-accumulation router
(`hydrology.py`) draining `soil_moisture`/`soil_moisture_deep` downslope into a per-cell
surface-water store, with discharge feeding `ocean.evolve_salinity` as real freshwater flux.
Three real bugs were found and fixed while calibrating it: salinity ignored runoff entirely,
the runoff trigger checked the chronically-floored surface bucket instead of the deep layer,
and standing water had no evaporative sink.

**Why it still defaults off**: the router has no channel-capacity or flow-velocity concept.
A continent-scale basin (Amazon/Congo-scale) funnels its full discharge into one grid cell
with no lateral spreading, producing area-averaged depths in the hundreds of meters (measured:
one cell hit 611m over a 10yr continuation, growing linearly, not asymptoting). A
`surface_water_cap_mm` (50m) hard ceiling was added as a safety valve, not a fix. See
`docs/ACCURACY_AUDIT.md` E1 for the full writeup.

**Remaining work**: actual channel hydraulics — a flow-accumulation-weighted capacity/velocity
term, so discharge exceeding a cell's physical channel capacity spills laterally instead of
pooling to unbounded depth. Rivers on the biome map remain the highest visual-payoff item
available once this is solved — lakes and wetlands as real evaporation sources would also
finally give the wetland-CH4 term (currently a toy constant) something physical to respond to.

### 2. Dynamic ice sheets with a real mass balance -- **Shipped 2026-07-28, default off**

`PlanetParams.enable_land_ice_dynamics` (default `False`, exact no-op) adds
`PlanetState.land_ice_thickness` (mass balance: gains `snow_depth`'s former 10 m-cap overflow,
loses to a degree-day ablation term) and a mass-conservative flux-form flow (thickness-weighted
diffusion, substepped for CFL stability -- see `simulate._land_ice_flow_step`), plus a derived
`PlanetState.sea_level_change_m` eustatic diagnostic. Real-terrain-tested (`saves/earth.pkl`,
10yr, seeded Antarctic/Greenland-scale reservoirs): numerically stable, spatially selective (zero
ice at desert/continental latitudes), sea-level diagnostic tracks ice volume correctly. Full
writeup, including a real CFL-stability bug found and fixed via the real-terrain check, in
`PLAN_PHYSICS_FIXES.md`'s 2026-07-28 entry.

**Deliberately not done this pass** (still real gaps, not oversights):
- **Flow doesn't follow terrain slope** -- diffuses thickness only, not ice-surface elevation,
  because `elevation` has no single canonical meters conversion in this codebase (the
  `max_elevation_km`-hardcoded-four-ways gap ROADMAP.md already tracks); adding a fifth
  conversion here would have compounded that gap rather than fixed it.
- **No albedo coupling** -- `land_ice_thickness` doesn't yet feed into surface albedo the way
  `ice_sheet_age`'s EF-threshold does.
- **No calving -> `evolve_salinity` coupling** -- ice lost at a coastline is discarded from the
  land reservoir, not credited as ocean freshwater input (the way hydrology's `runoff` is).
- **No mask/coastline feedback** -- `sea_level_change_m` is a real, tested, computed number, but
  it doesn't yet shift `masks.get_masks`'s land/sea split. That function has ~54 call sites
  across 22 files; threading an effective sea-level-adjusted elevation through all of them safely
  was judged too large for this session.
- **`ice_flow_diffusivity`'s default (2.0e-3) is uncalibrated** -- the real-terrain check showed
  a seeded 2000 m Antarctic-scale reservoir losing ~50% of its volume in 10 simulated years via
  flow-driven marginal loss, almost certainly too fast for a "stable ice sheet" target. Needs a
  dedicated multi-century calibration pass before the gate could default on.

All four are natural follow-ups, not blockers -- the core mass-balance/thickness/flow/sea-level
mechanism is real and tested. Once flow is calibrated, the **ice-age proof-of-concept scenario**
`PLAN_PHYSICS.md` has wanted since Effort 2E (pairs with item 3, the Milankovitch runner) and a
real **ice-albedo feedback with memory** both become reachable follow-ups.

### 3. Milankovitch scenario runner + an `experiments/` directory
**Effort: small-medium. Touches: new `experiments/`, `scripts/`.**

Obliquity and eccentricity already exist in `PlanetParams` and are already wired through the
insolation path. What is missing is the harness: a scripted sweep that drives them on orbital
timescales per the `PLAN_PHYSICS.md` ice-age recipe (low obliquity + low CO2 start) and
records the trajectory.

This is mostly wiring rather than new physics, and it would be the project's strongest single
validation showcase — "the model spontaneously glaciates under the right orbital forcing" is a
far more convincing result than any individual zonal-mean bound. Note that it is only
*meaningful* once item 2 exists; without dynamic ice sheets there is nothing for the orbital
forcing to act on except sea ice.

The `experiments/` directory referenced in PLAN.md's deferred list still does not exist.

### 4. A real reanalysis benchmark pack (ERA5/CRU at ~2°) with map-correlation scoring -- **Mostly done; pytest gate landed 2026-08-09**

The infrastructure this item called for already existed by the time it was picked back up: a real
CRU TS v4.10 0.5deg monthly land climatology (`testing/reference_data/cru_ts_v4.10_1991_2020.npz`,
built via `scripts/build_cru_ts_reference.py`), area-conservative regridding and weighted
bias/RMSE/correlation scoring (`monthly_climatology.py`), wired into
`scripts/run_real_terrain_validation.py --monthly-climatology`, with baselines already recorded in
`docs/MONTHLY_CLIMATOLOGY_REFERENCE.md` at both 64x128/1yr and 128x256/5yr. This item's own text
was stale -- it had not been updated after that work landed.

The one real gap was that none of it was an enforced regression gate: `testing/
test_reanalysis_validation.py` still only asserted against six hand-typed zonal bands on synthetic
terrain, and its own docstring still said map-correlation was "deferred." Fixed 2026-08-09: three
new tests (`test_cru_temperature_map_correlation_and_error`,
`test_cru_precipitation_map_correlation_and_error`, `test_cru_scored_area_fraction_is_meaningful`)
run the real bundled-Earth-DEM harness scored against the real CRU reference at 64x128/1yr+1yr and
assert bias/RMSE/correlation stay within headroom of the measured baseline (temp +2.98C bias/6.28C
RMSE/0.930 correlation; precip -0.109 mm/day bias/1.406 log-RMSE/0.463 log-correlation). They
`skipif` cleanly when the gitignored local `.npz` hasn't been built, per this file's own licensing
note. The old zonal-band tests were kept (cheap, terrain-independent general sanity checks) rather
than replaced.

**Optimizer integration -- done 2026-08-09.** `ClimateMetrics` gained four fields
(`cru_temp_correlation`, `cru_temp_rmse`, `cru_precip_log_correlation`, `cru_precip_log_rmse`) and
`ReferenceClimate` matching `(lo, hi, weight)` targets, weight defaulting to `0.0` (exact no-op for
every existing sweep, same convention as this project's other real-but-inert mechanisms).
`optimizer/headless.run_simulation` gained an optional `monthly_climatology_path` argument that
scores the run's own native monthly Köppen bins (`state.monthly_temp`/`monthly_precip`, already
maintained every step regardless) against the reference via `monthly_climatology.
score_monthly_climatology`, reusing `state`'s existing fields rather than adding new accumulation
logic. Meaningless on `run_simulation`'s default synthetic terrain -- map correlation needs real
geography -- so it must be paired with a real elevation (e.g.
`real_terrain_validation.load_bundled_earth_dem`). 4 new tests in `testing/test_optimizer_scoring.py`
cover the no-op default, opt-in weight actually moving the score, real-terrain population, and that
omitting the path leaves the fields untouched. `diagnostics.py` itself was never touched -- the
scoring logic lives entirely in `monthly_climatology.py`, which turned out to be a perfectly
reasonable home for it and not worth moving.

**What's still open, and now genuinely blocked, not just undone**: wind cannot be scored against
CRU TS -- verified 2026-08-09 by listing the actual CRU TS v4.10 server directory
(`crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.10/.../`): it publishes only `cld, dtr, frs, pet, pre,
tmn, tmp, tmx, vap, wet` -- no wind variable exists in this dataset at all, in any version
currently served. Wind reanalysis would need a different provider (e.g. ERA5 via the Copernicus
CDS API, or NCEP/NCAR Reanalysis 1 via NOAA PSL, which is anonymously downloadable but a coarser
~2.5deg product and a different measurement basis than CRU). Needs a real provider decision (and,
for ERA5, user-supplied CDS API credentials) before any implementation attempt -- do not assume
CRU's existing pattern extends to wind.

### 5. Prognostic AMOC + freshwater hosing
**Effort: medium. Touches: `simulate.py`, `PlanetParams`.**

AMOC is now partially prognostic: `amoc_factor` responds to both NH sea-ice cover and a real
North Atlantic salinity anomaly (`PlanetParams.salinity_amoc_scale`, a +1 PSU anomaly multiplies
it by 1.15) — see `simulate.py:595-631`. What's still missing is a *temperature* density
contribution (salinity-only today) and deriving the base `amoc_bonus_near`/`amoc_bonus_far`
magnitudes from an actual overturning-strength calculation rather than prescribing them as
constants; see `docs/ACCURACY_AUDIT.md` D2. Salinity itself is a real prognostic field, stable
at ~35.3 PSU since the Jul 2026 coefficient fix. Finishing the temperature-density coupling
would:
- Unlock hosing experiments, AMOC bistability, and Younger-Dryas-style collapse scenarios.
- Make the ocean module less of "the weakest module relative to its climate influence."
- Compose with item 1 (river discharge is the natural freshwater forcing) and item 2
  (meltwater pulses).

### 6. Latitude-dependent mixed-layer depth
**Effort: small. Touches: `ocean.py`, `simulate.py`, `PlanetParams`.**

A single effective ocean heat capacity everywhere misses the shallow-tropics /
deep-subpolar contrast that sets seasonal SST lag. A latitude-dependent (later:
map-based) mixed-layer depth is cheap and lets several hand-tuned ocean seasonal-lag
fractions be *derived* rather than prescribed. It also targets the seasonal-amplitude
discrepancies behind the MONTHLY-vs-DAILY divergence in Part 1.

---

## Tier 2 — Strong, but after the above

### 7. Planet generalization: tidally-locked regime, random planets, habitability report
**Effort: medium-large, splits cleanly into three.**

Goal #4 of the project ("generalizability to non-Earth bodies") has had almost no work: there
is a `MARS` singleton and that is it.

- **Tidally-locked regime** — substellar-point insolation instead of diurnal-mean-by-latitude.
  The temperature LUT machinery reportedly already mostly supports this (day length → ∞). Big
  payoff, since most known rocky planets in habitable zones are likely locked.
- **`PlanetParams.random(seed, archetype=...)`** — archetypes: temperate-terran, cold-desert,
  hothouse, waterworld, high-obliquity. Pair with procedural terrain driven by
  plate-tectonic-flavoured noise.
- **Habitability report** — a summary generated from the existing diagnostics. Cheap once the
  first two exist, and it is what makes random planets *interesting* rather than just random.

Do this after Tier 1 because random planets exercise every physics path at once; it is much
more valuable once the Earth case is not still fighting a 10x precip deficit.

### 8. Scenario / event system in the GUI
**Effort: medium. Touches: `main.py`, `simulate.py`, `PlanetParams`.**

Right now the simulation runs and you watch it. There is no way to *do* anything to the planet.
This is the largest gap versus the SimEarth inspiration, and it is comparatively cheap because
so much is already parameterized:
- **Volcanic eruption** — the aerosol/volcanic forcing field already exists in `PlanetParams`.
- **CO2 forcing ramp / solar-constant change** — both already parameters.
- **Asteroid impact** — dust loading + transient forcing, reusing the aerosol path.
- **Terrain editing** — `masks.invalidate()` already exists to handle in-place elevation edits,
  so the plumbing is half there.

### 9. More view layers
**Effort: small each. Touches: `main.py`.**

Ten views exist (a "Surface Water" view was added alongside the hydrology feature above).
Several of the most heavily-debugged fields in the entire project still have no way to be
looked at: **soil moisture, humidity, salinity, sea-ice thickness, snow depth,
subsidence suppression, and the pre-rescale precipitation potential**. Given how many sessions
were spent reasoning about these fields through box-averaged CLI output, a map view is
disproportionately useful.

Also recommended: a **zonal-mean cross-section panel** (latitude on one axis, the field's
zonal mean on the other, with an Earth reference overlaid). Several of the retracted findings
in the project history would have been caught immediately by one.

### 10. Diurnal cycle (DAILY mode only)
**Effort: small. Touches: `temperature.py`, `simulate.py`.**

A cheap sinusoidal `T_air` modulation over land, active only at DAILY (skipped at
weekly-and-coarser modes). Improves convective precipitation timing and continental climate
realism. On the roadmap since the beginning, never done.

---

## Tier 3 — Worth having, low urgency

- **Ocean carbonate chemistry** — a 2-box (surface/deep) split with a solubility pump, giving
  realistic century-scale CO2 drawdown and making ECS experiments more meaningful.
- **Recalibrate the toy carbon fluxes** — permafrost ppm-per-kgC and wetland ppb conversions
  are deliberately orders of magnitude below physical values; now that CH4 has a
  baseline-balancing natural source, they could be raised with the optimizer verifying
  stability.
- **Optimizer: multiprocess sweeps** — `headless.py` runs are embarrassingly parallel and the
  machine has 32 cores; a process-pool runner would make Bayesian search practical overnight.
- **Incremental render** — the GUI redraws the full RGB overlay every frame; dirty-region
  updates would cut UI cost at 512×1024.
- **GUI smoke automation** — headless Tk instantiation test. `main.py` is ~116 KB of code with
  no test coverage at all.
- **Adaptive substep count in `evolve_wind`** — 8 substeps regardless of dt or wind speed; a
  CFL-based count (2-8) would speed up calm periods.

---

## Explicitly not recommended

These remain correctly out of scope, and none of them is what is currently blocking realism:

- Full 3D atmosphere/ocean (GCM territory — wrong tool for an interactive simulator).
- Sub-daily weather realism (individual thunderstorms, fronts).
- Anthropogenic / land-use forcing scenarios. (Note: if "present-day Earth" saves should hold
  near-modern CH4 rather than decaying to the pre-industrial baseline, that is a small,
  separate, deliberately-declined change — see the CH4 scope decision of 2026-07-12.)
- Multiplayer / scripting APIs before the physics core stabilizes.
