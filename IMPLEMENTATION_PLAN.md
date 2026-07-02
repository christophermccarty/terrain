# PlanetSim — Benchmark Recovery Plan (archived)

> **STATUS (2026-07-01): SUPERSEDED.** This plan targeted a specific benchmark regression
> (weak circulation, SH/NH sea-ice imbalance, wet-subtropics precipitation bias, broad thermal
> warm bias). All three gate tests it called for now exist and pass:
> `testing/test_circulation_strength.py`, `testing/test_latitude_band_regression.py`, and
> `testing/test_polar_balance.py`. Mask convergence is done (`_ocean_mask_from_elevation` in
> ocean.py is unused dead code now — everything routes through `masks.get_masks`). PlanetParams
> wiring is done (see PLAN.md Phase 3). Kept for historical context on the reasoning behind those
> tests; not an open task list. See PLAN.md for current status.

Keep As Partial / Incomplete
These remain active roadmap items because they are only partly done:

Finish convergence on one mask system.
Finish PlanetParams end-to-end.
Centralize timestep policy.
Improve thermal baseline realism.
Improve circulation strength and structure.
Fix sea ice hemispheric balance.
Add missing prognostic state for snow_depth, and decide what to do with air_temperature and cloud_water.
Align carbon dynamics with stable biome/climate state instead of instantaneous weather.
Benchmark-Driven Priorities
Based on the latest benchmark, the next systems to investigate are:

Thermal baseline and SH asymmetry The model is globally warm, very warm in the subtropics, and still too cold at the far south pole. That points to broad imposed warming plus an Antarctic-specific cold branch still surviving underneath it.

Circulation forcing and wind resolution Trades and mid-lat winds are present in sign but too weak in magnitude. Hadley flow, ITCZ convergence, and jet latitude are failing. This is now a first-class blocker because many temperature and precipitation biases follow from it.

Sea ice logic and hemispheric balance NH ice is too large, SH ice is effectively absent, yet the SH pole is still too cold. That means the cryosphere and ocean branches are not behaving consistently.

Latitude-band precipitation structure The subtropics are too wet, especially in the south. That is consistent with weak subsidence belts and weak circulation organization.

Planet portability testing/test_planet_params.py still has xfail coverage for Mars-like and high-obliquity behavior, so this remains unfinished.

Revised Master Plan
Phase 1: Finish Structural Plumbing
Goal: close the “partial” items before retuning physics.

Replace remaining uses of _ocean_mask_from_elevation() and _derive_land_sea_masks() with masks.get_masks().
Audit main.py, graphs.py, terrain.py, ocean.py, and atmosphere.py for old mask paths.
Finish PlanetParams wiring through any remaining Earth-hardcoded paths, especially the base climatology and diagnostic wind generation.
Decide whether air_temperature, cloud_water, and snow_depth are real planned state or dead fields. If real, implement; if not, remove later.
Phase 2: Fix The Thermal Baseline
Goal: reduce imposed warm bias before touching more advanced physics.

Audit broad warming terms in simulate.py:
transport warming
hemispheric asymmetry terms
subtropical subsidence heating
land and ocean caps
Separate “Earth tuning” from “core solver physics”.
Re-benchmark after each reduction to avoid overcorrecting.
Primary target from benchmark:

lower tropical/subtropical warmth
reduce SH warm mid-lats without making Antarctica even colder
Phase 3: Fix Circulation Strength And Placement
Goal: get a stronger, better-shaped 3-cell surface circulation.

Investigate whether the benchmark fixture is under-resolving wind with wind_block_size=8 on a 64x128 grid.
Compare circulation diagnostics across wind_block_size=8, 4, 2, and 1.
Review forcing strength in:
simulate.py wind-relax settings
atmosphere.py diagnostic wind target generation
pressure-gradient scaling
tropical meridional flow terms
Prioritize:
stronger trades
stronger mid-lat westerlies
positive ITCZ convergence
Hadley surface return flow in the right direction
jets no longer pinned at the poles
Phase 4: Fix Cryosphere And Southern Hemisphere Balance
Goal: make ice, ocean, and polar land behavior internally consistent.

Recheck sea-ice threshold behavior and rate ordering in simulate.py and ocean.py.
Compare NH and SH ocean temperature bands against sea-ice growth/melt behavior.
Add explicit benchmark checks for:
NH vs SH ice fraction ratio
NH and SH ice edges
Southern Ocean temperatures
Antarctic land vs adjacent ocean contrast
Only after this, revisit sea-ice albedo strength and transport tuning.
Phase 5: Fix Latitude-Band Precipitation Structure
Goal: correct wet subtropics and weak dry belts.

Use circulation fixes first, then retune precipitation.
Investigate whether weak subsidence belts and weak meridional flow are causing subtropical rainfall excess.
Review:
moisture convergence
convective trigger strength
humidity source terms
soil moisture persistence
any broad rain smoothing
Phase 6: Finish Planet Generalization
Goal: make “other planets” a real supported mode, not partial support.

Remove xfail from testing/test_planet_params.py by making Mars-like and high-obliquity tests pass.
Ensure temperature.py base climatology is truly driven by PlanetParams.
Ensure generate_wind_field() and any remaining annual-cycle logic use PlanetParams.
Phase 7: Deferred Features After Benchmark Recovery
These should stay on the roadmap, but only after Phases 1-6 are stable:

prognostic land snowpack
soil carbon pool
save/load UI plumbing
planet preset/editor UI
event/intervention layer
optional diurnal cycle
deeper ocean improvements
Test Plan For The Next Revision Cycle
Promote Existing Diagnostics Into Hard Gates
Use diagnostics.py and testing/high_latitude_diagnostics.py directly in pytest instead of relying on manual reading.

Add to testing/test_earth_benchmark.py:

both-hemisphere mid-lat westerly strength thresholds, not just sign
both-hemisphere trade strength thresholds, not just sign
Hadley return-flow sign checks from ClimateDiagnostics.analyze_circulation()
positive ITCZ convergence check
jet latitude bounds using analyze_circulation()
SH asymmetry check using T_pole_north, T_pole_south, gradient_north, and gradient_south
sea-ice asymmetry and ice-edge checks
latitude-band precipitation bias checks focused on 25°–65° in both hemispheres
Keep Existing Fast Gates
Do not replace these:

testing/test_unit_physics.py
testing/test_stability.py
testing/test_conservation.py
testing/test_performance.py
Add New Focused Tests
test_circulation_strength.py Gate on wind magnitude and circulation structure using ClimateDiagnostics.analyze_snapshot() and analyze_circulation().

test_latitude_band_regression.py Assert bounded temperature and precipitation bias by latitude band using compute_latitude_band_stats().

test_polar_balance.py Use analyze_high_latitude_temperatures() to assert no extreme NH/SH imbalance flags.

test_wind_resolution_sensitivity.py Compare benchmark metrics across wind_block_size values so future tuning is not done blindly.

test_planet_portability.py Keep the current file, but make removal of xfail an explicit milestone.

Suggested Execution Order
Finish mask convergence.
Finish PlanetParams plumbing.
Add hard benchmark tests for circulation, SH asymmetry, and sea ice.
Investigate wind resolution and wind forcing.
Retune thermal baseline.
Retune cryosphere.
Retune precipitation structure.
Remove xfail from planet portability tests.
Only then resume feature additions.
Practical Definition Of “Done” For This Pass
I would consider this revision pass successful when all of these are true:

global mean stays near the current acceptable range without broad warm bias
SH equator-to-pole gradient is back inside benchmark bounds
trades and mid-lat westerlies pass both sign and strength checks
Hadley flow and ITCZ convergence pass
jets are in mid-lats, not at the poles
NH/SH sea ice are both present and not wildly imbalanced
test_planet_params.py no longer needs xfail for basic portability cases