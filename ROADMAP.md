# PlanetSim — Roadmap

> Created: 2026-07-03
> Companion to PLAN.md (live task tracking). This document is the long-horizon
> idea backlog: where the simulation can go next, roughly ordered by
> leverage within each theme. Nothing here is committed work.

---

## Guiding goals

1. **Earth first, planets second, random worlds third.** Every physics change
   should either improve Earth realism or remove an Earth-only assumption —
   ideally both.
2. **Performance and realism trade off explicitly.** Each tier of
   TimeScaleMode (daily → annual) is allowed to be more approximate, but its
   approximations should be *chosen*, not accidental.
3. **Everything tunable is optimizable.** New parameters belong in
   `PlanetParams` or `simulate_step` kwargs so the optimizer backend can sweep
   them.

---

## Theme 1 — Atmosphere depth (biggest realism lever)

The single-layer atmosphere is now well-tuned but fundamentally limits what
can emerge on its own (jets, monsoons, real storm dynamics).

- **1.5-layer atmosphere.** Add a single "upper" wind layer (or a prescribed
  baroclinic shear profile) so thermal wind, jet streams, and storm steering
  emerge from the temperature field instead of the 3-cell relaxation targets.
  This is the highest-value structural upgrade; most of the current
  parameterizations (cell relaxation, baroclinic jet mixing, Rossby modes)
  could then be weakened or retired.
- **Prognostic cloud water.** `cloud_water` exists in state but is never
  updated; clouds are re-diagnosed from RH each step. A simple
  condensation/precipitation/evaporation budget would give clouds memory and
  make the cloud-radiation feedback less twitchy.
- **Spherical metric completeness in precipitation.** The meridional-sign
  fixes (2026-07-03) made convergence/divergence directionally correct, but
  the zonal terms still lack the 1/cos(lat) metric factor and the poles are
  left at zero in the Numba kernels. Low urgency at Earth-like obliquity;
  matters more for polar-precipitation-dominated worlds.
- **CFL-linked humidity advection.** Humidity advection scales (`u_scale`,
  `v_scale`) are tuned constants; link them to actual wind CFL numbers so
  moisture transport speeds up/slows down consistently with the wind field.
- **Diurnal cycle (optional, DAILY mode only).** The docstring mentions one
  but none exists. A cheap sinusoidal T_air modulation over land would improve
  convective precipitation timing and continental climates; skip at
  weekly-and-coarser modes.

## Theme 2 — Ocean upgrade

The 1D zonal-mean transport plus parameterized AMOC/ACC is the weakest module
relative to its climate influence.

- **2D barotropic gyres.** Replace the zonal-mean transport with a
  streamfunction solve on the coarse grid (the FFT Poisson machinery already
  exists in atmosphere.py). Western boundary currents, subpolar gyres, and
  basin-shape sensitivity would emerge from topology instead of the
  land-west-of-ocean heuristic added 2026-07-03.
- **Prognostic AMOC strength.** AMOC is currently a scale factor; make it
  respond to high-latitude salinity/temperature (the salinity field already
  exists), enabling freshwater-hosing experiments and ice-age dynamics.
- **Mixed-layer depth map.** A single effective heat capacity everywhere
  misses the shallow-tropics/deep-subpolar contrast that shapes seasonal SST
  lag; a latitude-dependent mixed-layer depth is cheap and would let the
  hand-tuned ocean seasonal-lag fractions be derived instead of prescribed.
- **Ocean CO2 uptake with proper piston velocity.** Currently uses
  instantaneous wind² which double-counts storm variance at daily steps; use
  the monthly-mean wind speed (already effectively cached with the 30-day
  ocean update).

## Theme 3 — Planet generalization (toward random planets)

- **Planet parameter audit test.** A pytest that greps physics modules for
  numeric literals matching Earth constants (6371, 365, 1013.25, 288.15…)
  outside PlanetParams/EARTH definitions, with an allowlist. This is how the
  hardcoded storm radius and `% 365` cache bugs (both fixed 2026-07-03) would
  have been caught automatically.
- **Tidally-locked regime.** Substellar-point insolation instead of
  diurnal-mean-by-latitude; the temperature LUT machinery mostly supports
  this (day length → ∞). Big payoff for exoplanets (most known rocky planets
  in habitable zones are likely locked).
- **Non-water condensables.** Generalize latent heat/precipitation constants
  (already partially in PlanetParams) so CO2 (Mars) or CH4 (Titan) cycles are
  parameter choices, not new code paths.
- **Random planet generator.** A `PlanetParams.random(seed, archetype=...)`
  factory (archetypes: temperate-terran, cold-desert, hothouse, waterworld,
  high-obliquity) plus procedural terrain hooked to plate-tectonic-flavoured
  noise. Pair with a "habitability report" from the existing diagnostics.
- **GUI overlay generalization.** `generate_temperature_overlay` and several
  display ramps ignore PlanetParams; color scales should derive from the
  planet's actual temperature range.

## Theme 4 — Carbon cycle & long-run climate

- **Recalibrate the toy flux constants.** Permafrost ppm-per-kgC and wetland
  ppb conversions are orders of magnitude below physical values (deliberately
  conservative). Now that CH4 has a baseline-balancing natural source
  (2026-07-03), the perturbation fluxes could be raised toward realistic
  magnitudes with the optimizer verifying stability.
- **CLIMATE_SENSITIVITY constant cleanup.** carbon_cycle.py's default 1.4
  K/(W/m²) disagrees with PlanetParams' 0.8 default; callers all pass
  explicitly today, but the constant is a trap for new code.
- **Ocean carbonate chemistry.** A single well-mixed ocean CO2 reservoir now;
  a 2-box (surface/deep) split with a solubility pump would give realistic
  ~century-scale CO2 drawdown and make the ECS experiments more meaningful.
- **Milankovitch scenario runner.** Obliquity/eccentricity already exist in
  PlanetParams; a script sweeping them per the PLAN_PHYSICS ice-age recipe
  (low obliquity + low CO2 start) would be a strong validation showcase.

## Theme 5 — Performance

- **Profile-guided Numba pass.** The storm-anomaly loop, `_laplacian` chains
  in precipitation, and the Köppen reclassification are the remaining
  pure-NumPy hot spots at 512×1024.
- ~~**Float32 end-to-end audit.**~~ **Investigated 2026-07-04, not applied.**
  Wrapped the `np.gradient` call sites flagged by profiling
  (`evolve_wind`'s pressure-gradient terms, `generate_wind_field`'s
  elevation-slope/thermal-wind terms, `_evolve_temperature`'s cloud-fraction
  divergence/orographic terms) in explicit float32 casts. Two problems: (1) a
  quick isolated check showed `np.gradient` doesn't actually promote float32
  inputs to float64 on its own, so the premise only applies where an
  upstream array is unexpectedly float64 already — true for at least one
  site, since the casts measurably changed simulation output (caught by the
  new `test_golden_state.py`, ~0.1% drift in `co2_ocean` after 0.15
  simulated years); (2) three separate `scripts/benchmark_headless.py` runs
  on the same unchanged code varied 150–190s/year at 512×1024 DAILY — the
  environment's run-to-run noise floor is far larger than this change's
  theoretical few-ms/step benefit, so no improvement could be reliably
  measured either way. Reverted rather than keep an unproven behavior change;
  left here for whoever picks this up next to redo with steadier benchmarking
  conditions (e.g. a dedicated quiet machine, or profiling instruction counts
  instead of wall time).
- **Incremental render.** The GUI redraws the full RGB overlay each frame;
  dirty-region or double-buffer updates would cut UI cost at high resolution.
- **Adaptive substep count in evolve_wind.** 8 substeps are used regardless
  of dt or wind speed; a CFL-based count (2-8) would speed up calm periods.
- **Optimizer: multiprocess sweeps.** headless.py runs are embarrassingly
  parallel; a process-pool sweep runner would make Bayesian search practical
  overnight.

## Theme 6 — Validation & tooling

- **Reanalysis benchmark pack.** Monthly ERA5/CRU climatology (T, P, wind at
  ~2° resolution) as a versioned test fixture, with map-correlation scores in
  the diagnostics — moving beyond zonal-mean and landmark-sample checks.
- ~~**Conservation dashboards.**~~ **Done 2026-07-04.** `diagnostics.py` gained
  `area_weighted_global_mean()` and `compute_radiation_balance()`;
  `testing/test_conservation.py` gained `test_radiation_budget_near_equilibrium`
  (area-weighted TOA net radiation near zero at a 2-year-spinup state) and
  `test_ch4_equilibrium_holds_baseline` (CH4 natural-source/oxidation balance
  holds over 5 years). Freshwater (evap-precip) budget not yet covered — left
  as a follow-up.
- ~~**Golden-state regression tests.**~~ **Done 2026-07-04.**
  `scripts/generate_golden_state.py` + `testing/test_golden_state.py`, using
  `optimizer.headless.run_simulation` as the deterministic entry point at
  32×64/0.1yr spinup. Already proved its worth once during this same session
  (see the Theme 5 float32-audit entry above).
- **GUI smoke automation.** Headless Tk instantiation test (create window,
  render one frame, destroy) so GUI wiring bugs like the benchmark
  `nonlocal` issue (fixed 2026-07-03) surface in CI.
- ~~**Earth-constant audit test.**~~ **Done 2026-07-04** (moved here from
  Theme 3 for grouping with the other new test-infra items) —
  `testing/test_no_hardcoded_earth_constants.py`; also fixed the 5 remaining
  hardcoded `1013.25` hPa references (atmosphere.py/simulate.py Clausius-
  Clapeyron calls) and a hardcoded `3.0 * 365.25` ice-sheet-maturity threshold
  in simulate.py, both now sourced from `PlanetParams`.
- **Dead-parameter/wiring linter.** **Done 2026-07-04.**
  `testing/test_param_wiring.py`: runs a short simulation at a parameter's
  default vs. a substantially perturbed value and asserts the resulting state
  differs. Found a genuine new instance on first run —
  `simulate_step`'s `latent_cooling_coeff` is accepted but read nowhere in
  the codebase; documented in place as a deprecated no-op rather than newly
  wired (matches the existing `ocean_exchange_floor`/`span` convention).
- **Directional/sign regression tests.** **Done 2026-07-04.**
  `testing/test_derivative_signs.py`: unit-level convergence/divergence
  checks on `_moisture_convergence_numba`, an orographic windward-vs-leeward
  precipitation check, and a meridional-vs-zonal terrain-deflection check
  (regression for the `gx, gy = np.gradient(elev_c)` axis-swap bug) — all
  independent of any tuning constant, unlike the aggregate-metric tests that
  missed the original meridional-sign bug for months.

---

## Deliberately out of scope (for now)

- Full 3D atmosphere/ocean (GCM territory — wrong tool for an interactive sim)
- Sub-daily weather realism (individual thunderstorms, fronts)
- Human/land-use forcing scenarios
- Multiplayer/scripting APIs before the physics core stabilizes
