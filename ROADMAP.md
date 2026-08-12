# PlanetSim — Roadmap

> Created: 2026-07-03
> This document is the long-horizon idea backlog: where the simulation can go
> next, roughly ordered by leverage within each theme. Nothing here is
> committed work. `PLAN.md` is historical; use
> `docs/PRODUCT_SCOPE_AND_EXPERIMENTS.md` for the supported scope and current
> experimental decisions.

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

- **Land seasonal-cycle SHAPE — open, cause unknown (added 2026-08-02).** Land at
  25–50° spends **7.00 months above its own annual mean** (a sinusoid gives 6.00;
  ocean at the same latitudes gives 6.31–6.72), rising to 7.97 on the 23.8yr
  state — a broad warm plateau with a narrow deep winter trough. This blocks
  Köppen `Cfc` outright (needs <4 months >10 °C) and, with the winter cliff,
  pushes maritime mid-latitudes out of the C group entirely, so it is a direct
  cause of the model emitting no Mediterranean climate. The *ceiling* half of the
  problem is solved and documented (ACCURACY_AUDIT.md **C1b** — `_land_cap_1d`'s
  hard clamp), but the shape is **independent of it**: squareness stayed
  6.99–7.00 in the very run where the ceiling fraction fell 42.0% → 16.8%.
  Refuted: winter trough depth, the evap-cooling seasonal gate, snow/ice albedo,
  resolution quantization (see C1b for the numbers — don't re-derive these).
  **Best lead:** it grows with integration time (7.00 at 4yr → 7.97 at 23.8yr),
  so the mechanism has multi-year memory; test soil moisture first by holding it
  fixed over a long run and re-measuring.
  > **Superseded 2026-08-03 — cause found, and the soil-moisture lead was the
  > wrong trail** (ACCURACY_AUDIT.md **C1b-2026-08-03**). Traced stage by stage,
  > the radiative base is already fine (6.44–6.61, inside the ocean's own band);
  > the squareness is manufactured by **two** clamps, `_land_cap_1d` *and* the
  > excess-proportional evapotranspiration cooling, in roughly equal measure.
  > More importantly the binding error is the annual **mean** — the forcing runs
  > +21 K at 41°N — which is why the four knobs shipped here, all amplitude-side,
  > could never move it. Three new gated mechanisms take squareness **7.00 →
  > 6.00**, but cost ~0.5pp of H10 group accuracy at both 3 yr and a converged
  > 14 yr, so they ship inert. Next target is the **25–45°N winter warm bias**
  > (8–10 K), which is what remains after the shape is fixed.
  > **Corrected 2026-08-04 (ACCURACY_AUDIT.md C1b-2026-08-04): that "next
  > target" was mostly the scoreboard.** The 8–10 K figure came from
  > `land_seasonal_cycle`, whose level anchors describe mid-continental stations
  > while the metric area-averages *all* land in the band (25-35N land is 54%
  > arid subtropics). Scored against Köppen's own definitional thresholds
  > instead — the new, anchor-free `koppen_temperature_thresholds` — **100%** of
  > 25-35N reference-temperate land is correctly placed. The real residual is
  > ~22% of 35-45N being too warm *and* 99.5% of 45-55N maritime land being too
  > cold: two signs inside one zone, which no latitude-only term can fix. A
  > continentality mechanism was built for it and is a measured negative result
  > (`land_transport_maritime_decay`, inert) — at 35-45N the reference-D land is
  > neither more continental nor higher than the reference-C land.
  > **Superseded 2026-08-04, later the same day (ACCURACY_AUDIT.md
  > C1b-2026-08-04b): that negative result was itself a population artifact,
  > and the mechanism now ships enabled.** "D land is not more continental"
  > (0.312 vs 0.310) was measured by re-deriving the Köppen reference at the
  > model's 32×64 *forcing* grid — 35 C cells and 15 D cells for the whole band
  > — while the metric scores the 128×256 grid. On the scored population the two
  > separate by **0.84 sd**. Two fixes made it reachable: computing the maritime
  > field at native resolution (the coarse block mask inflates land 34% → 48% of
  > area and erases coastal contrast) and making it **anisotropic** — westerly
  > flow means the ocean that moderates a winter continent is the one *upwind*,
  > and held isotropic the mechanism is still net-negative. Shipped
  > `land_transport_maritime_decay` 0.0 → 1.0 and `land_transport_upwind_ratio`
  > 1.0 → 32.0, with a winter weight. Confirmed at 64×128 / 128×256 / 256×512:
  > H10 group +0.7 to +1.0pp, coldest-month threshold accuracy +1.9 to +2.3pp,
  > the worst zone (40-50N) **+14 to +18pp**, warmest month flat. Both headline
  > defects roughly halve, and reference-C and reference-D improve *together*,
  > which is what distinguishes contrast from a level shift.
  > **Followed up 2026-08-05 (ACCURACY_AUDIT.md C1b-2026-08-05), and the entry
  > above turns out to have been half the mechanism.** The winter gate that
  > mechanism needs is a symptom: it scales an additive *bonus*, so year-round it
  > warms maritime summers — wrong sign — and therefore cannot reach the
  > maritime-*summer* error at all (81.5% of −50:−40 reference-C land has a
  > warmest month above the 22 °C its class requires). The root cause is one
  > layer up: `temperature_kelvin_for_lat` returns instantaneous radiative
  > equilibrium, with an annual half-range of ~81 K at 41° against Earth land's
  > ~28 K, and **the land branch never had a thermal-inertia term at all** while
  > the ocean branch has damped its own swing since the model's early days.
  > Shipped `land_seasonal_amplitude` 1.0 → 0.75 (the missing damping, exactly
  > mean-preserving), `land_transport_gain` 1.0 → 0.5 (the trapezoids were sized
  > against an *undamped* winter) and `land_seasonal_amplitude_maritime`
  > 0.0 → 0.45 (the same continentality field, applied to the amplitude, which
  > has the right sign in both seasons and needs no gate). **Strictly dominant at
  > 64×128 / 128×256 / 256×512 — every tracked metric improves, nothing traded.**
  > Warmest-month threshold accuracy +4.2 / +4.5 / +3.0pp, the largest single
  > move any C1b session has produced.
  > **Corrected 2026-08-05 (ACCURACY_AUDIT.md C1b-EVAP): "squareness is
  > manufactured by *two* clamps" — the claim this entry has carried since
  > 2026-08-03 — is wrong, and so is the next target it implied.** There is one
  > clamp. The evapotranspiration cooling is applied on the line *before*
  > `_land_cap_1d`, and `min(T − cooling, cap)` does not depend on `cooling`
  > while the result is still above the cap, so **99.4% of the term is absorbed**
  > and removing it entirely leaves group accuracy, both threshold accuracies and
  > squareness bit-identical. The +0.66 of squareness attributed to it was read
  > from a *forcing* stage the next line overwrites — now process note 23.
  > Two real defects in the term were found and fixed as exact no-ops: its
  > threshold and coefficient were hardcoded (290 K silently disables it on Mars),
  > and the contraction **inverted** above `evap_cooling_strength` ≈ 1.18, where
  > it removed more than the whole excess and a hotter cell came out colder. The
  > square wave *is* removable from here (cycle error 5.44 → 2.77 at 128×256,
  > flat top essentially gone) but every route costs H10 group accuracy, and
  > **Cfc — the named falsifier — stays 0.00% at 128×256 throughout**. A
  > shape-only `evap_cooling_amplitude` built to dodge the trade improves every
  > bounded metric at 64×128 and **reverses sign at 128×256** (note 14's third
  > instance). All three knobs ship inert. The real clamp, `_land_cap_1d`, binds
  > on 45.1% of land-months and is latitude-only and moisture-blind — it caps the
  > Sahara at 27.9 °C against a real ~35 °C July mean — and for the first time the
  > overshoot it must absorb is small (35-45°N band-mean peak overshoot −0.19 K,
  > against +11.54 K before this week), which is the condition
  > `land_cap_softness_k` has always needed and never had.
  > **Scope narrowed 2026-08-03 (ACCURACY_AUDIT.md A6).** "A direct cause of the
  > model emitting no Mediterranean climate" is now only half true. **Csa was
  > recovered from precipitation alone** (0.01% → 1.93% of land, against Earth's
  > 1.94%) by giving the subtropical dry belt the seasonal migration it never
  > had, with this temperature defect entirely unfixed. What still rests on this
  > item is **Csb** (0.42% vs 1.16% — needs a warmest month under 22 °C) and
  > **Cfc** (0.00% vs 0.31%). Still worth doing, and the lead above is still the
  > lead; the prize is ~1.05% of land, not the whole Mediterranean.

- ~~**1.5-layer atmosphere.**~~ **Done 2026-07-04.** `atmosphere.evolve_wind_aloft()`
  gives the atmosphere a real, independent prognostic upper-level wind layer
  (own advection/Coriolis/PGF momentum budget, weak Rayleigh friction, no
  terrain/storm/Rossby terms), coupled back to the surface via a real
  per-cell, direction-sensitive relaxation term (replacing the old
  magnitude-only `|dT/dy|` hack). Deliberately additive: Rossby waves,
  discrete storms/trade waves, the meander-index/blocking state machine, and
  the 3-cell surface relaxation are all unchanged for this pass. **Follow-up
  (next item here):** now that a real vertical-shear mechanism exists,
  revisit whether any of those prescribed mechanisms can be weakened or
  retired — this is the validation-driven reassessment the original bullet
  called for, deferred until the new layer has more runtime/calibration
  behind it. See PLAN.md's 2026-07-04 entry for the full implementation
  writeup, including a real physics finding along the way: the model's pure
  PGF+Coriolis+friction dynamics don't organically produce realistic surface
  westerlies at all (confirmed by disabling the 3-cell relaxation) — the
  upper layer needed the *opposite* sign convention from the surface's
  thermal PGF term (warm column = higher upper-level pressure, not lower) to
  produce a correctly-signed jet.
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
- ~~**Prognostic AMOC strength.**~~ **Temperature term shipped 2026-08-10, default off.**
  AMOC now responds to both salinity (`salinity_amoc_scale`, on by default) and a real North
  Atlantic temperature anomaly (`temperature_amoc_scale`, off by default pending a real-terrain
  calibration pass -- ocean temperature has no restoring force pinning it near a reference, unlike
  salinity). Enables freshwater-hosing/ice-age experiments once calibrated. See FEATURES.md item 5.
  Still open: deriving `amoc_bonus_near`/`amoc_bonus_far` from an actual overturning-strength
  calculation instead of prescribed constants (`docs/ACCURACY_AUDIT.md` D2) -- a substantially
  larger undertaking, out of scope for the temperature-term pass.
- ~~**Mixed-layer depth map.**~~ **Latitude-dependent depth shipped 2026-08-10, default reproduces
  prior behavior exactly.** `PlanetParams.mixed_layer_depth_tropical_m`/`_polar_m` now expose what
  was a hardcoded 30m/200m ramp in `_evolve_temperature`'s T_sst relaxation, and a second,
  physically-derived `ocean_seasonal_frac` mode (`derive_ocean_seasonal_lag`, off by default) lets
  that same mixed-layer depth drive the previously-independent hand-tuned seasonal-lag polynomial
  in the `T_base_ocean` path. See FEATURES.md item 6. Map-based (non-latitude-only) depth is still
  a future step.
- ~~**Ocean CO2 uptake with proper piston velocity.**~~ **Done.** `carbon_cycle.ocean_co2_flux`
  now consumes `state.wind_speed_avg`, a rolling EMA over
  `PlanetParams.co2_wind_averaging_days` (default 30d) maintained in `simulate.py`, instead of
  the instantaneous per-step wind speed. See `docs/ACCURACY_AUDIT.md` D4.

## Theme 3 — Planet generalization (toward random planets)

- **Planet parameter audit test.** A pytest that greps physics modules for
  numeric literals matching Earth constants (6371, 365, 1013.25, 288.15…)
  outside PlanetParams/EARTH definitions, with an allowlist. This is how the
  hardcoded storm radius and `% 365` cache bugs (both fixed 2026-07-03) would
  have been caught automatically.
- **Manual audit performed 2026-07-27** (canvas-review Phase 4 item), ranked
  by fix priority at the time — this is the inventory that audit test above
  should encode. The HIGH items have since been fixed (see below); only the
  LOW item remains open:
  - **DONE (fixed 2026-08-02, commit `d1c08b0`) — max terrain elevation and
    lapse rate were hardcoded at Earth values (8848m Everest; 6.5 K/km) across
    `temperature.elevation_to_alt_km`, `climate_averages.compute_biome_type`,
    `main.py`, and `simulate._evolve_temperature`.** Added
    `PlanetParams.max_elevation_km: float = 8.848` and
    `PlanetParams.lapse_rate_k_per_km: float = 6.5` (`planet_params.py:95,110`,
    Earth defaults are exact no-ops), threaded through all call sites above,
    with `MARS` set to `max_elevation_km=21.9, lapse_rate_k_per_km=2.5`
    (`planet_params.py:3677,3680`). Regression-guarded by
    `testing/test_no_hardcoded_earth_constants.py`.
  - **LOW — `0.622` (Rd/Rv epsilon, water-vapor physics) hardcoded in four
    sites** (`atmosphere.py` x2, `simulate.py` x2). This ratio depends on
    atmosphere composition (M_water/M_dry_air); Mars's CO2 atmosphere gives
    a genuinely different value (~0.41 vs Earth's 0.622). Low priority in
    practice because `MARS.has_liquid_water_ocean=False` already gates
    essentially all humidity/water-cycle physics off for Mars in this model
    — this constant is currently inert for the one non-Earth preset that
    exists, so fixing it has no visible effect until a water-bearing
    exoplanet preset exists to exercise it.
  - No hardcoded gravity constant (9.81 m/s²) was found anywhere in
    `ocean.py`/`atmosphere.py`/`carbon_cycle.py` — Ekman/wind-driven ocean
    physics doesn't reference gravity explicitly, so there is no equivalent
    gap there.
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
- ~~**CLIMATE_SENSITIVITY constant cleanup.**~~ **Done.** `carbon_cycle.CLIMATE_SENSITIVITY`
  is now `0.8`, matching `PlanetParams.co2_climate_feedback`'s default (see the comment at
  `carbon_cycle.py:145`).
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

- ~~**Reanalysis benchmark pack.**~~ **Done 2026-08-09.** CRU TS v4.10 (T, P) and
  NCEP/NCAR Reanalysis 1 (wind) monthly climatologies, area-conservative regridding
  and map-correlation scoring (`monthly_climatology.py`), wired into
  `run_real_terrain_validation.py` (`--monthly-climatology`/`--wind-climatology`)
  and the optimizer, with enforced pytest regression gates. See FEATURES.md item 4
  and `docs/MONTHLY_CLIMATOLOGY_REFERENCE.md` for the full writeup.
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
