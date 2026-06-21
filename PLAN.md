# PlanetSim — Development Plan

> Last updated: 2026-06-20  
> Branch: claude_code_review → target: main

---

## Project Goals (restated)

1. Realistic Earth-calibrated climate simulation usable as a test bed
2. Adaptive time-scaling: 1 day/cycle (max resolution) → weeks → months → years, each increasingly approximate but stable
3. Correct multi-layer feedback loops (ice-albedo, CO2, vegetation, ocean-atmosphere)
4. Generalizability to non-Earth bodies (Mars, exoplanets) — no Earth-only constants buried in physics
5. Automated parameter optimization backend (headless runs, scoring, sweep/Bayesian search)

---

## Current State Summary

| Module | Lines | Status |
|--------|-------|--------|
| main.py | 1,397 | Mature GUI; some cache safety gaps |
| simulate.py | 1,680 | Core engine; has dead param `heat_transport_coeff` |
| atmosphere.py | 1,418 | Numba-accelerated; wind coupling weak |
| temperature.py | 495 | LUT-based; Earth-specific ocean_seasonal_frac |
| ocean.py | 500 | 1D zonal transport; AMOC/ACC hardcoded magnitudes |
| carbon_cycle.py | 571 | CO2 feedback param unused (`co2_climate_feedback`) |
| climate_averages.py | 300+ | 20-type Köppen; good |
| planet_params.py | 210 | Clean dataclass; could add ocean_fraction, rotation_direction |
| masks.py | 98 | Solid; fingerprint cache |
| diagnostics.py | 300+ | Earth reference comparisons; time-series export |
| **Testing** | 3,401 | 69 passed / 2 xfailed; comprehensive |

**Dead code / wiring gaps identified:**
- `heat_transport_coeff=0.8` passed to `_evolve_temperature` but never used inside it
- `co2_climate_feedback` parameter exists in simulate.py but is not applied
- `snow_depth` field computed but unclear if it feeds back into surface albedo
- Ocean AMOC/ACC bonus magnitudes are Earth-specific numbers with no scaling to planet params
- `temperature_kelvin_for_lat` uses Earth-empirical `ocean_seasonal_frac` constant
- Wind cache uses `id()` key without guaranteed GC safety across all call sites

---

## Phase 0 — Code Audit & Cleanup

**Goal:** Remove dead code, fix wiring gaps, standardize interfaces. No physics changes yet.

### Audit Results (2026-06-20)

- **Baseline:** 98 passed / 11 xfailed / 5 xpassed (memory was outdated; more tests were added)
- `co2_climate_feedback` — correctly wired; `co2_temp_offset` applied to T_base_land (line 469), T_base_ocean (line 602), T_base_ocean_full (line 642). ✓
- `snow_depth → albedo` — correctly wired; `snow_depth_coarse` passed to `_evolve_temperature` which computes `snow_cover = clip(snow_depth/0.1, 0, 1)` and applies it to surface albedo. ✓
- `_RELAX_CACHE` — uses a proper tuple key (h, w, day_of_year, 8 float params); not `id()`-based. ✓
- No TODO/FIXME comments anywhere in the codebase. ✓
- `heat_transport_coeff` — genuinely dead: accepted in `simulate_step` (line 244) and `_evolve_temperature` (line 1194), passed at line 898, but never read inside `_evolve_temperature`. ✗ → **FIXED**

**Earth-specific constants in ocean.py** (for Phase 3 parameterization):

| Location | Constant | Earth value | What it represents |
|----------|----------|-------------|-------------------|
| line 155 | NH Gaussian `exp(-((lat-45)/20)²)` | peak at 45°N, σ=20° | Gulf Stream / AMOC meridional profile |
| line 156 | `nh_subpolar_bonus = 0.22` | 22% bonus | AMOC subpolar intensification |
| line 157 | `sh_strength = 0.35` | flat 35% | ACC zonal (no meridional peak) |
| line 165 | `polar_damp` thresholds 80°, 15°, 0.2 | Earth-specific | Cutoff latitude for polar transport |
| line 176 | ice blocking `0.18` NH / `0.26` SH | Earth-specific | NH/SH asymmetric ice coverage effect |
| line 245 | `exchange_rate = 0.03` K/day | Earth-specific | Ocean-atmosphere sensible exchange rate |
| line 265 | `ekman_coefficient = 0.03` | 3% of wind | Wind → current coupling (scales with surface_pressure) |

### Tasks

- [x] **Remove `heat_transport_coeff` dead parameter** — removed from `simulate_step`, `_evolve_temperature`, and `testing/quick_diagnosis.py`
- [x] **Confirm `co2_climate_feedback` wiring** — verified correctly applied
- [x] **Confirm `snow_depth → albedo` path** — verified correctly wired
- [x] **Confirm `_RELAX_CACHE` safety** — verified uses tuple key, not `id()`
- [x] **Audit Earth constants in ocean.py** — documented in table above for Phase 3
- [x] **Check for orphaned TODO/FIXME** — none found
- [x] **Run baseline test suite** — 98 passed / 11 xfailed / 5 xpassed

---

## Phase 1 — Adaptive Time-Scaling Architecture

**Goal:** 1 day/step runs full physics. As steps grow, processes are parameterized rather than explicitly resolved. Each scale should be stable and feel live.

### Design

```
1 day/step   →  Weather scale    — all physics at full resolution
1 week/step  →  Synoptic scale   — sub-step 7× but cheaper wind (fewer sub-steps/day)
1 month/step →  Seasonal scale   — climatological mean wind, no storm events
1 year/step  →  Climate scale    — annual-mean forcing, ice sheets + CO2 only
```

Each scale removes the fastest processes and replaces them with their statistical effect:
- Weather→Synoptic: reduce wind sub-steps from 8 to 2; average daily precipitation
- Synoptic→Seasonal: swap prognostic wind for cached relaxation target; no daily ice oscillation
- Seasonal→Climate: skip seasonal cycle in temperature; evolve only slow variables (ice sheets, vegetation biomass, CO2 ppm)

### Tasks

- [ ] **Add `TimeScaleMode` enum** to simulate.py — values: `DAILY`, `WEEKLY`, `MONTHLY`, `ANNUAL`
- [ ] **Refactor `simulate_step` into dispatch** — new function selects appropriate physics path based on mode; existing code becomes the `DAILY` path
- [ ] **Implement WEEKLY path** — 7 sub-steps using existing code but with `wind_substeps=2` and precipitation averaged across the 7 days
- [ ] **Implement MONTHLY path** — use cached wind relaxation target (already exists as `_RELAX_CACHE`); parameterize daily precip as monthly-mean; ice evolves with monthly temperature; skip individual storm trigger logic
- [ ] **Implement ANNUAL path** — annual-mean insolation forcing; evolve only: ice sheet age, vegetation biomass, CO2 ppm, climate averages; temperature evolves toward radiative equilibrium with no weather noise
- [ ] **Update main.py speed dropdown** — replace current "1 Day / 1 Week / 1 Month" string options with enum-backed selection; hook to new dispatch
- [ ] **Stability test** — advance Earth benchmark 1 year via DAILY, WEEKLY, MONTHLY paths; verify final global mean T agrees within ±3K across all three

---

## Phase 2 — Layer Interaction Model Review

**Goal:** Audit every feedback loop. Confirm each is correctly wired, correctly scaled for different time steps, and has an enable/disable flag for testing.

### Feedback loop inventory

| Loop | Direction | Current Status | Action |
|------|-----------|----------------|--------|
| Ice-albedo | ice → albedo → T | ✅ Active | Verify scales correctly at MONTHLY/ANNUAL |
| Snow-albedo | snow_depth → albedo → T | ⚠️ Tracked but path unclear | Confirm or add coupling |
| AMOC/ACC dynamic | ice → circulation → T | ✅ Phase 6 (dynamic factors) | No change needed |
| CO2 greenhouse | CO2 ppm → epsilon → T | ❌ `co2_climate_feedback` unused | Phase 0 fix |
| Vegetation-albedo | biome → surface albedo | ✅ Via Köppen albedo table | Verify active every step |
| Ocean SST → evaporation | T_sst → q → precip | ✅ Partial | Audit coupling coefficient |
| Wind → Ekman → SST | wind → ocean drift → SST | ⚠️ 3% coupling, very weak | Consider strengthening in Phase 4 |
| Precip → soil → vegetation | P → soil_moisture → NPP | ✅ In carbon_cycle | No change |
| Ice → wind (pressure) | ice → surface albedo → pressure | ❌ Not modeled | Low priority; note as known gap |

### Audit Results (2026-06-20)

| Loop | Verified Status | Notes |
|------|-----------------|-------|
| Ice-albedo | ✅ Active | `sea_ice` → `albedo_sfc`; latent heat ΔT also applied |
| Snow-albedo | ✅ Active | `snow_depth_coarse` → `snow_cover` → `albedo_sfc` over land |
| AMOC/ACC dynamic | ✅ Active | `amoc_factor`/`acc_factor` computed from 60-75° ice cover → multiplied into bonuses |
| CO2 greenhouse | ✅ Active | `co2_temp_offset` applied to T_base_land + T_base_ocean + T_base_ocean_full |
| Vegetation-albedo | ✅ Active | `vegetation_albedo(biome, koppen_type)` → `albedo_veg` for land cells |
| Ocean SST → evaporation → precip | ✅ Active | T_sst (T_full) drives both `_evolve_temperature` evap and `generate_precipitation` trigger |
| Wind → Ekman → SST | ❌ Not wired | `compute_ekman_transport()` exists in ocean.py but is never called from `simulate_step` |
| Precipitation → soil → vegetation | ✅ Active | `soil_next` from `generate_precipitation` → `carbon_cycle_step` |

**Ekman transport gap**: `compute_ekman_transport` in ocean.py computes wind-driven surface currents but the result is not used anywhere in the simulation loop. The primary ocean-wind coupling is the 3% Ekman coefficient parameter in `get_major_ocean_currents` for visualization only. This is a known gap — ocean heat transport is parameterized, not wind-driven. Deferred to Phase 3 (planet generalization).

### Tasks

- [x] **Audit each loop** — completed; see table above
- [x] **Add `feedback_flags` dict to `simulate_step`** — `dict[str, bool] | None = None`; flags: `ice_albedo`, `snow_albedo`, `amoc_acc`, `co2_greenhouse`, `vegetation_albedo`, `ocean_transport`
- [ ] **Add targeted tests** — one test per major loop in `test_feedback_flags.py` (Phase 6)

---

## Phase 3 — Planet Generalization

**Goal:** No physics constants should be Earth-only values buried in formulas. Each should either live in `PlanetParams` or be derivable from values already there.

### Known Earth-specific constants to relocate

| File | Constant | What it represents | Proposal |
|------|----------|--------------------|----------|
| ocean.py | AMOC +18K NH bonus | Atlantic overturning strength | Scale with `rotation_rate × ocean_fraction`; make configurable in PlanetParams |
| ocean.py | ACC +28K SH upper ramp | Antarctic circumpolar current | Same as above |
| ocean.py | Ekman 3% coefficient | Wind → current coupling | Scale with `surface_pressure / 101325` |
| temperature.py | `ocean_seasonal_frac` | Ocean heat capacity seasonal damping | Express as function of `ocean_fraction` and `sidereal_day_hours` |
| atmosphere.py | baroclinic jet amplitude 1e6 | Mid-lat westerly strength | Scale with `surface_pressure × rotation_rate²` |
| carbon_cycle.py | CO2_PREINDUSTRIAL=280 | Earth pre-industrial CO2 | Move to PlanetParams as `co2_baseline_ppm` |
| carbon_cycle.py | CO2_CURRENT=415 | Earth current CO2 | Move to PlanetParams as `co2_initial_ppm` |

### PlanetParams extensions

- [ ] **Add `ocean_fraction: float = 0.71`** — fraction of surface covered by ocean; drives heat capacity, evaporation, Ekman scaling
- [ ] **Add `co2_baseline_ppm: float = 280.0`** — pre-industrial reference for radiative forcing formula
- [ ] **Add `co2_initial_ppm: float = 415.0`** — starting atmospheric CO2 for simulation
- [ ] **Add `rotation_direction: int = 1`** — +1 prograde (most planets), -1 retrograde (Venus); flips Coriolis sign
- [ ] **Add `has_liquid_water_ocean: bool = True`** — disables ocean heat transport model when False (Mars, dry planets)
- [ ] **Relocate all identified constants** from the table above into PlanetParams fields or derived properties
- [ ] **Fix xfailed test `test_mars_params_colder_than_earth`** — once AMOC/ACC scales with planet params, Mars polar warming should disappear; remove xfail or tighten threshold
- [ ] **Add Mars preset** to planet_params.py as `MARS = PlanetParams(...)` singleton

---

## Phase 4 — Parameter Optimization Backend

**Goal:** Automated headless simulation runner with a scoring function and search strategy. No GUI dependency. Can run many configurations in parallel to find optimal physics parameters.

### Architecture

```
optimizer/
  __init__.py
  scoring.py      — ClimateScore class, weighted metric comparison vs reference
  headless.py     — run_simulation(params, years) → ClimateMetrics (no GUI/threads)
  sweep.py        — grid_search(), latin_hypercube_sample(), random_search()
  bayesian.py     — optuna-based Bayesian optimization (optional, graceful fallback)
  runner.py       — CLI: python -m optimizer --mode sweep --config sweep_config.json
  results.py      — save/load/analyze result tables (CSV + JSON)
  configs/
    earth_params.json     — current best Earth parameters
    sweep_wind.json       — wind parameter sweep definition
    sweep_ocean.json      — ocean parameter sweep definition
```

### Scoring function (ClimateScore)

Computes a 0–100 score from a 2-year spinup + 1-year evaluation:

| Metric | Weight | Target (Earth) | Tolerance |
|--------|--------|----------------|-----------|
| Global mean T | 2.0 | 288K | ±2K full score |
| Equator-pole gradient | 1.5 | 45–60K | linear penalty outside range |
| ITCZ latitude | 1.0 | ±5° of equator | penalize displacement |
| NH ice fraction | 1.5 | ~5% of NH ocean | ±2% full score |
| SH ice fraction | 1.0 | ~7% of SH ocean | ±3% full score |
| Trade wind speed | 1.0 | 5–8 m/s at 10–20° | |
| Mid-lat wind speed | 1.0 | 6–10 m/s at 40–60° | |
| Seasonal amplitude NH | 1.0 | 35–50K at 50°N | |
| Global mean precip | 0.5 | 2.5–3.0 mm/day | |

### Tasks

- [ ] **Create `optimizer/` directory** with `__init__.py`
- [ ] **Implement `headless.py`** — wraps `simulate_multiple_steps()` with no GUI/thread overhead; takes `PlanetParams` + physics kwargs; returns final `PlanetState` + `ClimateDiagnostics`
- [ ] **Implement `scoring.py`** — `ClimateScore.score(state, diagnostics) → float`; each metric independently capped so one bad metric cannot dominate
- [ ] **Implement `sweep.py`** — `random_search(param_space, n_samples, n_jobs)` using `multiprocessing.Pool`; latin hypercube sampling via `scipy.stats.qmc.LatinHypercube`; results written to CSV incrementally
- [ ] **Implement `runner.py`** — CLI with `--mode {sweep,bayes,single}`, `--config CONFIG_JSON`, `--output DIR`, `--jobs N`, `--spinup-years N`, `--eval-years N`
- [ ] **Implement `bayesian.py`** — optuna `TPESampler` with median pruner (stop runs that diverge in year 1); graceful `ImportError` fallback to random search if optuna not installed
- [ ] **Implement `results.py`** — save/load/sort/plot result tables; `top_n(df, n)` returns best configurations
- [ ] **Add `configs/earth_params.json`** — capture current best physics parameter values as a baseline
- [ ] **Add test `test_optimizer.py`** — confirm headless run matches threaded run result; confirm Earth baseline scores ≥ 65/100

---

## Phase 5 — Performance Audit

**Goal:** Identify and fix the actual bottlenecks. Ensure each time-scale mode maintains a responsive UI (frame delivered < 100ms).

### Known performance characteristics
- Numba JIT active: advection, diffusion, friction, Coriolis kernels
- Wind evolution: 8 sub-steps at full `wind_block_size`-downsampled resolution
- Temperature evolution: block_size=3 downsample → full upsample
- Particle animation: 50ms refresh, up to 60K particles
- Ocean heat transport: cached every 30 days

### Tasks

- [ ] **Profile `simulate_step` with cProfile** — identify top-3 CPU consumers; output flame graph or table
- [ ] **Reduce particle count at MONTHLY/ANNUAL modes** — wind particles are decorative; halve or disable them at large time steps for faster UI refresh
- [ ] **Check array layout for Numba kernels** — confirm all arrays passed to Numba are C-contiguous; add `.ascontiguousarray()` guards where missing
- [ ] **Set a headless benchmark target** — 1 simulated year in headless mode should complete in < 90 seconds on a modern CPU (baseline: measure current)
- [ ] **Add `--profile` flag to optimizer runner** — lets user collect per-trial timing without modifying source

---

## Phase 6 — Testing Expansion

**Goal:** Cover all new functionality added in Phases 1–5.

### Tasks

- [ ] **`test_time_scaling.py`** — advance Earth 365 days via DAILY, WEEKLY, MONTHLY; global mean T must agree within ±5K; no NaN/Inf in any path
- [ ] **`test_planet_generalization.py`** — Mars preset runs 100 steps without NaN; Mars global mean < Earth global mean; rotation_direction=-1 (Venus-like) flips wind direction
- [ ] **`test_feedback_flags.py`** — disable ice-albedo feedback; confirm Arctic T rises > 5K; re-enable; confirm it returns to baseline
- [ ] **`test_optimizer_scoring.py`** — Earth baseline state scores ≥ 65/100; random PlanetParams scores significantly lower
- [ ] **`test_headless.py`** — headless 30-step run matches threaded 30-step run result (same numpy seed) within float tolerance
- [ ] **Update xfailed tests** — after Phase 3 planet generalization, attempt to unskip `test_mars_params_colder_than_earth` and `test_high_obliquity_larger_seasonal_range`

---

## Execution Order

```
Phase 0  →  Phase 1  →  Phase 2
                ↓
           Phase 3  →  Phase 4
                ↓
           Phase 5  →  Phase 6 (runs in parallel with each phase)
```

Phase 0 must come first (cleans up interfaces everything else depends on).
Phases 1 and 2 are the core simulation work and build on each other.
Phase 3 (planet generalization) depends on Phase 2 (feedback audit gives clearer picture of what's Earth-specific).
Phase 4 (optimizer) depends on Phase 1 (headless needs multi-scale paths) and Phase 3 (needs to support non-Earth scoring).
Phase 5 (performance) is most useful after Phase 1 because the new paths may introduce new bottlenecks.
Phase 6 (testing) runs incrementally alongside each phase.

---

## Deferred / Out of Scope (for now)

- **Stratosphere / upper atmosphere** — single-layer model is intentional for performance; document the gap
- **3D ocean** — zonal-mean 1D transport is the chosen approximation; real OGCM would require order-of-magnitude more compute
- **Lightning / wildfire dynamics** — carbon_cycle has stubs; defer until biome model is mature
- **Cloud microphysics** — cloud_cover prognostic but no precipitation↔cloud feedback loop; known gap
- **Continent topology-aware gyres** — ocean currents use topology where land elevation is available but not fully geometry-driven

---

## Open Questions

1. Should MONTHLY mode use the cached relaxation-target wind or a monthly-mean of daily-evolved wind? (Cached is faster; daily-mean is more accurate but requires sub-stepping)
2. Should the optimizer scoring function be planet-agnostic (relative to a supplied reference) or Earth-specific? — Recommend: relative to a configurable `ReferenceClimate` object so it can later target Mars or exoplanet observations
3. What is an acceptable score threshold for "Earth-like enough"? Proposed: 65/100 for basic correctness, 80/100 as a stretch goal
4. Should the ANNUAL time scale still update Köppen classification each step, or only at the end of each simulated year? — Recommend: end of each simulated year only (classification needs 12 monthly means)
